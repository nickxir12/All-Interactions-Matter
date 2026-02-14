import logging
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import (ReduceLROnPlateau,
                                      CosineAnnealingWarmRestarts,
                                      SequentialLR, _LRScheduler)
from tqdm import tqdm

from ...utils import MetricsTop, dict_to_str

logger = logging.getLogger('MMSA')


class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, initial_lr=None, last_epoch=-1, verbose=False):
        self.target_lr = [g['lr'] for g in optimizer.param_groups]
        self.initial_lr = [initial_lr for _ in self.target_lr] if initial_lr is not None else [0.0 for _ in self.target_lr]
        self.warmup_steps = max(1, int(warmup_steps))
        #FIND ME 
        try:
            super().__init__(optimizer, last_epoch, verbose)
        except TypeError:
            super().__init__(optimizer, last_epoch)


    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            inc = [(t - i) / self.warmup_steps for t, i in zip(self.target_lr, self.initial_lr)]
            return [i + d * self.last_epoch for i, d in zip(self.initial_lr, inc)]
        return self.target_lr

    def step(self, epoch=None):
        epoch = self.last_epoch + 1 if epoch is None else epoch
        self.last_epoch = epoch
        new_lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, new_lrs):
            group['lr'] = lr
        self._last_lr = new_lrs


class UNI_ENC():
    """
    Trainer for the UNI_ENC (audio-only or vision-only).
    Only uses the requested modality from the dataloader batches.
    """
    def __init__(self, args):
        self.args = args
        self.scheduler_cfg = args.get("scheduler", False)

        # criterion
        if args.train_mode == 'regression':
            self.criterion = nn.L1Loss()  # or nn.MSELoss()
        elif args.train_mode == 'binary':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

        # for convenience
        self.modality = args.get("uni_enc", {}).get("modality", "audio").lower()

    # ---- helpers ----
    def _prepare_labels(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Shape/dtype fixes depending on train_mode and head outputs.
        """
        if self.args.train_mode == 'classification':
            # CE expects labels: Long [B], logits: [B, C]
            labels = labels.view(-1).long()
        elif self.args.train_mode == 'binary':
            # BCEWithLogits expects float [B,1]
            labels = labels.view(-1, 1).to(dtype=logits.dtype)
        else:
            # regression expects float [B,1]
            labels = labels.view(-1, 1).to(dtype=logits.dtype)
        return labels

    # ---- main loops ----
    def do_train(self, model, dataloader, return_epoch_results=False):
        optimizer = optim.AdamW(model.parameters(), lr=self.args.learning_rate)

        if self.scheduler_cfg is False:
            print("------- Ongoing with the ReduceLROnPlateau -----------")
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, verbose=True, patience=self.args.patience
            )
        else:
            print("------- Ongoing with the LR with Warmup -----------")
            steps_per_epoch = int(len(dataloader['train']) / self.args.update_epochs)
            warmup_steps = steps_per_epoch * self.args['scheduler']['warmup_epochs']
            if self.args['scheduler'].get('T0', 0) > 0:
                restart_T = (self.args['scheduler']['T0']) * steps_per_epoch
                print(f"----> using restart period = {restart_T}")
            else:
                restart_T = (self.args['max_epochs'] + 1) * len(dataloader['train']) - warmup_steps

            scheduler_steplr = CosineAnnealingWarmRestarts(
                optimizer, eta_min=1e-7, last_epoch=-1, T_0=restart_T
            )
            scheduler_warmup = LinearWarmupScheduler(optimizer, warmup_steps)
            scheduler = SequentialLR(optimizer, [scheduler_warmup, scheduler_steplr], milestones=[warmup_steps])

        # init tracking
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {'train': [], 'valid': [], 'test': []}
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        while True:
            epochs += 1
            model.train()
            y_pred, y_true = [], []
            train_loss = 0.0
            left_epochs = self.args.update_epochs

            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1

                    # pick only selected modality
                    vision = batch_data['vision'].to(self.args.device)
                    audio  = batch_data['audio'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)

                    logits = model(None, audio, vision)  # UNI_ENC returns logits (B,C) from chosen modality
                    labels = self._prepare_labels(logits, labels)

                    loss = self.criterion(logits, labels)
                    loss.backward()

                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_(
                            [p for p in model.parameters() if p.requires_grad],
                            self.args.grad_clip
                        )

                    train_loss += loss.item()
                    y_pred.append(logits.detach().cpu())
                    y_true.append(labels.detach().cpu())

                    if not left_epochs:
                        optimizer.step()
                        if self.scheduler_cfg is not False:
                            scheduler.step()
                        left_epochs = self.args.update_epochs

                if not left_epochs:
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)

            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> loss: {round(train_loss, 4)} {dict_to_str(train_results)}"
            )

            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            if self.scheduler_cfg is False:
                scheduler.step(val_results['Loss'])

            # save best
            isBetter = (cur_valid <= best_valid - 1e-6) if min_or_max == 'min' else (cur_valid >= best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                print(f"********* Saving Model at {self.args.model_save_path}")
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)

            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0

        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {"Feature_m": []}  # unimodal logits

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio  = batch_data['audio'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)

                    logits = model(None, audio, vision)
                    labels = self._prepare_labels(logits, labels)

                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        features["Feature_m"].append(logits.cpu().numpy())
                        all_labels.extend(labels.cpu().detach().tolist())
                        sample_results.extend(logits.cpu().numpy().squeeze())

                    loss = self.criterion(logits, labels)
                    eval_loss += loss.item()
                    y_pred.append(logits.cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results
