"""
Tri-modal transformer encoder trainer that jointly trains:
1. Audio-only encoder
2. Vision-only encoder  
3. Audio-Visual bimodal encoder
"""

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
        # target_lr is derived from each parameter group in the optimizer
        self.target_lr = [group['lr'] for group in optimizer.param_groups]

        # If initial_lr is provided, use it; otherwise, default to 0.0 for each group
        if initial_lr is not None:
            self.initial_lr = [initial_lr for _ in self.target_lr]
        else:
            self.initial_lr = [0.0 for _ in self.target_lr]

        self.warmup_steps = warmup_steps
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Compute the learning rate for each parameter group
            lr_increment = [(target - initial) / self.warmup_steps for target, initial in zip(self.target_lr, self.initial_lr)]
            return [initial + increment * self.last_epoch for initial, increment in zip(self.initial_lr, lr_increment)]
        else:
            # Warmup complete, use target learning rates
            return self.target_lr

    def step(self, epoch=None):
        # Overriding step to allow updates per batch (or "step") instead of per epoch
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # Update the learning rate for each parameter group and keep track of the new rates
        new_lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr

        # Update the _last_lr attribute with the new learning rates
        self._last_lr = new_lrs


class TRI_ENC():
    """
    Trainer for the TRI_ENC model that jointly trains audio, vision, and audiovisual encoders
    with weighted loss combination.
    """
    def __init__(self, args):
        self.args = args
        self.scheduler_cfg = args.get("scheduler", False)
        
        # Base criterion - match BI_ENC exactly
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
            
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        
        # Get loss weights from config
        tri_cfg = args.get("tri_enc", {})
        self.lambda_audio = tri_cfg.get("lambda_audio", 1.0)
        self.lambda_vision = tri_cfg.get("lambda_vision", 1.0)
        self.lambda_bimodal = tri_cfg.get("lambda_bimodal", 1.0)
        
        print(f"TRI_ENC Loss weights: Audio={self.lambda_audio}, Vision={self.lambda_vision}, Bimodal={self.lambda_bimodal}")

    def _compute_combined_loss(self, outputs: dict, labels: torch.Tensor) -> dict:
        """
        Compute weighted combined loss from all three encoder outputs.
        
        Args:
            outputs: Dict with 'audio', 'vision', 'bimodal' logits
            labels: Ground truth labels
            
        Returns:
            Dict with total_loss and individual loss components
        """
        # Prepare labels inline like BI_ENC (no helper function)
        if self.args.train_mode == 'classification':
            labels_formatted = labels.view(-1).long()
        else:
            labels_formatted = labels.view(-1, 1)
        
        # Compute individual losses
        loss_audio = self.criterion(outputs['audio'], labels_formatted)
        loss_vision = self.criterion(outputs['vision'], labels_formatted)
        loss_bimodal = self.criterion(outputs['bimodal'], labels_formatted)
        
        # Weighted combination
        total_loss = (self.lambda_audio * loss_audio + 
                     self.lambda_vision * loss_vision +
                     self.lambda_bimodal * loss_bimodal)
        
        return {
            'total_loss': total_loss,
            'loss_audio': loss_audio,
            'loss_vision': loss_vision,
            'loss_bimodal': loss_bimodal
        }

    def do_train(self, model, dataloader, return_epoch_results=False):
        optimizer = optim.AdamW(model.parameters(), lr=self.args.learning_rate)
        
        if self.scheduler_cfg is False:
            print(f"------- Ongoing with the ReduceLROnPlateau -----------")
            scheduler = ReduceLROnPlateau(optimizer,
                                        mode='min',
                                        factor=0.1,
                                        verbose=True,
                                        patience=self.args.patience)
        else:
            print(f"------- Ongoing with the LR with Warmup -----------")
            steps_per_epoch = int(len(dataloader['train']) / self.args.update_epochs)
            warmup_steps = steps_per_epoch * self.args['scheduler']['warmup_epochs']
            if self.args['scheduler'].get('T0', 0) > 0:
                restart_T = (self.args['scheduler']['T0']) * steps_per_epoch
                print(f"----> using restart period = {restart_T}")
            else:
                restart_T = (self.args['max_epochs']+1) * len(dataloader['train']) - warmup_steps
            scheduler_steplr = CosineAnnealingWarmRestarts(
                optimizer,
                eta_min=1e-7,
                last_epoch=-1,
                T_0=restart_T,
            )
            scheduler_warmup = LinearWarmupScheduler(
                optimizer,
                warmup_steps,
            )
            scheduler = SequentialLR(
                optimizer,
                [scheduler_warmup, scheduler_steplr],
                milestones=[warmup_steps]
            )
            
        # Initialize results
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        
        while True:
            epochs += 1
            # train
            model.train()
            train_losses = {
                'total': 0.0,
                'audio': 0.0,
                'vision': 0.0,
                'bimodal': 0.0
            }
            
            # For metrics computation, we'll use bimodal predictions as primary
            y_pred, y_true = [], []
            left_epochs = self.args.update_epochs
            
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    # grad accumulation
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)

                    # Forward pass - get all three outputs
                    outputs = model(None, audio, vision)
                    
                    # Prepare labels inline like BI_ENC
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    
                    # Compute combined loss
                    loss_dict = self._compute_combined_loss(outputs, labels)
                    loss = loss_dict['total_loss']
                    
                    # Backward pass
                    loss.backward()
                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], self.args.grad_clip)
                    
                    # Store losses
                    train_losses['total'] += loss.item()
                    train_losses['audio'] += loss_dict['loss_audio'].item()
                    train_losses['vision'] += loss_dict['loss_vision'].item() 
                    train_losses['bimodal'] += loss_dict['loss_bimodal'].item()
                    
                    # For metrics, use bimodal predictions (you could also ensemble)
                    y_pred.append(outputs['bimodal'].cpu())
                    y_true.append(labels.cpu())
                    
                    if not left_epochs:
                        optimizer.step()
                        if self.scheduler_cfg is not False:
                            scheduler.step()
                        left_epochs = self.args.update_epochs
                        
                if not left_epochs:
                    optimizer.step()
                    
            # Average losses
            for key in train_losses:
                train_losses[key] = train_losses[key] / len(dataloader['train'])
            
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] >> "
                f"total_loss: {round(train_losses['total'], 4)} "
                f"(a: {round(train_losses['audio'], 4)}, "
                f"v: {round(train_losses['vision'], 4)}, "
                f"av: {round(train_losses['bimodal'], 4)}) "
                f"{dict_to_str(train_results)}"
            )
            
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            if self.scheduler_cfg is False:
                scheduler.step(val_results['Loss'])
                
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                print(f"***********************************************Saving Model at {self.args.model_save_path}")
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
                
            # epoch results
            if return_epoch_results:
                train_results["Loss"] = train_losses['total']
                train_results["Loss_Audio"] = train_losses['audio']
                train_results["Loss_Vision"] = train_losses['vision']
                train_results["Loss_Bimodal"] = train_losses['bimodal']
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
                
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        eval_losses = {
            'total': 0.0,
            'audio': 0.0,
            'vision': 0.0,
            'bimodal': 0.0
        }
        
        # For metrics, use bimodal predictions as primary
        y_pred, y_true = [], []
        
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_a": [],   # audio features
                "Feature_v": [],   # vision features  
                "Feature_av": [],  # bimodal features
            }
            # Store all three prediction types
            predictions = {
                "audio_preds": [],
                "vision_preds": [],
                "bimodal_preds": []
            }
            
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    
                    # Forward pass
                    outputs = model(None, audio, vision)
                    
                    # Prepare labels inline like BI_ENC
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    
                    # Compute losses
                    loss_dict = self._compute_combined_loss(outputs, labels)
                    
                    eval_losses['total'] += loss_dict['total_loss'].item()
                    eval_losses['audio'] += loss_dict['loss_audio'].item()
                    eval_losses['vision'] += loss_dict['loss_vision'].item()
                    eval_losses['bimodal'] += loss_dict['loss_bimodal'].item()
                    
                    # For metrics, use bimodal predictions
                    y_pred.append(outputs['bimodal'].cpu())
                    y_true.append(labels.cpu())
                    
                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        
                        # Store features (these would be the encoded representations)
                        # Note: TRI_ENC model would need to return features if you want them
                        features["Feature_a"].append(outputs['audio'].cpu().detach().numpy())
                        features["Feature_v"].append(outputs['vision'].cpu().detach().numpy()) 
                        features["Feature_av"].append(outputs['bimodal'].cpu().detach().numpy())
                        
                        all_labels.extend(labels.cpu().detach().tolist())
                        
                        # Store all prediction types
                        predictions["audio_preds"].extend(outputs['audio'].cpu().detach().numpy().squeeze())
                        predictions["vision_preds"].extend(outputs['vision'].cpu().detach().numpy().squeeze())
                        predictions["bimodal_preds"].extend(outputs['bimodal'].cpu().detach().numpy().squeeze())
                        
                        # Primary results use bimodal predictions
                        sample_results.extend(outputs['bimodal'].cpu().detach().numpy().squeeze())
                        
        # Average losses  
        for key in eval_losses:
            eval_losses[key] = eval_losses[key] / len(dataloader)
            
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_losses['total'], 4)
        eval_results["Loss_Audio"] = round(eval_losses['audio'], 4)
        eval_results["Loss_Vision"] = round(eval_losses['vision'], 4)
        eval_results["Loss_Bimodal"] = round(eval_losses['bimodal'], 4)
        
        logger.info(
            f"{mode}-({self.args.model_name}) >> "
            f"total_loss: {eval_results['Loss']} "
            f"(a: {eval_results['Loss_Audio']}, "
            f"v: {eval_results['Loss_Vision']}, "
            f"av: {eval_results['Loss_Bimodal']}) "
            f"{dict_to_str({k: v for k, v in eval_results.items() if not k.startswith('Loss')})}"
        )

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            
            # Concatenate features
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            
            # Add all prediction types
            eval_results['Predictions'] = predictions
            eval_results['Labels'] = all_labels

        return eval_results