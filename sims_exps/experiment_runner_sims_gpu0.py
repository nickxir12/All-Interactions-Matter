#!/usr/bin/env python3
"""
Sequential Experiment Runner for MSALM Model
Updates config -> runs model -> repeats for 5 experiments
"""

import json
import copy
import subprocess
import sys
from pathlib import Path

class ExperimentRunner:
    def __init__(self, base_config_path, model_script):
        """
        Initialize the experiment runner
        
        Args:
            base_config_path: Path to the base configuration JSON file
            model_script: Path to the model training script
        """
        self.base_config_path = base_config_path
        self.model_script = model_script
        
        # Load base configuration
        with open(base_config_path, 'r') as f:
            self.base_config = json.load(f)
    
    def define_experiment_variants(self):
        """
        Define 5 different experiment configurations
        """
        experiments = [
            {
                "name": "exp1",
                "changes": {
                    # === fusion & mmgpt ===
                    "msalm.datasetParams.sims.n_bn_fusion": 16,     # from 8
                    "msalm.datasetParams.sims.na_bn_fusion": 12,    # from 4
                    "msalm.datasetParams.sims.nv_bn_fusion": 10,    # from 4
                    "msalm.datasetParams.sims.mmgpt.attention_pattern": "avseesall",
                    "msalm.datasetParams.sims.learning_rate_mmgpt": 6e-4,
                    "msalm.datasetParams.sims.learning_rate_av": 6e-4,
                    "msalm.datasetParams.sims.mmgpt.mm_layer": [4,5,6,7,8,9,10,11],
                    "msalm.datasetParams.sims.av_enc.load_one_path": True,
                    "msalm.datasetParams.sims.av_enc.single_pretrained_path": "/leonardo/home/userexternal/nxiros00/Deepmlf/checkpoints/SIMS/sims-pre-train-bienc-seed1991/bienc-sims-1991-d_enc_30-n_head_6-nlevels_6-maxlen_55.pth",
                    "msalm.datasetParams.sims.av_enc.load_multiple_paths": False,

                    # === audio encoder ===
                    "msalm.datasetParams.sims.av_enc.audio_encoder.d_enc": 30,  # change from 32
                    "msalm.datasetParams.sims.av_enc.vision_encoder.n_embd": 30,  # change from 30
                    "msalm.datasetParams.sims.mmgpt.kv_dim_a": 30,  # change from 32
                    "msalm.datasetParams.sims.av_enc.audio_encoder.n_head": 6,  # change from 30
                    "msalm.datasetParams.sims.av_enc.audio_encoder.nlevels": 6,  # change from 30
                    "msalm.datasetParams.sims.av_enc.audio_encoder.maxlen": 55,  # change from 80
                    "msalm.datasetParams.sims.av_enc.audio_encoder.enc_dropout": 0.1,
                    "msalm.datasetParams.sims.av_enc.audio_encoder.use_ln": False,
                    "msalm.datasetParams.sims.av_enc.audio_encoder.use_bn": True,
                    "msalm.datasetParams.sims.av_enc.audio_encoder.use_softperm": True,
                    "msalm.datasetParams.sims.av_enc.audio_encoder.p_perm": 0.2,
                    "msalm.datasetParams.sims.av_enc.audio_encoder.p_mask": 0.15,
                    "msalm.datasetParams.sims.av_enc.audio_encoder.pooling": "no",
                    "msalm.datasetParams.sims.av_enc.audio_encoder.audio_pretrained_path": "/dont/care",

                    # === vision encoder ===
                    "msalm.datasetParams.sims.av_enc.vision_encoder.d_enc": 30,  # change from 32
                    "msalm.datasetParams.sims.av_enc.vision_encoder.n_embd": 30,  # change from 30
                    "msalm.datasetParams.sims.mmgpt.kv_dim_v": 30,  # change from 32
                    "msalm.datasetParams.sims.av_enc.vision_encoder.n_head":6,  # change from 30
                    "msalm.datasetParams.sims.av_enc.vision_encoder.nlevels": 6,  # change from 30
                    "msalm.datasetParams.sims.av_enc.vision_encoder.maxlen": 55,  # change from 80
                    "msalm.datasetParams.sims.av_enc.vision_encoder.enc_dropout": 0.1,
                    "msalm.datasetParams.sims.av_enc.vision_encoder.use_ln": False,
                    "msalm.datasetParams.sims.av_enc.vision_encoder.use_bn": True,
                    "msalm.datasetParams.sims.av_enc.vision_encoder.use_softperm": True,
                    "msalm.datasetParams.sims.av_enc.vision_encoder.p_perm": 0.2,
                    "msalm.datasetParams.sims.av_enc.vision_encoder.p_mask": 0.15,
                    "msalm.datasetParams.sims.av_enc.vision_encoder.pooling": "no",
                    "msalm.datasetParams.sims.av_enc.vision_encoder.video_pretrained_path": "/dont/care",

                    # === bimodal encoder ===
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.d_enc": 30,  # change from 30
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.d_enc_out": 30,  # change from 30
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.n_embd": 30,  # change from 30
                    "msalm.datasetParams.sims.mmgpt.kv_dim_av": 30,  # change from 32
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.n_head": 6,  # change from 30
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.nlevels": 6,  # change from 30
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.maxlen": 55,  # change from 55
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.bimodal_pretrained_path": "/dont/care"
                }
            },
            {
                "name": "exp1",
                "changes": {
                    # === fusion & mmgpt ===
                    "msalm.datasetParams.sims.n_bn_fusion": 16,     # from 8
                    "msalm.datasetParams.sims.na_bn_fusion": 12,    # from 4
                    "msalm.datasetParams.sims.nv_bn_fusion": 10,    # from 4
                    "msalm.datasetParams.sims.mmgpt.attention_pattern": "avseesall",
                    "msalm.datasetParams.sims.learning_rate_mmgpt": 6e-4,
                    "msalm.datasetParams.sims.learning_rate_av": 6e-4,
                    "msalm.datasetParams.sims.mmgpt.mm_layer": [4,5,6,7,8,9,10,11],
                    "msalm.datasetParams.sims.av_enc.load_one_path": True,
                    "msalm.datasetParams.sims.av_enc.single_pretrained_path": "/leonardo/home/userexternal/nxiros00/Deepmlf/checkpoints/SIMS/sims-pre-train-bienc-seed1990/bienc-sims-1992-d_enc_32-n_head_8-nlevels_6-maxlen_55.pth",
                    "msalm.datasetParams.sims.av_enc.load_multiple_paths": False,

                    # === audio encoder ===
                    "msalm.datasetParams.sims.av_enc.audio_encoder.d_enc": 32,  # change from 32
                    "msalm.datasetParams.sims.av_enc.vision_encoder.n_embd": 32,  # change from 30
                    "msalm.datasetParams.sims.mmgpt.kv_dim_a": 32,  # change from 32
                    "msalm.datasetParams.sims.av_enc.audio_encoder.n_head": 8,  # change from 30
                    "msalm.datasetParams.sims.av_enc.audio_encoder.nlevels": 6,  # change from 30
                    "msalm.datasetParams.sims.av_enc.audio_encoder.maxlen": 55,  # change from 80
                    "msalm.datasetParams.sims.av_enc.audio_encoder.enc_dropout": 0.1,
                    "msalm.datasetParams.sims.av_enc.audio_encoder.use_ln": False,
                    "msalm.datasetParams.sims.av_enc.audio_encoder.use_bn": True,
                    "msalm.datasetParams.sims.av_enc.audio_encoder.use_softperm": True,
                    "msalm.datasetParams.sims.av_enc.audio_encoder.p_perm": 0.2,
                    "msalm.datasetParams.sims.av_enc.audio_encoder.p_mask": 0.15,
                    "msalm.datasetParams.sims.av_enc.audio_encoder.pooling": "no",
                    "msalm.datasetParams.sims.av_enc.audio_encoder.audio_pretrained_path": "/dont/care",

                    # === vision encoder ===
                    "msalm.datasetParams.sims.av_enc.vision_encoder.d_enc": 32,  # change from 32
                    "msalm.datasetParams.sims.av_enc.vision_encoder.n_embd": 32,  # change from 30
                    "msalm.datasetParams.sims.mmgpt.kv_dim_v": 32,  # change from 32
                    "msalm.datasetParams.sims.av_enc.vision_encoder.n_head": 8,  # change from 30
                    "msalm.datasetParams.sims.av_enc.vision_encoder.nlevels": 6,  # change from 30
                    "msalm.datasetParams.sims.av_enc.vision_encoder.maxlen": 55,  # change from 80
                    "msalm.datasetParams.sims.av_enc.vision_encoder.enc_dropout": 0.1,
                    "msalm.datasetParams.sims.av_enc.vision_encoder.use_ln": False,
                    "msalm.datasetParams.sims.av_enc.vision_encoder.use_bn": True,
                    "msalm.datasetParams.sims.av_enc.vision_encoder.use_softperm": True,
                    "msalm.datasetParams.sims.av_enc.vision_encoder.p_perm": 0.2,
                    "msalm.datasetParams.sims.av_enc.vision_encoder.p_mask": 0.15,
                    "msalm.datasetParams.sims.av_enc.vision_encoder.pooling": "no",
                    "msalm.datasetParams.sims.av_enc.vision_encoder.video_pretrained_path": "/dont/care",

                    # === bimodal encoder ===
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.d_enc": 32,  # change from 30
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.d_enc_out": 30,  # change from 30
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.n_embd": 30,  # change from 30
                    "msalm.datasetParams.sims.mmgpt.kv_dim_av": 30,  # change from 32
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.n_head": 8,  # change from 30
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.nlevels": 6,  # change from 30
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.maxlen": 55,  # change from 55
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.bimodal_pretrained_path": "/dont/care"
                }
            },
            {
                "name": "exp1",
                "changes": {
                    # === fusion & mmgpt ===
                    "msalm.datasetParams.sims.n_bn_fusion": 16,     # from 8
                    "msalm.datasetParams.sims.na_bn_fusion": 12,    # from 4
                    "msalm.datasetParams.sims.nv_bn_fusion": 10,    # from 4
                    "msalm.datasetParams.sims.mmgpt.attention_pattern": "avseesall",
                    "msalm.datasetParams.sims.learning_rate_mmgpt": 6e-4,
                    "msalm.datasetParams.sims.learning_rate_av": 6e-4,
                    "msalm.datasetParams.sims.mmgpt.mm_layer": [4,5,6,7,8,9,10,11],
                    "msalm.datasetParams.sims.av_enc.load_one_path": True,
                    "msalm.datasetParams.sims.av_enc.single_pretrained_path": "/leonardo/home/userexternal/nxiros00/Deepmlf/checkpoints/SIMS/sims-pre-train-bienc/bienc-sims-1990-d_enc_30-n_head_6-nlevels_3-maxlen_39.pth",
                    "msalm.datasetParams.sims.av_enc.load_multiple_paths": False,

                    # === audio encoder ===
                    "msalm.datasetParams.sims.av_enc.audio_encoder.d_enc": 30,  # change from 32
                    "msalm.datasetParams.sims.av_enc.audio_encoder.n_embd": 30,  # change from 30
                    "msalm.datasetParams.sims.mmgpt.kv_dim_a": 30,  # change from 32
                    "msalm.datasetParams.sims.av_enc.audio_encoder.n_head": 6,  # change from 30
                    "msalm.datasetParams.sims.av_enc.audio_encoder.nlevels": 3,  # change from 30
                    "msalm.datasetParams.sims.av_enc.audio_encoder.maxlen": 55,  # change from 80
                    "msalm.datasetParams.sims.av_enc.audio_encoder.enc_dropout": 0.1,
                    "msalm.datasetParams.sims.av_enc.audio_encoder.use_ln": False,
                    "msalm.datasetParams.sims.av_enc.audio_encoder.use_bn": True,
                    "msalm.datasetParams.sims.av_enc.audio_encoder.use_softperm": True,
                    "msalm.datasetParams.sims.av_enc.audio_encoder.p_perm": 0.2,
                    "msalm.datasetParams.sims.av_enc.audio_encoder.p_mask": 0.15,
                    "msalm.datasetParams.sims.av_enc.audio_encoder.pooling": "no",
                    "msalm.datasetParams.sims.av_enc.audio_encoder.audio_pretrained_path": "/dont/care",

                    # === vision encoder ===
                    "msalm.datasetParams.sims.av_enc.vision_encoder.d_enc": 30,  # change from 32
                    "msalm.datasetParams.sims.av_enc.vision_encoder.n_embd": 30,  # change from 30
                    "msalm.datasetParams.sims.mmgpt.kv_dim_v": 30,  # change from 32
                    "msalm.datasetParams.sims.av_enc.vision_encoder.n_head":6,  # change from 30
                    "msalm.datasetParams.sims.av_enc.vision_encoder.nlevels": 3,  # change from 30
                    "msalm.datasetParams.sims.av_enc.vision_encoder.maxlen": 55,  # change from 80
                    "msalm.datasetParams.sims.av_enc.vision_encoder.enc_dropout": 0.1,
                    "msalm.datasetParams.sims.av_enc.vision_encoder.use_ln": False,
                    "msalm.datasetParams.sims.av_enc.vision_encoder.use_bn": True,
                    "msalm.datasetParams.sims.av_enc.vision_encoder.use_softperm": True,
                    "msalm.datasetParams.sims.av_enc.vision_encoder.p_perm": 0.2,
                    "msalm.datasetParams.sims.av_enc.vision_encoder.p_mask": 0.15,
                    "msalm.datasetParams.sims.av_enc.vision_encoder.pooling": "no",
                    "msalm.datasetParams.sims.av_enc.vision_encoder.video_pretrained_path": "/dont/care",

                    # === bimodal encoder ===
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.d_enc": 30,  # change from 30
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.d_enc_out": 30,  # change from 30
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.n_embd": 30,  # change from 30
                    "msalm.datasetParams.sims.mmgpt.kv_dim_av": 30,  # change from 32
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.n_head": 6,  # change from 30
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.nlevels": 3,  # change from 30
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.maxlen": 55,  # change from 55
                    "msalm.datasetParams.sims.av_enc.bimodal_encoder.bimodal_pretrained_path": "/dont/care"
                }
            },
        ]
        
        return experiments
    
    def apply_changes_to_config(self, config, changes):
        """
        Apply changes to configuration using dot notation
        """
        modified_config = copy.deepcopy(config)
        
        for key, value in changes.items():
            keys = key.split('.')
            current = modified_config
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set the value
            current[keys[-1]] = value
        
        return modified_config
    
    def update_config(self, changes):
        """Update the original config file with changes"""
        modified_config = self.apply_changes_to_config(self.base_config, changes)
        
        with open(self.base_config_path, 'w') as f:
            json.dump(modified_config, f, indent=4)
    
    def run_model(self, exp_name):
        """Run the model script with config parameter and experiment name"""
        cmd = [sys.executable, self.model_script, "--config", str(self.base_config_path), "--exp-name", exp_name]
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=None, stderr=None)
        return result.returncode == 0, "", ""
    
    def run_all_experiments(self):
        """Run all 5 experiments sequentially"""
        experiments = self.define_experiment_variants()
        
        for i, experiment in enumerate(experiments, 1):
            print(f"Running Experiment {i}: {experiment['name']}")
            
            # Update config
            self.update_config(experiment['changes'])
            print(f"Config updated with changes: {experiment['changes']}")
            
            # Run model
            success, _, _ = self.run_model(experiment['name'])
            
            if success:
                print(f"Experiment {i} completed successfully")
            else:
                print(f"Experiment {i} failed")
                # Continue with next experiment even if this one fails
            
            print("-" * 50)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run sequential experiments')
    parser.add_argument('--base_config', required=True, help='Path to base configuration file')
    parser.add_argument('--model_script', required=True, help='Path to model training script')
    
    args = parser.parse_args()
    
    # Run experiments
    runner = ExperimentRunner(args.base_config, args.model_script)
    runner.run_all_experiments()

if __name__ == "__main__":
    main()