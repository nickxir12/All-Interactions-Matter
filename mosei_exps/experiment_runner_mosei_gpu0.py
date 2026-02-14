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
            # {
            #     "name": "exp1",
            #     "changes": {
            #         "msalm.datasetParams.sims.n_bn_fusion": 8,  # from 8
            #         "msalm.datasetParams.sims.na_bn_fusion": 4,  # from 4
            #         "msalm.datasetParams.sims.nv_bn_fusion": 4,  # from 4
            #         "msalm.datasetParams.sims.mmgpt.attention_pattern": "original",
            #         "msalm.datasetParams.sims.learning_rate_mmgpt": 4e-4,
            #         "msalm.datasetParams.sims.learning_rate_av": 1e-4,
            #         "msalm.datasetParams.sims.mmgpt.dropout": 0.2,
            #         "msalm.datasetParams.sims.mmgpt.layer_dropout": 0.1,
            #         "msalm.datasetParams.sims.weight_decay_mmgpt": 0.2,
            #         "msalm.datasetParams.sims.weight_decay_av": 0.2,
            #         "msalm.datasetParams.sims.grad_clip": 2.0,
            #         "msalm.datasetParams.sims.mmgpt.mm_layer": [8,9,10,11],
            #         "msalm.commonParams.early_stop": 15
            #     }
            # }
            {
                "name": "exp0",
                "changes": {
                    "msalm.datasetParams.sims.n_bn_fusion": 16,  # from 8
                    "msalm.datasetParams.sims.na_bn_fusion": 8,  # from 4
                    "msalm.datasetParams.sims.nv_bn_fusion": 8,  # from 4
                    "msalm.datasetParams.sims.mmgpt.attention_pattern": "avseesall",
                    "msalm.datasetParams.sims.learning_rate_mmgpt": 4e-4,
                    "msalm.datasetParams.sims.learning_rate_av": 1e-4,
                    "msalm.datasetParams.sims.mmgpt.dropout": 0.2,
                    "msalm.datasetParams.sims.mmgpt.layer_dropout": 0.1,
                    "msalm.datasetParams.sims.weight_decay_mmgpt": 0.2,
                    "msalm.datasetParams.sims.weight_decay_av": 0.2,
                    "msalm.datasetParams.sims.grad_clip": 2.0,
                    "msalm.datasetParams.sims.mmgpt.mm_layer": [5,6,7,8,9,10,11],
                    "msalm.commonParams.early_stop": 15
                }
            }
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