#!/usr/bin/env python3
"""
Model wrapper script that runs the actual MSALM training
This script can accept a config file path as argument or use default
"""
import subprocess
import sys
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run MSALM training with specified config')
    parser.add_argument('--config', help='Path to config file (optional)')
    parser.add_argument('--exp-name', default='mosei-train', help='Experiment name')
    
    args = parser.parse_args()
    
    # Paths
    work_dir = Path.home() / "Deepmlf" / "ModifiedDeepMLF"
    
    # Use provided config or default
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = work_dir / "MMSA" / "config" / "regression" / "deepmlf" / "mosei" / "base_best.json"
    
    # Verify config exists
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Change to work directory
    os.chdir(work_dir)
    
    # Run the actual training command
    cmd = [
        sys.executable,
        "experiments/regression/mult_base.py",
        "-m", "msalm",
        "-d", "mosei",
        "-g", "0",
        "--exp-name", args.exp_name,
        "-c", str(config_path),
        "--res-save-dir", "/leonardo/home/userexternal/nxiros00/Deepmlf/results/MOSEI",
        "--model-save-dir", "/leonardo/home/userexternal/nxiros00/Deepmlf/checkpoints/MOSEI/model",
        "-n", "2"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    # Execute the training
    result = subprocess.run(cmd, stdout=None, stderr=None)
    
    # Exit with the same code as the training script
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()