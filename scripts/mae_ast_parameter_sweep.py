#!/usr/bin/env python3
"""
MAE-AST Parameter Sweep Script
This script systematically tests different MAE-AST configurations for overnight training.
Based on the mae_ast.yaml config but tests various parameter combinations.
"""

import os
import subprocess
import json
import time
import datetime
from pathlib import Path
from itertools import product
import argparse

class MAEASTSweep:
    def __init__(self, base_config="mae_ast", data_dir=None, output_base_dir=None):
        self.base_config = base_config
        
        # Set up paths
        self.project_root = Path(__file__).parent.parent
        self.config_dir = self.project_root / "config" / "pretrain"
        
        # Data directory - use the audio chunks folder
        if data_dir is None:
            self.data_dir = self.project_root / "data" / "audio_chunks-MARS-20171030T000000Z-10secs"
        else:
            self.data_dir = Path(data_dir)
            
        # Output directory
        if output_base_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_base_dir = self.project_root / "outputs" / f"mae_ast_sweep_{timestamp}"
        else:
            self.output_base_dir = Path(output_base_dir)
            
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # User directory for mae_ast
        self.user_dir = self.project_root / "mae_ast"
        
        # Results tracking
        self.results_file = self.output_base_dir / "sweep_results.json"
        self.results = []
        
    def define_parameter_space(self):
        """Define the parameter space to explore"""
        # Based on the conversation and typical MAE-AST configurations
        parameter_space = {
            # Encoder layers - keep decoder at 1, vary encoder
            "encoder_layers": [2, 4, 6, 8],
            
            # Decoder layers - mostly keep at 1, but test 2 for comparison
            "decoder_layers": [1, 2],
            
            # Max tokens - test researcher's small value vs base large value
            "max_tokens": [1048576, 1400000],
            
            # Warmup updates - test both values
            "warmup_updates": [4000, 8000, 16000, 32000],
            
            # Update frequency - test different values
            "update_freq": [4, 8, 16],
            
            # Learning rate - test different LRs
            "lr": [0.0001, 0.0005, 0.001],
            
            # Mask probability - important for MAE performance
            "random_mask_prob": [0.65, 0.75, 0.85],
        }
        
        return parameter_space
    
    def generate_configs(self, max_configs=None):
        """Generate configuration combinations"""
        param_space = self.define_parameter_space()
        
        # Create all combinations
        keys = list(param_space.keys())
        values = list(param_space.values())
        
        all_combinations = list(product(*values))
        
        # Limit number of configurations if specified
        if max_configs and len(all_combinations) > max_configs:
            print(f"Limiting to {max_configs} configurations out of {len(all_combinations)} possible")
            # Select a diverse subset
            step = len(all_combinations) // max_configs
            all_combinations = all_combinations[::step][:max_configs]
        
        configs = []
        for i, combo in enumerate(all_combinations):
            config = dict(zip(keys, combo))
            config['config_id'] = i
            configs.append(config)
            
        print(f"Generated {len(configs)} configurations to test")
        return configs
    
    def run_training(self, config):
        """Run a single training configuration"""
        config_id = config['config_id']
        run_name = f"run_{config_id:03d}"
        
        # Create output directory for this run
        run_output_dir = self.output_base_dir / run_name
        run_output_dir.mkdir(exist_ok=True)
        
        # Build the fairseq command
        cmd = [
            "fairseq-hydra-train",
            f"--config-dir={self.config_dir}",
            f"--config-name={self.base_config}",
            f"common.user_dir={self.user_dir}",
            f"task.data={self.data_dir}",
            "model._name=mae_ast",
            "criterion._name=mae_ast",
            f"model.encoder_layers={config['encoder_layers']}",
            f"model.decoder_layers={config['decoder_layers']}",
            f"dataset.max_tokens={config['max_tokens']}",
            f"lr_scheduler.warmup_updates={config['warmup_updates']}",
            f"optimization.update_freq=[{config['update_freq']}]",
            f"optimization.lr=[{config['lr']}]",
            f"model.random_mask_prob={config['random_mask_prob']}",
            f"hydra.run.dir={run_output_dir}",
            # Reduce max_update for faster testing - adjust based on your needs
            "optimization.max_update=50000",  # Reduced for overnight testing
            # Add early stopping conditions
            "dataset.validate_interval=2",
            "dataset.validate_interval_updates=5000",
            "checkpoint.save_interval_updates=10000",
            "common.log_interval=100",
        ]
        
        # Save configuration
        config_file = run_output_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save command
        cmd_file = run_output_dir / "command.txt"
        with open(cmd_file, 'w') as f:
            f.write(' '.join(cmd))
        
        print(f"\n{'='*60}")
        print(f"Starting training for configuration {config_id}")
        print(f"Config: {config}")
        print(f"Output: {run_output_dir}")
        print(f"{'='*60}")
        
        # Run training
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout per run
            )
            
            success = result.returncode == 0
            error_msg = result.stderr if not success else None
            
        except subprocess.TimeoutExpired:
            success = False
            error_msg = "Training timed out after 2 hours"
            
        except Exception as e:
            success = False
            error_msg = str(e)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Record results
        result_record = {
            "config_id": config_id,
            "config": config,
            "success": success,
            "duration_seconds": duration,
            "duration_formatted": str(datetime.timedelta(seconds=int(duration))),
            "output_dir": str(run_output_dir),
            "error_msg": error_msg,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Try to extract loss from logs if available
        try:
            log_files = list(run_output_dir.glob("**/train.log"))
            if log_files:
                result_record["log_file"] = str(log_files[0])
        except:
            pass
        
        self.results.append(result_record)
        
        # Save results after each run
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Completed configuration {config_id} in {datetime.timedelta(seconds=int(duration))}")
        print(f"Success: {success}")
        if error_msg:
            print(f"Error: {error_msg}")
        
        return result_record
    
    def run_sweep(self, max_configs=None, continue_from=None):
        """Run the full parameter sweep"""
        configs = self.generate_configs(max_configs)
        
        # Filter configs if continuing from a specific point
        if continue_from is not None:
            configs = [c for c in configs if c['config_id'] >= continue_from]
            print(f"Continuing from configuration {continue_from}")
        
        print(f"\nStarting MAE-AST parameter sweep")
        print(f"Total configurations: {len(configs)}")
        print(f"Output directory: {self.output_base_dir}")
        print(f"Data directory: {self.data_dir}")
        
        # Save sweep metadata
        metadata = {
            "start_time": datetime.datetime.now().isoformat(),
            "total_configs": len(configs),
            "config_space": self.define_parameter_space(),
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_base_dir)
        }
        
        with open(self.output_base_dir / "sweep_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Run each configuration
        for i, config in enumerate(configs):
            print(f"\nProgress: {i+1}/{len(configs)}")
            try:
                self.run_training(config)
            except KeyboardInterrupt:
                print("\nSweep interrupted by user")
                break
            except Exception as e:
                print(f"Error in configuration {config['config_id']}: {e}")
                # Record the error and continue
                error_record = {
                    "config_id": config['config_id'],
                    "config": config,
                    "success": False,
                    "error_msg": str(e),
                    "timestamp": datetime.datetime.now().isoformat()
                }
                self.results.append(error_record)
        
        print(f"\nSweep completed!")
        print(f"Results saved to: {self.results_file}")
        self.print_summary()
    
    def print_summary(self):
        """Print a summary of results"""
        if not self.results:
            print("No results to summarize")
            return
        
        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]
        
        print(f"\n{'='*60}")
        print(f"SWEEP SUMMARY")
        print(f"{'='*60}")
        print(f"Total runs: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            durations = [r['duration_seconds'] for r in successful]
            avg_duration = sum(durations) / len(durations)
            print(f"Average duration: {datetime.timedelta(seconds=int(avg_duration))}")
        
        if failed:
            print(f"\nFailed configurations:")
            for r in failed:
                print(f"  Config {r['config_id']}: {r.get('error_msg', 'Unknown error')}")

def main():
    parser = argparse.ArgumentParser(description="MAE-AST Parameter Sweep")
    parser.add_argument("--max-configs", type=int, default=20, 
                       help="Maximum number of configurations to test (default: 20)")
    parser.add_argument("--data-dir", type=str, 
                       help="Path to data directory (default: auto-detect)")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for results (default: auto-generate)")
    parser.add_argument("--continue-from", type=int,
                       help="Continue sweep from specific configuration ID")
    parser.add_argument("--config", type=str, default="mae_ast",
                       help="Base config name (default: mae_ast)")
    
    args = parser.parse_args()
    
    # Create sweep instance
    sweep = MAEASTSweep(
        base_config=args.config,
        data_dir=args.data_dir,
        output_base_dir=args.output_dir
    )
    
    # Run the sweep
    sweep.run_sweep(
        max_configs=args.max_configs,
        continue_from=args.continue_from
    )

if __name__ == "__main__":
    main() 