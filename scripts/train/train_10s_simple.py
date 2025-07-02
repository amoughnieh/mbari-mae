#!/usr/bin/env python3
"""
Simple script to train 10s data, log time and final loss.
"""
import subprocess
import time
import os
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_final_loss_from_tb(log_dir):
    """Get final loss from tensorboard logs"""
    event_files = list(Path(log_dir).glob("**/events.out.tfevents.*"))
    if not event_files:
        print(f"No tensorboard event files found in {log_dir}")
        return None
    
    latest_event_file = max(event_files, key=lambda x: x.stat().st_mtime)
    try:
        ea = EventAccumulator(str(latest_event_file))
        ea.Reload()
        
        # Try 'loss' first, then 'train'
        if 'loss' in ea.Tags()['scalars']:
            vals = ea.Scalars('loss')
        elif 'train' in ea.Tags()['scalars']:
            vals = ea.Scalars('train')
        else:
            print(f"No 'loss' or 'train' scalar found in {latest_event_file}")
            return None
        
        if vals:
            return vals[-1].value
    except Exception as e:
        print(f"Error reading tensorboard log: {e}")
    return None

def main():
    print("===== TRAINING 10S DATA =====")
    
    # Set up paths (relative to current project structure)
    project_root = Path(__file__).parent.parent.parent  # Go up to mbari-mae/
    data_dir = project_root / "data" / "manifests" / "audio_chunks-MARS-20171030T000000Z-10secs"
    config_dir = project_root / "config" / "pretrain"
    user_dir = project_root / "mae_ast"
    output_dir = project_root / "outputs" / "mae_ast_test_10s_simple"
    
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print paths for verification
    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    print(f"Config directory: {config_dir}")
    print(f"Output directory: {output_dir}")
    print(f"User directory: {user_dir}")
    
    # Set environment variables
    env = os.environ.copy()
    env['HYDRA_FULL_ERROR'] = '1'
    
    # Parameters for 2kHz, 10s chunks
    max_sample_size = 20000  # 10 seconds at 2kHz
    max_tokens = 25000  # Must be > 20,000 since dataset.num_tokens() returns raw sample count
    max_updates = 100  # Small number for testing
    
    # Build the training command (using the working approach)
    cmd = [
        "fairseq-hydra-train",
        "--config-dir", str(config_dir),
        "--config-name", "mbari_10s_2khz.yaml",
        f"task.data={data_dir}",
        f"common.user_dir={user_dir}",
        "model._name=mae_ast",
        "criterion._name=mae_ast",
        f"dataset.max_tokens={max_tokens}",
        f"task.max_sample_size={max_sample_size}",
        f"+task.max_keep_size={max_sample_size}",
        "task.sample_rate=2000",
        f"optimization.max_update={max_updates}",
        "common.log_interval=10",
        "distributed_training.distributed_world_size=1",
        "distributed_training.nprocs_per_node=1",
        f"hydra.run.dir={output_dir}"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Time the training
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time:.2f} seconds.")
        
        if result.returncode == 0:
            print("Training succeeded!")
            
            # Get final loss
            tb_dir = output_dir / "tblog" / "train"
            final_loss = get_final_loss_from_tb(tb_dir)
            
            if final_loss is not None:
                print(f"Final loss: {final_loss}")
            else:
                print("Could not find final loss in tensorboard logs.")
                
        else:
            print(f"Training failed. STDERR:\n{result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("Training timed out.")
    except Exception as e:
        print(f"Error running training: {e}")

if __name__ == "__main__":
    main() 