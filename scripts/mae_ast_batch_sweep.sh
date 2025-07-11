#!/bin/bash

# MAE-AST Parameter Sweep Batch Script
# This script runs different MAE-AST configurations in sequence
# Usage: ./mae_ast_batch_sweep.sh

set -e  # Exit on error

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/config/pretrain"
USER_DIR="${PROJECT_ROOT}/mae_ast"
DATA_DIR="${PROJECT_ROOT}/data/audio_chunks-MARS-20171030T000000Z-10secs"
BASE_CONFIG="mae_ast"

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_BASE="${PROJECT_ROOT}/outputs/mae_ast_batch_sweep_${TIMESTAMP}"
mkdir -p "${OUTPUT_BASE}"

# Results file
RESULTS_FILE="${OUTPUT_BASE}/batch_results.txt"

echo "MAE-AST Parameter Sweep Started at $(date)" | tee "${RESULTS_FILE}"
echo "Output directory: ${OUTPUT_BASE}" | tee -a "${RESULTS_FILE}"
echo "Data directory: ${DATA_DIR}" | tee -a "${RESULTS_FILE}"
echo "================================================" | tee -a "${RESULTS_FILE}"

# Function to run a single configuration
run_config() {
    local config_name="$1"
    local encoder_layers="$2"
    local decoder_layers="$3"
    local max_tokens="$4"
    local warmup_updates="$5"
    local update_freq="$6"
    local lr="$7"
    local mask_prob="$8"
    
    local run_dir="${OUTPUT_BASE}/${config_name}"
    mkdir -p "${run_dir}"
    
    echo "Starting ${config_name} at $(date)" | tee -a "${RESULTS_FILE}"
    echo "Config: enc=${encoder_layers}, dec=${decoder_layers}, tokens=${max_tokens}, warmup=${warmup_updates}, freq=${update_freq}, lr=${lr}, mask=${mask_prob}" | tee -a "${RESULTS_FILE}"
    
    # Save configuration
    cat > "${run_dir}/config.txt" << EOF
encoder_layers=${encoder_layers}
decoder_layers=${decoder_layers}
max_tokens=${max_tokens}
warmup_updates=${warmup_updates}
update_freq=${update_freq}
lr=${lr}
random_mask_prob=${mask_prob}
EOF
    
    # Run training
    local start_time=$(date +%s)
    local success=true
    
    fairseq-hydra-train \
        --config-dir="${CONFIG_DIR}" \
        --config-name="${BASE_CONFIG}" \
        common.user_dir="${USER_DIR}" \
        task.data="${DATA_DIR}" \
        model._name=mae_ast \
        criterion._name=mae_ast \
        model.encoder_layers="${encoder_layers}" \
        model.decoder_layers="${decoder_layers}" \
        dataset.max_tokens="${max_tokens}" \
        lr_scheduler.warmup_updates="${warmup_updates}" \
        optimization.update_freq="[${update_freq}]" \
        optimization.lr="[${lr}]" \
        model.random_mask_prob="${mask_prob}" \
        hydra.run.dir="${run_dir}" \
        optimization.max_update=300 \
        dataset.validate_interval=50 \
        dataset.validate_interval_updates=300 \
        checkpoint.save_interval_updates=300 \
        checkpoint.keep_interval_updates=1 \
        checkpoint.no_epoch_checkpoints=true \
        common.log_interval=100 || success=false
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_formatted=$(printf '%02dh:%02dm:%02ds' $((duration/3600)) $((duration%3600/60)) $((duration%60)))
    
    if [ "$success" = true ]; then
        echo "✓ ${config_name} completed successfully in ${duration_formatted}" | tee -a "${RESULTS_FILE}"
    else
        echo "✗ ${config_name} failed after ${duration_formatted}" | tee -a "${RESULTS_FILE}"
    fi
    
    echo "------------------------------------------------" | tee -a "${RESULTS_FILE}"
}

# Configuration matrix
# Format: config_name encoder_layers decoder_layers max_tokens warmup_updates update_freq lr mask_prob

echo "Starting parameter sweep with configurations:" | tee -a "${RESULTS_FILE}"

# Core configurations focusing on encoder layers (keeping decoder=1)
run_config "config_01_2enc_1dec_base" 2 1 1048576 4000 8 0.0001 0.75
run_config "config_02_4enc_1dec_base" 4 1 1048576 4000 8 0.0001 0.75
run_config "config_03_6enc_1dec_base" 6 1 1048576 4000 8 0.0001 0.75
run_config "config_04_8enc_1dec_base" 8 1 1048576 4000 8 0.0001 0.75

# Test with larger max_tokens (from base config)
run_config "config_05_4enc_1dec_large_tokens" 4 1 1400000 8000 8 0.0001 0.75
run_config "config_06_6enc_1dec_large_tokens" 6 1 1400000 8000 8 0.0001 0.75

# Test with larger warmup
run_config "config_07_4enc_1dec_large_warmup" 4 1 1048576 16000 8 0.0001 0.75
run_config "config_08_6enc_1dec_large_warmup" 6 1 1048576 16000 8 0.0001 0.75

# Test different learning rates
run_config "config_09_4enc_1dec_high_lr" 4 1 1048576 4000 8 0.0005 0.75
run_config "config_10_6enc_1dec_high_lr" 6 1 1048576 4000 8 0.0005 0.75

# Test different mask probabilities
run_config "config_11_4enc_1dec_low_mask" 4 1 1048576 4000 8 0.0001 0.65
run_config "config_12_4enc_1dec_high_mask" 4 1 1048576 4000 8 0.0001 0.85

# Test different update frequencies
run_config "config_13_4enc_1dec_low_freq" 4 1 1048576 4000 4 0.0001 0.75
run_config "config_14_4enc_1dec_high_freq" 4 1 1048576 4000 16 0.0001 0.75

# Test with 2 decoder layers (comparison)
run_config "config_15_4enc_2dec_base" 4 2 1048576 4000 8 0.0001 0.75
run_config "config_16_6enc_2dec_base" 6 2 1048576 4000 8 0.0001 0.75

# Optimal candidates based on typical MAE performance
run_config "config_17_optimal_small" 4 1 1048576 8000 8 0.0003 0.75
run_config "config_18_optimal_medium" 6 1 1400000 16000 8 0.0002 0.75
run_config "config_19_optimal_large" 8 1 1400000 32000 4 0.0001 0.85

echo "================================================" | tee -a "${RESULTS_FILE}"
echo "All configurations completed at $(date)" | tee -a "${RESULTS_FILE}"

# Generate summary
echo "Generating summary..." | tee -a "${RESULTS_FILE}"
cd "${OUTPUT_BASE}"

echo "Summary of results:" | tee -a "${RESULTS_FILE}"
for config_dir in config_*; do
    if [ -d "$config_dir" ]; then
        if [ -f "${config_dir}/checkpoints/checkpoint_last.pt" ]; then
            echo "✓ $config_dir: Training completed successfully" | tee -a "${RESULTS_FILE}"
        elif [ -f "${config_dir}/train.log" ]; then
            echo "~ $config_dir: Training started but may not have completed" | tee -a "${RESULTS_FILE}"
        else
            echo "✗ $config_dir: Training failed to start" | tee -a "${RESULTS_FILE}"
        fi
    fi
done

echo "================================================" | tee -a "${RESULTS_FILE}"
echo "Results saved in: ${OUTPUT_BASE}" | tee -a "${RESULTS_FILE}"
echo "To analyze results, check individual config directories for checkpoints and logs" | tee -a "${RESULTS_FILE}"

# Make the script executable
chmod +x "${BASH_SOURCE[0]}" 