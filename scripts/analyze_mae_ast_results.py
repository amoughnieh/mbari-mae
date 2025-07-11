#!/usr/bin/env python3
"""
MAE-AST Results Analysis Script
Analyzes the results from MAE-AST parameter sweep experiments
"""

import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np

class MAEASTResultsAnalyzer:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.data = []
        
    def extract_metrics_from_logs(self, log_file: Path) -> Dict:
        """Extract training metrics from log files"""
        metrics = {
            'final_train_loss': None,
            'final_valid_loss': None,
            'best_valid_loss': None,
            'final_reconstruction_loss': None,
            'final_classification_loss': None,
            'convergence_epoch': None,
            'total_epochs': 0
        }
        
        if not log_file.exists():
            return metrics
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            train_losses = []
            valid_losses = []
            recon_losses = []
            class_losses = []
            
            for line in lines:
                if 'train_loss' in line:
                    # Extract training loss (handle both quoted and unquoted numbers)
                    match = re.search(r'"train_loss":\s*"?([0-9.]+)"?', line)
                    if match:
                        train_losses.append(float(match.group(1)))
                
                if 'valid_loss' in line:
                    # Extract validation loss (handle both quoted and unquoted numbers)
                    match = re.search(r'"valid_loss":\s*"?([0-9.]+)"?', line)
                    if match:
                        valid_losses.append(float(match.group(1)))
                
                if 'reconstruction_loss' in line:
                    match = re.search(r'"reconstruction_loss":\s*"?([0-9.]+)"?', line)
                    if match:
                        recon_losses.append(float(match.group(1)))
                
                if 'classification_loss' in line:
                    match = re.search(r'"classification_loss":\s*"?([0-9.]+)"?', line)
                    if match:
                        class_losses.append(float(match.group(1)))
            
            # Extract final values
            if train_losses:
                metrics['final_train_loss'] = train_losses[-1]
                metrics['total_epochs'] = len(train_losses)
            
            if valid_losses:
                metrics['final_valid_loss'] = valid_losses[-1]
                metrics['best_valid_loss'] = min(valid_losses)
                
                # Find convergence point (where validation loss stops improving significantly)
                if len(valid_losses) > 10:
                    best_idx = np.argmin(valid_losses)
                    metrics['convergence_epoch'] = best_idx + 1
            
            if recon_losses:
                metrics['final_reconstruction_loss'] = recon_losses[-1]
            
            if class_losses:
                metrics['final_classification_loss'] = class_losses[-1]
                
        except Exception as e:
            print(f"Error parsing log file {log_file}: {e}")
        
        return metrics
    
    def collect_results(self):
        """Collect results from all experiment directories"""
        print(f"Collecting results from {self.results_dir}")
        
        # Look for different types of result directories
        config_dirs = []
        
        # Python sweep results
        if (self.results_dir / "sweep_results.json").exists():
            with open(self.results_dir / "sweep_results.json", 'r') as f:
                sweep_data = json.load(f)
            
            for result in sweep_data:
                if result['success']:
                    config_dirs.append(Path(result['output_dir']))
        
        # Batch sweep results
        else:
            config_dirs = list(self.results_dir.glob("config_*"))
        
        print(f"Found {len(config_dirs)} configuration directories")
        
        for config_dir in config_dirs:
            if not config_dir.is_dir():
                continue
                
            config_data = self.extract_config_data(config_dir)
            if config_data:
                self.data.append(config_data)
        
        print(f"Collected data from {len(self.data)} successful experiments")
    
    def extract_config_data(self, config_dir: Path) -> Optional[Dict]:
        """Extract configuration and results data from a single experiment"""
        try:
            # Load configuration
            config = {}
            
            # Try JSON config first
            config_file = config_dir / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                # Try text config
                config_file = config_dir / "config.txt"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        for line in f:
                            if '=' in line:
                                key, value = line.strip().split('=', 1)
                                try:
                                    config[key] = float(value) if '.' in value else int(value)
                                except ValueError:
                                    config[key] = value
            
            if not config:
                print(f"No config found in {config_dir}")
                return None
            
            # Extract metrics from logs
            log_files = list(config_dir.glob("**/train.log")) + list(config_dir.glob("**/hydra_train.log"))
            metrics = {}
            
            if log_files:
                metrics = self.extract_metrics_from_logs(log_files[0])
            
            # Check if training completed
            checkpoint_files = list(config_dir.glob("**/checkpoint_last.pt"))
            completed = len(checkpoint_files) > 0
            
            # Combine all data
            result = {
                'config_name': config_dir.name,
                'config_dir': str(config_dir),
                'completed': completed,
                **config,
                **metrics
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing {config_dir}: {e}")
            return None
    
    def create_analysis_report(self) -> pd.DataFrame:
        """Create a comprehensive analysis report"""
        if not self.data:
            print("No data to analyze")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.data)
        
        # Clean up data types
        numeric_cols = ['encoder_layers', 'decoder_layers', 'max_tokens', 'warmup_updates', 
                       'update_freq', 'lr', 'random_mask_prob', 'final_train_loss', 
                       'final_valid_loss', 'best_valid_loss', 'total_epochs']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def plot_results(self, df: pd.DataFrame, output_dir: Path):
        """Generate analysis plots"""
        output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Plot 1: Loss vs Encoder Layers
        if 'encoder_layers' in df.columns and 'best_valid_loss' in df.columns:
            plt.figure(figsize=(10, 6))
            completed_df = df[df['completed'] == True]
            
            if not completed_df.empty:
                # Clean the data for boxplot
                plot_df = completed_df[['encoder_layers', 'best_valid_loss']].dropna()
                if not plot_df.empty:
                    # Convert encoder_layers to string to ensure categorical plotting
                    plot_df['encoder_layers'] = plot_df['encoder_layers'].astype(str)
                    sns.boxplot(data=plot_df, x='encoder_layers', y='best_valid_loss')
                    plt.title('Validation Loss vs Encoder Layers')
                    plt.ylabel('Best Validation Loss')
                    plt.xlabel('Number of Encoder Layers')
                    plt.savefig(output_dir / 'loss_vs_encoder_layers.png', dpi=300, bbox_inches='tight')
                    plt.close()
        
        # Plot 2: Learning Rate Analysis
        if 'lr' in df.columns and 'best_valid_loss' in df.columns:
            plt.figure(figsize=(10, 6))
            completed_df = df[df['completed'] == True]
            
            if not completed_df.empty:
                sns.scatterplot(data=completed_df, x='lr', y='best_valid_loss', 
                              hue='encoder_layers', size='warmup_updates', alpha=0.7)
                plt.title('Validation Loss vs Learning Rate')
                plt.ylabel('Best Validation Loss')
                plt.xlabel('Learning Rate')
                plt.xscale('log')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.savefig(output_dir / 'loss_vs_learning_rate.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Plot 3: Mask Probability Analysis
        if 'random_mask_prob' in df.columns and 'best_valid_loss' in df.columns:
            plt.figure(figsize=(10, 6))
            completed_df = df[df['completed'] == True]
            
            if not completed_df.empty:
                # Clean the data for boxplot
                plot_df = completed_df[['random_mask_prob', 'best_valid_loss']].dropna()
                if not plot_df.empty:
                    # Convert random_mask_prob to string to ensure categorical plotting
                    plot_df['random_mask_prob'] = plot_df['random_mask_prob'].astype(str)
                    sns.boxplot(data=plot_df, x='random_mask_prob', y='best_valid_loss')
                    plt.title('Validation Loss vs Mask Probability')
                    plt.ylabel('Best Validation Loss')
                    plt.xlabel('Random Mask Probability')
                    plt.savefig(output_dir / 'loss_vs_mask_prob.png', dpi=300, bbox_inches='tight')
                    plt.close()
        
        # Plot 4: Convergence Analysis
        if 'convergence_epoch' in df.columns and 'total_epochs' in df.columns:
            plt.figure(figsize=(12, 8))
            completed_df = df[df['completed'] == True]
            
            if not completed_df.empty:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Convergence epoch distribution
                ax1.hist(completed_df['convergence_epoch'].dropna(), bins=20, alpha=0.7)
                ax1.set_title('Distribution of Convergence Epochs')
                ax1.set_xlabel('Convergence Epoch')
                ax1.set_ylabel('Frequency')
                
                # Total epochs vs convergence
                ax2.scatter(completed_df['total_epochs'], completed_df['convergence_epoch'], alpha=0.7)
                ax2.plot([0, completed_df['total_epochs'].max()], [0, completed_df['total_epochs'].max()], 
                        'r--', alpha=0.5, label='y=x')
                ax2.set_title('Total Epochs vs Convergence Epoch')
                ax2.set_xlabel('Total Epochs')
                ax2.set_ylabel('Convergence Epoch')
                ax2.legend()
                
                plt.tight_layout()
                plt.savefig(output_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Plot 5: Parameter Interaction Heatmap
        numeric_params = ['encoder_layers', 'decoder_layers', 'lr', 'warmup_updates', 
                         'random_mask_prob', 'update_freq']
        available_params = [col for col in numeric_params if col in df.columns]
        
        if len(available_params) > 1 and 'best_valid_loss' in df.columns:
            plt.figure(figsize=(12, 8))
            completed_df = df[df['completed'] == True]
            
            if not completed_df.empty:
                # Create correlation matrix
                corr_data = completed_df[available_params + ['best_valid_loss']].corr()
                
                sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                           square=True, fmt='.3f')
                plt.title('Parameter Correlation Matrix (including Best Valid Loss)')
                plt.tight_layout()
                plt.savefig(output_dir / 'parameter_correlations.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def generate_report(self, output_file: Optional[Path] = None):
        """Generate a comprehensive text report"""
        if not self.data:
            return
        
        df = self.create_analysis_report()
        
        if output_file is None:
            output_file = self.results_dir / "analysis_report.txt"
        
        with open(output_file, 'w') as f:
            f.write("MAE-AST Parameter Sweep Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic statistics
            f.write(f"Total experiments: {len(df)}\n")
            f.write(f"Completed experiments: {df['completed'].sum()}\n")
            f.write(f"Success rate: {df['completed'].mean():.2%}\n\n")
            
            # Best configurations
            completed_df = df[df['completed'] == True]
            if not completed_df.empty and 'best_valid_loss' in completed_df.columns:
                # Check if we have valid best_valid_loss values
                valid_loss_df = completed_df[completed_df['best_valid_loss'].notna()]
                if not valid_loss_df.empty:
                    best_idx = valid_loss_df['best_valid_loss'].idxmin()
                    best_config = completed_df.loc[best_idx]
                    
                    f.write("Best Configuration (Lowest Validation Loss):\n")
                    f.write("-" * 40 + "\n")
                    for key, value in best_config.items():
                        if key not in ['config_dir', 'completed']:
                            f.write(f"{key}: {value}\n")
                    f.write("\n")
                else:
                    f.write("No valid validation loss data found for completed experiments.\n\n")
                
                # Parameter analysis
                f.write("Parameter Analysis:\n")
                f.write("-" * 20 + "\n")
                
                for param in ['encoder_layers', 'decoder_layers', 'lr', 'warmup_updates', 'random_mask_prob']:
                    if param in completed_df.columns and 'best_valid_loss' in completed_df.columns:
                        # Only calculate correlation if we have valid data
                        valid_data = completed_df[[param, 'best_valid_loss']].dropna()
                        if len(valid_data) > 1:
                            corr = valid_data[param].corr(valid_data['best_valid_loss'])
                            f.write(f"{param}: correlation with loss = {corr:.3f}\n")
                        else:
                            f.write(f"{param}: insufficient data for correlation\n")
                
                f.write("\n")
            
            # Summary statistics
            f.write("Summary Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(str(completed_df.describe()))
            
        print(f"Analysis report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze MAE-AST parameter sweep results")
    parser.add_argument("results_dir", help="Directory containing sweep results")
    parser.add_argument("--output-dir", help="Output directory for analysis (default: results_dir/analysis)")
    parser.add_argument("--plots-only", action="store_true", help="Generate only plots, skip text report")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Create analyzer
    analyzer = MAEASTResultsAnalyzer(results_dir)
    analyzer.collect_results()
    
    if not analyzer.data:
        print("No valid results found")
        return
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / "analysis"
    
    output_dir.mkdir(exist_ok=True)
    
    # Generate analysis
    df = analyzer.create_analysis_report()
    
    # Save DataFrame
    df.to_csv(output_dir / "results_summary.csv", index=False)
    print(f"Results summary saved to {output_dir / 'results_summary.csv'}")
    
    # Generate plots
    analyzer.plot_results(df, output_dir)
    print(f"Plots saved to {output_dir}")
    
    # Generate text report
    if not args.plots_only:
        analyzer.generate_report(output_dir / "analysis_report.txt")

if __name__ == "__main__":
    main() 