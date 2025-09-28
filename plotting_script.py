#!/usr/bin/env python3
"""
Plotting script for visualizing algorithm performance results.
Creates comprehensive plots for time complexity, recursion depth, and comparisons.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import argparse
import json
import sys
import os

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ResultsPlotter:
    """Class for creating various performance visualization plots."""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize plotter with results directory.
        
        Args:
            results_dir: Directory containing benchmark results
        """
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load data
        self.csv_data = self._load_csv_data()
        self.json_data = self._load_json_data()
    
    def _load_csv_data(self) -> pd.DataFrame:
        """Load CSV benchmark data."""
        csv_file = self.results_dir / "benchmark_data.csv"
        if csv_file.exists():
            return pd.read_csv(csv_file)
        else:
            print(f"Warning: {csv_file} not found. Some plots may not be available.")
            return pd.DataFrame()
    
    def _load_json_data(self) -> dict:
        """Load JSON benchmark results."""
        json_file = self.results_dir / "benchmark_results.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: {json_file} not found. Some plots may not be available.")
            return {}
    
    def plot_time_complexity(self, algorithms: list = None, save: bool = True) -> None:
        """
        Plot execution time vs input size for different algorithms.
        
        Args:
            algorithms: List of algorithm names to plot (None for all)
            save: Whether to save the plot
        """
        if self.csv_data.empty:
            print("No CSV data available for time complexity plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Filter algorithms if specified
        if algorithms:
            data = self.csv_data[self.csv_data['algorithm'].isin(algorithms)]
        else:
            data = self.csv_data
        
        # Create subplot for each algorithm category
        sorting_algs = ['MergeSort', 'QuickSort', 'QuickSort3Way']
        selection_algs = ['DeterministicSelect', 'QuickSelect']
        closest_pair_algs = ['ClosestPair', 'BruteForceClosestPair']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Algorithm Performance: Execution Time vs Input Size', fontsize=16)
        
        # Plot sorting algorithms
        self._plot_algorithm_group(data, sorting_algs, axes[0, 0], 'Sorting Algorithms', 'time')
        
        # Plot selection algorithms  
        self._plot_algorithm_group(data, selection_algs, axes[0, 1], 'Selection Algorithms', 'time')
        
        # Plot closest pair algorithms
        self._plot_algorithm_group(data, closest_pair_algs, axes[1, 0], 'Closest Pair Algorithms', 'time')
        
        # Plot all algorithms together (log scale)
        axes[1, 1].set_yscale('log')
        self._plot_algorithm_group(data, None, axes[1, 1], 'All Algorithms (Log Scale)', 'time')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plots_dir / 'time_complexity.png', dpi=300, bbox_inches='tight')
            print(f"Time complexity plot saved to {self.plots_dir / 'time_complexity.png'}")
        
        plt.show()
    
    def plot_recursion_depth(self, algorithms: list = None, save: bool = True) -> None:
        """
        Plot recursion depth vs input size.
        
        Args:
            algorithms: List of algorithm names to plot
            save: Whether to save the plot
        """
        if self.csv_data.empty:
            print("No CSV data available for recursion depth plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        if algorithms:
            data = self.csv_data[self.csv_data['algorithm'].isin(algorithms)]
        else:
            data = self.csv_data
        
        # Plot recursion depth
        for alg in data['algorithm'].unique():
            alg_data = data[data['algorithm'] == alg]
            if not alg_data.empty and 'max_recursion_depth' in alg_data.columns:
                plt.plot(alg_data['input_size'], alg_data['max_recursion_depth'], 
                        marker='o', linewidth=2, markersize=6, label=alg)
        
        plt.xlabel('Input Size', fontsize=12)
        plt.ylabel('Maximum Recursion Depth', fontsize=12)
        plt.title('Recursion Depth vs Input Size', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add theoretical lines
        if not data.empty:
            x_range = np.linspace(data['input_size'].min(), data['input_size'].max(), 100)
            plt.plot(x_range, np.log2(x_range), '--', alpha=0.7, label='O(log n)', color='gray')
            plt.plot(x_range, x_range**0.5, '--', alpha=0.7, label='O(√n)', color='lightgray')
            plt.legend()
        
        if save:
            plt.savefig(self.plots_dir / 'recursion_depth.png', dpi=300, bbox_inches='tight')
            print(f"Recursion depth plot saved to {self.plots_dir / 'recursion_depth.png'}")
        
        plt.show()
    
    def plot_comparisons_analysis(self, save: bool = True) -> None:
        """
        Plot comparison count analysis for different algorithms.
        
        Args:
            save: Whether to save the plot
        """
        if self.csv_data.empty:
            print("No CSV data available for comparisons plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Algorithm Comparison Analysis', fontsize=16)
        
        # Sorting algorithms comparisons
        sorting_data = self.csv_data[self.csv_data['algorithm'].isin(['MergeSort', 'QuickSort'])]
        if not sorting_data.empty:
            for alg in sorting_data['algorithm'].unique():
                alg_data = sorting_data[sorting_data['algorithm'] == alg]
                axes[0, 0].plot(alg_data['input_size'], alg_data['comparisons'], 
                              marker='o', label=alg, linewidth=2)
            
            # Add theoretical n log n line
            if not sorting_data.empty:
                x_vals = sorting_data['input_size'].unique()
                x_vals = np.sort(x_vals)
                nlogn_vals = x_vals * np.log2(x_vals)
                # Normalize to fit the scale
                if len(sorting_data) > 0:
                    scale_factor = sorting_data['comparisons'].mean() / np.mean(nlogn_vals)
                    axes[0, 0].plot(x_vals, nlogn_vals * scale_factor, '--', 
                                  alpha=0.7, label='O(n log n)', color='gray')
            
            axes[0, 0].set_xlabel('Input Size')
            axes[0, 0].set_ylabel('Comparisons')
            axes[0, 0].set_title('Sorting: Comparisons vs Input Size')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Selection algorithms comparisons
        selection_data = self.csv_data[self.csv_data['algorithm'].isin(['DeterministicSelect', 'QuickSelect'])]
        if not selection_data.empty:
            for alg in selection_data['algorithm'].unique():
                alg_data = selection_data[selection_data['algorithm'] == alg]
                axes[0, 1].plot(alg_data['input_size'], alg_data['comparisons'], 
                              marker='o', label=alg, linewidth=2)
            
            # Add theoretical linear line
            x_vals = selection_data['input_size'].unique()
            x_vals = np.sort(x_vals)
            if len(selection_data) > 0:
                scale_factor = selection_data['comparisons'].mean() / np.mean(x_vals)
                axes[0, 1].plot(x_vals, x_vals * scale_factor, '--', 
                              alpha=0.7, label='O(n)', color='gray')
            
            axes[0, 1].set_xlabel('Input Size')
            axes[0, 1].set_ylabel('Comparisons')
            axes[0, 1].set_title('Selection: Comparisons vs Input Size')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Memory allocations
        if 'allocations' in self.csv_data.columns:
            for alg in self.csv_data['algorithm'].unique():
                alg_data = self.csv_data[self.csv_data['algorithm'] == alg]
                if not alg_data.empty:
                    axes[1, 0].plot(alg_data['input_size'], alg_data['allocations'], 
                                  marker='o', label=alg, linewidth=2)
            
            axes[1, 0].set_xlabel('Input Size')
            axes[1, 0].set_ylabel('Memory Allocations')
            axes[1, 0].set_title('Memory Allocations vs Input Size')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Algorithm efficiency comparison (time per comparison)
        if not self.csv_data.empty and 'comparisons' in self.csv_data.columns:
            # Calculate efficiency metric
            efficiency_data = self.csv_data.copy()
            efficiency_data['time_per_comparison'] = (
                efficiency_data['execution_time'] / 
                (efficiency_data['comparisons'] + 1)  # +1 to avoid division by zero
            )
            
            for alg in efficiency_data['algorithm'].unique():
                alg_data = efficiency_data[efficiency_data['algorithm'] == alg]
                if not alg_data.empty:
                    axes[1, 1].plot(alg_data['input_size'], alg_data['time_per_comparison'], 
                                  marker='o', label=alg, linewidth=2)
            
            axes[1, 1].set_xlabel('Input Size')
            axes[1, 1].set_ylabel('Time per Comparison (seconds)')
            axes[1, 1].set_title('Algorithm Efficiency: Time per Comparison')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plots_dir / 'comparisons_analysis.png', dpi=300, bbox_inches='tight')
            print(f"Comparisons analysis plot saved to {self.plots_dir / 'comparisons_analysis.png'}")
        
        plt.show()
    
    def _plot_algorithm_group(self, data: pd.DataFrame, algorithms: list, ax, title: str, metric: str):
        """Helper method to plot a group of algorithms."""
        if algorithms:
            plot_data = data[data['algorithm'].isin(algorithms)]
        else:
            plot_data = data
        
        for alg in plot_data['algorithm'].unique():
            alg_data = plot_data[plot_data['algorithm'] == alg]
            if not alg_data.empty and metric in alg_data.columns:
                ax.plot(alg_data['input_size'], alg_data[metric], 
                       marker='o', linewidth=2, markersize=6, label=alg)
        
        ax.set_xlabel('Input Size')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_pattern_comparison(self, save: bool = True) -> None:
        """
        Plot performance across different data patterns.
        
        Args:
            save: Whether to save the plot
        """
        if not self.json_data or 'sorting' not in self.json_data:
            print("No JSON data available for pattern comparison")
            return
        
        # Extract sorting results by pattern
        sorting_results = self.json_data['sorting']
        
        patterns = list(sorting_results.keys())
        algorithms = ['MergeSort', 'QuickSort', 'QuickSort3Way']
        
        # Create subplot for each algorithm
        fig, axes = plt.subplots(1, len(algorithms), figsize=(18, 6))
        if len(algorithms) == 1:
            axes = [axes]
        
        for i, alg in enumerate(algorithms):
            pattern_times = []
            pattern_names = []
            
            for pattern in patterns:
                if pattern in sorting_results and alg in sorting_results[pattern]:
                    alg_results = sorting_results[pattern][alg]
                    if alg_results:
                        avg_time = np.mean([r['execution_time'] for r in alg_results])
                        pattern_times.append(avg_time)
                        pattern_names.append(pattern.replace('_', '\n'))
            
            if pattern_times:
                bars = axes[i].bar(pattern_names, pattern_times)
                axes[i].set_title(f'{alg} Performance by Data Pattern')
                axes[i].set_ylabel('Average Execution Time (s)')
                axes[i].tick_params(axis='x', rotation=45)
                
                # Color bars by performance
                colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.plots_dir / 'pattern_comparison.png', dpi=300, bbox_inches='tight')
            print(f"Pattern comparison plot saved to {self.plots_dir / 'pattern_comparison.png'}")
        
        plt.show()
    
    def create_summary_dashboard(self, save: bool = True) -> None:
        """
        Create a comprehensive dashboard with multiple plots.
        
        Args:
            save: Whether to save the dashboard
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main time complexity plot
        ax1 = fig.add_subplot(gs[0, :2])
        if not self.csv_data.empty:
            for alg in self.csv_data['algorithm'].unique():
                alg_data = self.csv_data[self.csv_data['algorithm'] == alg]
                if not alg_data.empty:
                    ax1.loglog(alg_data['input_size'], alg_data['execution_time'], 
                             marker='o', linewidth=2, markersize=4, label=alg)
            
            ax1.set_xlabel('Input Size (log scale)')
            ax1.set_ylabel('Execution Time (log scale)')
            ax1.set_title('Time Complexity Overview (Log-Log Scale)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Recursion depth comparison
        ax2 = fig.add_subplot(gs[0, 2])
        recursive_algs = ['MergeSort', 'QuickSort', 'DeterministicSelect']
        depth_data = self.csv_data[self.csv_data['algorithm'].isin(recursive_algs)]
        if not depth_data.empty and 'max_recursion_depth' in depth_data.columns:
            for alg in recursive_algs:
                alg_data = depth_data[depth_data['algorithm'] == alg]
                if not alg_data.empty:
                    ax2.plot(alg_data['input_size'], alg_data['max_recursion_depth'], 
                           marker='o', label=alg, linewidth=2)
            
            ax2.set_xlabel('Input Size')
            ax2.set_ylabel('Max Recursion Depth')
            ax2.set_title('Recursion Depth')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Memory efficiency
        ax3 = fig.add_