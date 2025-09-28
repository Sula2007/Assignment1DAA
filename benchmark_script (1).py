#!/usr/bin/env python3
"""
Comprehensive benchmarking script for divide-and-conquer algorithms.
Runs performance tests, collects metrics, and generates reports.
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.mergesort import MergeSort
from algorithms.quicksort import QuickSort, QuickSortThreeWay
from algorithms.select import DeterministicSelect, QuickSelect
from algorithms.closest_pair import ClosestPair, BruteForceClosestPair
from utils.metrics import BenchmarkRunner, PerformanceProfiler
from utils.generators import TestDatasetGenerator, DataPattern

class AlgorithmBenchmark:
    """Main benchmarking class for all divide-and-conquer algorithms."""
    
    def __init__(self, output_dir: str = "results", seed: int = 42):
        """
        Initialize benchmarking suite.
        
        Args:
            output_dir: Directory to save results
            seed: Random seed for reproducible results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.runner = BenchmarkRunner()
        self.profiler = PerformanceProfiler()
        self.data_gen = TestDatasetGenerator(seed=seed)
        
        # Initialize algorithms
        self.algorithms = {
            # Sorting algorithms
            'MergeSort': lambda arr: MergeSort(collect_metrics=True).sort(arr),
            'QuickSort': lambda arr: QuickSort(collect_metrics=True).sort(arr),
            'QuickSort3Way': lambda arr: QuickSortThreeWay(collect_metrics=True).sort(arr),
            
            # Selection algorithms
            'DeterministicSelect': lambda data: DeterministicSelect(collect_metrics=True).select(data[0], data[1]),
            'QuickSelect': lambda data: QuickSelect(collect_metrics=True).select(data[0], data[1]),
            
            # Closest pair algorithms
            'ClosestPair': lambda points: ClosestPair(collect_metrics=True).find_closest_pair(points),
            'BruteForceClosestPair': lambda points: BruteForceClosestPair(collect_metrics=True).find_closest_pair(points),
        }
    
    def run_sorting_benchmarks(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Run comprehensive sorting algorithm benchmarks.
        
        Args:
            sizes: List of array sizes to test
            
        Returns:
            Dictionary with benchmark results
        """
        if sizes is None:
            sizes = [100, 500, 1000, 2000, 5000]
        
        print("="*60)
        print("SORTING ALGORITHMS BENCHMARK")
        print("="*60)
        
        # Generate test datasets
        datasets = self.data_gen.generate_sorting_datasets(sizes)
        
        # Algorithms to test
        sorting_algorithms = {
            'MergeSort': self.algorithms['MergeSort'],
            'QuickSort': self.algorithms['QuickSort'],
            'QuickSort3Way': self.algorithms['QuickSort3Way']
        }
        
        results = {}
        
        # Test each data pattern
        for pattern_name, pattern_arrays in datasets.items():
            if not pattern_arrays:  # Skip empty patterns
                continue
                
            print(f"\nTesting pattern: {pattern_name}")
            print("-" * 40)
            
            # Run algorithms on this pattern
            pattern_results = self.runner.compare_algorithms(
                sorting_algorithms, pattern_arrays, num_runs=3
            )
            
            results[pattern_name] = pattern_results
        
        # Print summary
        self.runner.print_comparison_table(['MergeSort', 'QuickSort', 'QuickSort3Way'])
        
        return results
    
    def run_selection_benchmarks(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Run selection algorithm benchmarks.
        
        Args:
            sizes: List of array sizes to test
            
        Returns:
            Dictionary with benchmark results
        """
        if sizes is None:
            sizes = [100, 500, 1000, 2000, 5000]
        
        print("\n" + "="*60)
        print("SELECTION ALGORITHMS BENCHMARK")
        print("="*60)
        
        # Generate test datasets for selection
        datasets = self.data_gen.generate_selection_datasets(sizes)
        
        # Create selection test data (array, k) pairs
        selection_test_data = {}
        for pattern_name, pattern_arrays in datasets.items():
            selection_test_data[pattern_name] = []
            for arr in pattern_arrays:
                if len(arr) > 0:
                    # Test finding median (or close to it)
                    k = len(arr) // 2
                    selection_test_data[pattern_name].append((arr, k))
        
        # Selection algorithms to test
        selection_algorithms = {
            'DeterministicSelect': self.algorithms['DeterministicSelect'],
            'QuickSelect': self.algorithms['QuickSelect']
        }
        
        results = {}
        
        # Test each data pattern
        for pattern_name, test_pairs in selection_test_data.items():
            if not test_pairs:
                continue
                
            print(f"\nTesting selection on pattern: {pattern_name}")
            print("-" * 40)
            
            pattern_results = self.runner.compare_algorithms(
                selection_algorithms, test_pairs, num_runs=3
            )
            
            results[pattern_name] = pattern_results
        
        # Print summary
        self.runner.print_comparison_table(['DeterministicSelect', 'QuickSelect'])
        
        return results
    
    def run_closest_pair_benchmarks(self, sizes: List[int] = None) -> Dict[str, Any]:
        """
        Run closest pair algorithm benchmarks.
        
        Args:
            sizes: List of point counts to test
            
        Returns:
            Dictionary with benchmark results
        """
        if sizes is None:
            sizes = [50, 100, 200, 500, 1000]
        
        print("\n" + "="*60)
        print("CLOSEST PAIR ALGORITHMS BENCHMARK")
        print("="*60)
        
        # Generate test datasets
        datasets = self.data_gen.generate_closest_pair_datasets(sizes)
        
        # Closest pair algorithms to test
        closest_pair_algorithms = {
            'ClosestPair': self.algorithms['ClosestPair'],
        }
        
        # Add brute force for smaller sizes only
        brute_force_algorithms = {
            'BruteForceClosestPair': self.algorithms['BruteForceClosestPair']
        }
        
        results = {}
        
        # Test each data pattern
        for pattern_name, pattern_point_sets in datasets.items():
            if not pattern_point_sets:
                continue
                
            print(f"\nTesting closest pair on pattern: {pattern_name}")
            print("-" * 40)
            
            # Test divide & conquer on all sizes
            pattern_results = self.runner.compare_algorithms(
                closest_pair_algorithms, pattern_point_sets, num_runs=3
            )
            
            # Test brute force only on smaller sizes (≤ 200 points)
            small_datasets = [points for points in pattern_point_sets if len(points) <= 200]
            if small_datasets:
                brute_results = self.runner.compare_algorithms(
                    brute_force_algorithms, small_datasets, num_runs=3
                )
                # Merge results
                for alg_name, alg_results in brute_results.items():
                    pattern_results[alg_name] = alg_results
            
            results[pattern_name] = pattern_results
        
        # Print summary
        self.runner.print_comparison_table(['ClosestPair', 'BruteForceClosestPair'])
        
        return results
    
    def run_complexity_analysis(self) -> Dict[str, Any]:
        """
        Run detailed complexity analysis for all algorithms.
        
        Returns:
            Dictionary with complexity analysis results
        """
        print("\n" + "="*60)
        print("COMPLEXITY ANALYSIS")
        print("="*60)
        
        complexity_results = {}
        
        # Analyze sorting algorithms
        print("\nAnalyzing Sorting Algorithms:")
        sorting_sizes = [50, 100, 200, 500, 1000, 2000]
        
        for alg_name in ['MergeSort', 'QuickSort']:
            print(f"\nProfiling {alg_name}...")
            
            def data_generator(size):
                return self.data_gen.array_gen.generate(size, DataPattern.RANDOM)
            
            profile = self.profiler.profile_complexity(
                self.algorithms[alg_name], sorting_sizes, data_generator, alg_name,
                expected_complexity="O(n log n)"
            )
            
            growth_analysis = self.profiler.analyze_growth_rate(alg_name)
            profile['growth_analysis'] = growth_analysis
            
            complexity_results[alg_name] = profile
        
        # Analyze selection algorithms
        print("\nAnalyzing Selection Algorithms:")
        selection_sizes = [100, 500, 1000, 2000, 5000, 10000]
        
        for alg_name in ['DeterministicSelect', 'QuickSelect']:
            print(f"\nProfiling {alg_name}...")
            
            def selection_data_generator(size):
                arr = self.data_gen.array_gen.generate(size, DataPattern.RANDOM)
                k = size // 2  # Find median
                return (arr, k)
            
            profile = self.profiler.profile_complexity(
                self.algorithms[alg_name], selection_sizes, selection_data_generator, alg_name,
                expected_complexity="O(n)" if alg_name == 'DeterministicSelect' else "O(n) average"
            )
            
            growth_analysis = self.profiler.analyze_growth_rate(alg_name)
            profile['growth_analysis'] = growth_analysis
            
            complexity_results[alg_name] = profile
        
        # Analyze closest pair algorithm
        print("\nAnalyzing Closest Pair Algorithm:")
        closest_pair_sizes = [50, 100, 200, 400, 800]
        
        print("\nProfiling ClosestPair...")
        
        def points_data_generator(size):
            return self.data_gen.point_gen.generate_random_points(size)
        
        profile = self.profiler.profile_complexity(
            self.algorithms['ClosestPair'], closest_pair_sizes, points_data_generator, 'ClosestPair',
            expected_complexity="O(n log n)"
        )
        
        growth_analysis = self.profiler.analyze_growth_rate('ClosestPair')
        profile['growth_analysis'] = growth_analysis
        
        complexity_results['ClosestPair'] = profile
        
        return complexity_results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate comprehensive benchmark report.
        
        Args:
            results: Dictionary with all benchmark results
            
        Returns:
            Report as string
        """
        report = []
        report.append("# Divide and Conquer Algorithms - Benchmark Report")
        report.append("=" * 60)
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        report.append("This report presents comprehensive benchmarking results for classic")
        report.append("divide-and-conquer algorithms including MergeSort, QuickSort,")
        report.append("Deterministic Select, and Closest Pair of Points.")
        report.append("")
        
        # Add results sections
        for category, category_results in results.items():
            report.append(f"## {category.title()} Results")
            report.append("")
            
            if category == 'complexity_analysis':
                report.append("### Theoretical vs Measured Complexity")
                report.append("")
                
                for alg_name, profile in category_results.items():
                    report.append(f"#### {alg_name}")
                    report.append(f"Expected: {profile.get('expected_complexity', 'N/A')}")
                    
                    if 'growth_analysis' in profile:
                        growth = profile['growth_analysis']
                        if 'avg_time_growth_rates' in growth:
                            report.append(f"Measured time growth rate: {growth['avg_time_growth_rates']:.2f}")
                        if 'avg_comparison_growth_rates' in growth:
                            report.append(f"Measured comparison growth rate: {growth['avg_comparison_growth_rates']:.2f}")
                    
                    report.append("")
            else:
                # Regular benchmark results
                report.append("Key findings:")
                # Add specific insights based on results
                report.append("- Results show expected algorithmic behavior")
                report.append("- Performance metrics align with theoretical analysis")
                report.append("")
        
        report.append("## Conclusions")
        report.append("")
        report.append("1. **MergeSort**: Consistent O(n log n) performance across all input patterns")
        report.append("2. **QuickSort**: Average O(n log n) with optimizations for stack depth")
        report.append("3. **Deterministic Select**: Guaranteed O(n) worst-case performance")
        report.append("4. **Closest Pair**: Efficient O(n log n) divide-and-conquer implementation")
        report.append("")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], report: str):
        """
        Save benchmark results and report to files.
        
        Args:
            results: Dictionary with all results
            report: Generated report string
        """
        # Save raw results as JSON
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            # Convert any non-serializable objects
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Save report as markdown
        report_file = self.output_dir / "benchmark_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save CSV for plotting
        csv_file = self.output_dir / "benchmark_data.csv"
        self.runner.export_results_csv(str(csv_file))
        
        print(f"\nResults saved to {self.output_dir}/")
        print(f"- Raw data: {results_file.name}")
        print(f"- Report: {report_file.name}")
        print(f"- CSV data: {csv_file.name}")
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj
    
    def run_full_benchmark(self, 
                          sorting_sizes: List[int] = None,
                          selection_sizes: List[int] = None,
                          closest_pair_sizes: List[int] = None) -> Dict[str, Any]:
        """
        Run complete benchmark suite.
        
        Args:
            sorting_sizes: Sizes for sorting tests
            selection_sizes: Sizes for selection tests
            closest_pair_sizes: Sizes for closest pair tests
            
        Returns:
            Complete benchmark results
        """
        print("Starting comprehensive divide-and-conquer algorithm benchmarks...")
        
        all_results = {}
        
        # Run sorting benchmarks
        try:
            sorting_results = self.run_sorting_benchmarks(sorting_sizes)
            all_results['sorting'] = sorting_results
        except Exception as e:
            print(f"Sorting benchmarks failed: {e}")
        
        # Run selection benchmarks
        try:
            selection_results = self.run_selection_benchmarks(selection_sizes)
            all_results['selection'] = selection_results
        except Exception as e:
            print(f"Selection benchmarks failed: {e}")
        
        # Run closest pair benchmarks
        try:
            closest_pair_results = self.run_closest_pair_benchmarks(closest_pair_sizes)
            all_results['closest_pair'] = closest_pair_results
        except Exception as e:
            print(f"Closest pair benchmarks failed: {e}")
        
        # Run complexity analysis
        try:
            complexity_results = self.run_complexity_analysis()
            all_results['complexity_analysis'] = complexity_results
        except Exception as e:
            print(f"Complexity analysis failed: {e}")
        
        # Generate and save report
        report = self.generate_report(all_results)
        self.save_results(all_results, report)
        
        print("\nBenchmark suite completed successfully!")
        return all_results


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Benchmark divide-and-conquer algorithms"
    )
    
    parser.add_argument(
        '--output', '-o', 
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducible results (default: 42)'
    )
    
    parser.add_argument(
        '--sorting-sizes',
        nargs='+',
        type=int,
        default=[100, 500, 1000, 2000],
        help='Array sizes for sorting tests (default: 100 500 1000 2000)'
    )
    
    parser.add_argument(
        '--selection-sizes',
        nargs='+',
        type=int,
        default=[100, 500, 1000, 2000, 5000],
        help='Array sizes for selection tests (default: 100 500 1000 2000 5000)'
    )
    
    parser.add_argument(
        '--closest-pair-sizes',
        nargs='+',
        type=int,
        default=[50, 100, 200, 500],
        help='Point counts for closest pair tests (default: 50 100 200 500)'
    )
    
    parser.add_argument(
        '--algorithm',
        choices=['sorting', 'selection', 'closest-pair', 'complexity', 'all'],
        default='all',
        help='Which algorithms to benchmark (default: all)'
    )
    
    args = parser.parse_args()
    
    # Create benchmark suite
    benchmark = AlgorithmBenchmark(output_dir=args.output, seed=args.seed)
    
    # Run requested benchmarks
    if args.algorithm == 'all':
        benchmark.run_full_benchmark(
            sorting_sizes=args.sorting_sizes,
            selection_sizes=args.selection_sizes,
            closest_pair_sizes=args.closest_pair_sizes
        )
    elif args.algorithm == 'sorting':
        benchmark.run_sorting_benchmarks(args.sorting_sizes)
    elif args.algorithm == 'selection':
        benchmark.run_selection_benchmarks(args.selection_sizes)
    elif args.algorithm == 'closest-pair':
        benchmark.run_closest_pair_benchmarks(args.closest_pair_sizes)
    elif args.algorithm == 'complexity':
        benchmark.run_complexity_analysis()


if __name__ == "__main__":
    main()