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
            'Qu