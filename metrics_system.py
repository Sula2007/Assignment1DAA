"""
Metrics collection system for analyzing algorithm performance.
Tracks time, recursion depth, comparisons, and memory allocations.
"""

import time
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

@dataclass
class AlgorithmMetrics:
    """Container for algorithm performance metrics."""
    algorithm_name: str
    input_size: int
    execution_time: float
    comparisons: int
    allocations: int
    max_recursion_depth: int
    additional_data: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """Thread-safe metrics collector for algorithm analysis."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()
    
    def reset(self):
        """Reset all metrics to initial values."""
        with self._lock:
            self.comparisons = 0
            self.allocations = 0
            self.max_depth = 0
            self.start_time = None
            self.end_time = None
    
    def increment_comparisons(self, count: int = 1):
        """Increment comparison counter."""
        with self._lock:
            self.comparisons += count
    
    def record_allocation(self, size: int):
        """Record a memory allocation of given size."""
        with self._lock:
            self.allocations += size
    
    def update_max_depth(self, depth: int):
        """Update maximum recursion depth if current depth is greater."""
        with self._lock:
            self.max_depth = max(self.max_depth, depth)
    
    def start_timing(self):
        """Start timing measurement."""
        with self._lock:
            self.start_time = time.perf_counter()
    
    def end_timing(self):
        """End timing measurement."""
        with self._lock:
            self.end_time = time.perf_counter()
    
    def get_execution_time(self) -> float:
        """Get execution time in seconds."""
        with self._lock:
            if self.start_time is not None and self.end_time is not None:
                return self.end_time - self.start_time
            return 0.0
    
    def get_metrics_dict(self, algorithm_name: str = "", input_size: int = 0) -> Dict[str, Any]:
        """Get current metrics as dictionary."""
        with self._lock:
            return {
                'algorithm': algorithm_name,
                'input_size': input_size,
                'time': self.get_execution_time(),
                'comparisons': self.comparisons,
                'allocations': self.allocations,
                'max_depth': self.max_depth
            }


class BenchmarkRunner:
    """Run benchmarks and collect comprehensive metrics."""
    
    def __init__(self):
        self.results: List[AlgorithmMetrics] = []
    
    def run_algorithm_benchmark(self, 
                              algorithm_func, 
                              test_data: List[Any],
                              algorithm_name: str,
                              num_runs: int = 5) -> List[AlgorithmMetrics]:
        """
        Run benchmarks for an algorithm with multiple test datasets.
        
        Args:
            algorithm_func: Function that returns (result, metrics_dict)
            test_data: List of test inputs
            algorithm_name: Name of the algorithm
            num_runs: Number of runs per test for averaging
            
        Returns:
            List of AlgorithmMetrics for each test case
        """
        benchmark_results = []
        
        for i, data