"""
Deterministic Select (Median-of-Medians) Algorithm
Guarantees O(n) worst-case time complexity for finding k-th order statistic.

Key optimizations:
- Groups of 5 for median calculation
- Recurse only into needed side
- Prefer recursing into smaller partition
- In-place partitioning
"""

import time
from typing import List, Tuple, Optional
from ..utils.metrics import MetricsCollector

class DeterministicSelect:
    def __init__(self, collect_metrics: bool = True):
        """
        Initialize Deterministic Select algorithm.
        
        Args:
            collect_metrics: Whether to collect performance metrics
        """
        self.collector = MetricsCollector() if collect_metrics else None
    
    def insertion_sort_median(self, arr: List, start: int, end: int) -> int:
        """
        Sort small group (≤5 elements) and return median index.
        
        Args:
            arr: Array containing the group
            start: Start index of group
            end: End index of group (inclusive)
            
        Returns:
            Index of median element
        """
        # Simple insertion sort for small group
        for i in range(start + 1, end + 1):
            key = arr[i]
            j = i - 1
            
            while j >= start and arr[j] > key:
                if self.collector:
                    self.collector.increment_comparisons()
                arr[j + 1] = arr[j]
                j -= 1
            
            arr[j + 1] = key
        
        # Return median index
        return start + (end - start) // 2
    
    def median_of_medians(self, arr: List, start: int, end: int, depth: int) -> int:
        """
        Find median-of-medians pivot for guaranteed good splits.
        
        Args:
            arr: Array to find pivot in
            start: Start index
            end: End index (inclusive)
            depth: Current recursion depth
            
        Returns:
            Index of median-of-medians element
        """
        if self.collector:
            self.collector.update_max_depth(depth)
        
        n = end - start + 1
        
        # Base case: small group
        if n <= 5:
            return self.insertion_sort_median(arr, start, end)
        
        # Divide into groups of 5 and find medians
        medians = []
        for i in range(start, end + 1, 5):
            group_end = min(i + 4, end)
            median_idx = self.insertion_sort_median(arr, i, group_end)
            medians.append(arr[median_idx])
        
        if self.collector:
            self.collector.record_allocation(len(medians))
        
        # Find median of medians recursively
        if len(medians) == 1:
            # Find index of this median in original array
            for i in range(start, end + 1):
                if arr[i] == medians[0]:
                    return i
        
        # Recursively find median of medians
        mom_value = self._select_recursive(medians, 0, len(medians) - 1, 
                                         len(medians) // 2, depth + 1)
        
        # Find index of median-of-medians in original array
        for i in range(start, end + 1):
            if arr[i] == mom_value:
                return i
        
        # This shouldn't happen with correct implementation
        return start
    
    def partition_around_pivot(self, arr: List, start: int, end: int, pivot_idx: int) -> int:
        """
        Partition array around given pivot element.
        
        Args:
            arr: Array to partition
            start: Start index
            end: End index (inclusive)
            pivot_idx: Index of pivot element
            
        Returns:
            Final position of pivot after partitioning
        """
        # Move pivot to end
        arr[pivot_idx], arr[end] = arr[end], arr[pivot_idx]
        pivot_value = arr[end]
        
        # Standard partitioning (Lomuto scheme)
        i = start - 1
        
        for j in range(start, end):
            if self.collector:
                self.collector.increment_comparisons()
            
            if arr[j] <= pivot_value:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        # Place pivot in correct position
        arr[i + 1], arr[end] = arr[end], arr[i + 1]
        return i + 1
    
    def _select_recursive(self, arr: List, start: int, end: int, k: int, depth: int) -> any:
        """
        Recursive select function with guaranteed O(n) complexity.
        
        Args:
            arr: Array to select from
            start: Start index
            end: End index (inclusive)
            k: Target rank (0-indexed)
            depth: Current recursion depth
            
        Returns:
            k-th order statistic
        """
        if self.collector:
            self.collector.update_max_depth(depth)
        
        n = end - start + 1