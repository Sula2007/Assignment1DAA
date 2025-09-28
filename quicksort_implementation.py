"""
Robust QuickSort implementation with optimizations:
- Randomized pivot selection
- Tail recursion optimization (recurse on smaller partition)
- Bounded stack depth ≈ O(log n) typical case
- Metrics collection for analysis
"""

import random
import time
from typing import List, Tuple
from ..utils.metrics import MetricsCollector

class QuickSort:
    def __init__(self, collect_metrics: bool = True, random_seed: int = None):
        """
        Initialize QuickSort with optimizations.
        
        Args:
            collect_metrics: Whether to collect performance metrics
            random_seed: Seed for reproducible randomization (for testing)
        """
        self.collector = MetricsCollector() if collect_metrics else None
        if random_seed is not None:
            random.seed(random_seed)
    
    def randomized_partition(self, arr: List, low: int, high: int) -> int:
        """
        Partition with randomized pivot selection.
        
        Args:
            arr: Array to partition
            low: Lower bound
            high: Upper bound
            
        Returns:
            Pivot position after partitioning
        """
        # Random pivot selection
        pivot_idx = random.randint(low, high)
        arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]
        
        return self.partition(arr, low, high)
    
    def partition(self, arr: List, low: int, high: int) -> int:
        """
        Lomuto partition scheme.
        
        Args:
            arr: Array to partition
            low: Lower bound
            high: Upper bound (pivot element)
            
        Returns:
            Final position of pivot
        """
        pivot = arr[high]
        i = low - 1  # Index of smaller element
        
        for j in range(low, high):
            if self.collector:
                self.collector.increment_comparisons()
            
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    def _quicksort_optimized(self, arr: List, low: int, high: int, depth: int) -> None:
        """
        Optimized quicksort with tail recursion elimination.
        Recurse on smaller partition, iterate on larger one.
        
        Args:
            arr: Array to sort
            low: Lower bound
            high: Upper bound
            depth: Current recursion depth
        """
        while low < high:
            if self.collector:
                self.collector.update_max_depth(depth)
            
            # Partition and get pivot position
            pivot_pos = self.randomized_partition(arr, low, high)
            
            # Calculate sizes of partitions
            left_size = pivot_pos - low
            right_size = high - pivot_pos
            
            # Recurse on smaller partition, iterate on larger
            if left_size < right_size:
                # Left partition is smaller - recurse on it
                self._quicksort_optimized(arr, low, pivot_pos - 1, depth + 1)
                # Continue with right partition (tail recursion elimination)
                low = pivot_pos + 1
            else:
                # Right partition is smaller - recurse on it
                self._quicksort_optimized(arr, pivot_pos + 1, high, depth + 1)
                # Continue with left partition (tail recursion elimination)
                high = pivot_pos - 1
    
    def sort(self, arr: List) -> Tuple[List, dict]:
        """
        Sort array using optimized quicksort.
        
        Args:
            arr: Array to sort
            
        Returns:
            Tuple of (sorted_array, metrics_dict)
        """
        if not arr:
            return arr, {}
        
        if self.collector:
            self.collector.reset()
            self.collector.record_allocation(len(arr))  # Copy allocation
        
        # Make a copy to avoid modifying original
        arr_copy = arr.copy()
        
        start_time = time.perf_counter()
        
        self._quicksort_optimized(arr_copy, 0, len(arr_copy) - 1, 0)
        
        end_time = time.perf_counter()
        
        metrics = {}
        if self.collector:
            metrics = {
                'time': end_time - start_time,
                'comparisons': self.collector.comparisons,
                'allocations': self.collector.allocations,
                'max_depth': self.collector.max_depth,
                'algorithm': 'QuickSort'
            }
        
        return arr_copy, metrics


class QuickSortThreeWay:
    """
    Three-way quicksort for arrays with many duplicate elements.
    Handles equal elements more efficiently.
    """
    
    def __init__(self, collect_metrics: bool = True):
        self.collector = MetricsCollector() if collect_metrics else None
    
    def three_way_partition(self, arr: List, low: int, high: int) -> Tuple[int, int]:
        """
        Three-way partitioning: < pivot | = pivot | > pivot
        
        Returns:
            Tuple of (lt, gt) where:
            - arr[low:lt] < pivot
            - arr[lt:gt+1] = pivot  
            - arr[gt+1:high+1] > pivot
        """
        pivot = arr[low]
        lt = low      # arr[low:lt] < pivot
        i = low + 1   # arr[lt:i] = pivot
        gt = high     # arr[gt+1:high+1] > pivot
        
        while i <= gt:
            if self.collector:
                self.collector.increment_comparisons()
            
            if arr[i] < pivot:
                arr[lt], arr[i] = arr[i], arr[lt]
                lt += 1
                i += 1
            elif arr[i] > pivot:
                arr[i], arr[gt] = arr[gt], arr[i]
                gt -= 1
                # Don't increment i - need to check swapped element
            else:
                i += 1  # arr[i] == pivot
        
        return lt, gt
    
    def _quicksort_3way(self, arr: List, low: int, high: int, depth: int) -> None:
        """Three-way quicksort recursive function."""
        if low >= high:
            return
        
        if self.collector:
            self.collector.update_max_depth(depth)
        
        # Random pivot selection
        pivot_idx = random.randint(low, high)
        arr[low], arr[pivot_idx] = arr[pivot_idx], arr[low]
        
        lt, gt = self.three_way_partition(arr, low, high)
        
        self._quicksort_3way(arr, low, lt - 1, depth + 1)
        self._quicksort_3way(arr, gt + 1, high, depth + 1)
    
    def sort(self, arr: List) -> Tuple[List, dict]:
        """Sort using three-way quicksort."""
        if not arr:
            return arr, {}
        
        if self.collector:
            self.collector.reset()
            self.collector.record_allocation(len(arr))
        
        arr_copy = arr.copy()
        start_time = time.perf_counter()
        
        self._quicksort_3way(arr_copy, 0, len(arr_copy) - 1, 0)
        
        end_time = time.perf_counter()
        
        metrics = {}
        if self.collector:
            metrics = {
                'time': end_time - start_time,
                'comparisons': self.collector.comparisons,
                'allocations': self.collector.allocations,
                'max_depth': self.collector.max_depth,
                'algorithm': 'QuickSort3Way'
            }
        
        return arr_copy, metrics


def quicksort(arr: List, three_way: bool = False) -> List:
    """
    Convenience function for quicksort.
    
    Args:
        arr: Array to sort
        three_way: Use three-way partitioning for duplicate-heavy arrays
        
    Returns:
        Sorted array
    """
    if three_way:
        sorter = QuickSortThreeWay(collect_metrics=False)
    else:
        sorter = QuickSort(collect_metrics=False)
    
    result, _ = sorter.sort(arr)
    return result


# Example usage and testing
if __name__ == "__main__":
    # Test arrays with different characteristics
    test_arrays = [
        [64, 34, 25, 12, 22, 11, 90],           # Random
        [5, 2, 4, 6, 1, 3],                     # Small
        [1],                                     # Single element
        [],                                      # Empty
        [3, 3, 3, 3, 3, 3],                    # All duplicates
        list(range(100, 0, -1)),                # Reverse sorted
        [1, 3, 2, 3, 1, 2, 3, 1, 2],          # Many duplicates
    ]
    
    # Test regular quicksort
    print("Testing Regular QuickSort:")
    sorter = QuickSort(collect_metrics=True, random_seed=42)
    
    for i, arr in enumerate(test_arrays):
        print(f"\nTest {i + 1}: {arr[:10]}{'...' if len(arr) > 10 else ''}")
        sorted_arr, metrics = sorter.sort(arr)
        print(f"Sorted: {sorted_arr[:10]}{'...' if len(sorted_arr) > 10 else ''}")
        print(f"Metrics: {metrics}")
        
        # Verify correctness
        assert sorted_arr == sorted(arr), f"Test {i + 1} failed!"
    
    # Test three-way quicksort on duplicate-heavy array
    print("\n" + "="*50)
    print("Testing Three-Way QuickSort on duplicates:")
    
    duplicate_array = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5] * 10
    
    regular_sorter = QuickSort(collect_metrics=True)
    three_way_sorter = QuickSortThreeWay(collect_metrics=True)
    
    _, regular_metrics = regular_sorter.sort(duplicate_array)
    _, three_way_metrics = three_way_sorter.sort(duplicate_array)
    
    print(f"Regular QuickSort - Comparisons: {regular_metrics['comparisons']}, "
          f"Depth: {regular_metrics['max_depth']}")
    print(f"3-Way QuickSort - Comparisons: {three_way_metrics['comparisons']}, "
          f"Depth: {three_way_metrics['max_depth']}")
    
    print("\nAll tests passed! ✅")
