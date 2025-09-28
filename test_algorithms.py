"""
Comprehensive test suite for divide-and-conquer algorithms.
Tests correctness, edge cases, and basic performance characteristics.
"""

import pytest
import random
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.mergesort import MergeSort, merge_sort
from algorithms.quicksort import QuickSort, QuickSortThreeWay, quicksort
from algorithms.select import DeterministicSelect, QuickSelect, select, find_median
from algorithms.closest_pair import ClosestPair, BruteForceClosestPair, closest_pair
from utils.generators import ArrayGenerator, PointGenerator, DataPattern

class TestMergeSort:
    """Test suite for MergeSort implementation."""
    
    @pytest.fixture
    def merger(self):
        return MergeSort(collect_metrics=True)
    
    def test_empty_array(self, merger):
        """Test sorting empty array."""
        result, metrics = merger.sort([])
        assert result == []
        assert metrics['comparisons'] == 0
    
    def test_single_element(self, merger):
        """Test sorting single element array."""
        result, metrics = merger.sort([42])
        assert result == [42]
        assert metrics['max_depth'] >= 0
    
    def test_sorted_array(self, merger):
        """Test already sorted array."""
        arr = [1, 2, 3, 4, 5]
        result, metrics = merger.sort(arr)
        assert result == [1, 2, 3, 4, 5]
        assert result == sorted(arr)
    
    def test_reverse_sorted(self, merger):
        """Test reverse sorted array."""
        arr = [5, 4, 3, 2, 1]
        result, metrics = merger.sort(arr)
        assert result == [1, 2, 3, 4, 5]
    
    def test_duplicates(self, merger):
        """Test array with duplicate elements."""
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        result, metrics = merger.sort(arr)
        assert result == sorted(arr)
        assert len(result) == len(arr)
    
    def test_random_arrays(self, merger):
        """Test multiple random arrays."""
        random.seed(42)
        for size in [10, 50, 100]:
            arr = [random.randint(0, 1000) for _ in range(size)]
            result, metrics = merger.sort(arr)
            expected = sorted(arr)
            assert result == expected
            assert metrics['algorithm'] == 'MergeSort'
    
    def test_cutoff_optimization(self):
        """Test cutoff optimization works correctly."""
        # Test with different cutoff values
        for cutoff in [5, 10, 20]:
            merger = MergeSort(cutoff=cutoff, collect_metrics=True)
            arr = [random.randint(0, 100) for _ in range(50)]
            result, _ = merger.sort(arr)
            assert result == sorted(arr)
    
    def test_convenience_function(self):
        """Test convenience function."""
        arr = [64, 34, 25, 12, 22, 11, 90]
        result = merge_sort(arr)
        assert result == sorted(arr)


class TestQuickSort:
    """Test suite for QuickSort implementations."""
    
    @pytest.fixture
    def quick_sorter(self):
        return QuickSort(collect_metrics=True, random_seed=42)
    
    @pytest.fixture
    def three_way_sorter(self):
        return QuickSortThreeWay(collect_metrics=True)
    
    def test_empty_array(self, quick_sorter):
        """Test sorting empty array."""
        result, metrics = quick_sorter.sort([])
        assert result == []
    
    def test_single_element(self, quick_sorter):
        """Test sorting single element."""
        result, metrics = quick_sorter.sort([42])
        assert result == [42]
    
    def test_random_arrays(self, quick_sorter):
        """Test random arrays."""
        random.seed(42)
        for size in [10, 50, 100]:
            arr = [random.randint(0, 1000) for _ in range(size)]
            result, metrics = quick_sorter.sort(arr)
            assert result == sorted(arr)
            # Check that recursion depth is reasonable
            assert metrics['max_depth'] <= 2 * size  # Very generous upper bound
    
    def test_worst_case_input(self, quick_sorter):
        """Test on sorted array (potential worst case)."""
        arr = list(range(100))
        result, metrics = quick_sorter.sort(arr)
        assert result == arr
        # With randomization, should still have reasonable depth
        assert metrics['max_depth'] < 100
    
    def test_three_way_partitioning(self, three_way_sorter):
        """Test three-way quicksort on duplicate-heavy array."""
        arr = [3, 3, 3, 1, 1, 1, 2, 2, 2] * 5
        result, metrics = three_way_sorter.sort(arr)
        assert result == sorted(arr)
        assert metrics['algorithm'] == 'QuickSort3Way'
    
    def test_convenience_function(self):
        """Test convenience functions."""
        arr = [64, 34, 25, 12, 22, 11, 90]
        
        # Regular quicksort
        result1 = quicksort(arr, three_way=False)
        assert result1 == sorted(arr)
        
        # Three-way quicksort
        result2 = quicksort(arr, three_way=True)
        assert result2 == sorted(arr)


class TestSelect:
    """Test suite for selection algorithms."""
    
    @pytest.fixture
    def det_select(self):
        return DeterministicSelect(collect_metrics=True)
    
    @pytest.fixture
    def quick_select(self):
        return QuickSelect(collect_metrics=True)
    
    def test_empty_array(self, det_select):
        """Test selection on empty array."""
        result, _ = det_select.select([], 0)
        assert result is None
    
    def test_single_element(self, det_select):
        """Test selection on single element."""
        result, metrics = det_select.select([42], 0)
        assert result == 42
    
    def test_select_minimum(self, det_select):
        """Test selecting minimum element."""
        arr = [3, 6, 8, 10, 1, 2, 1]
        result, metrics = det_select.select(arr, 0)
        assert result == min(arr)
    
    def test_select_maximum(self, det_select):
        """Test selecting maximum element."""
        arr = [3, 6, 8, 10, 1, 2, 1]
        result, metrics = det_select.select(arr, len(arr) - 1)
        assert result == max(arr)
    
    def test_select_median(self, det_select):
        """Test selecting median element."""
        arr = [1, 2, 3, 4, 5]
        result, metrics = det_select.select(arr, 2)  # Middle element
        assert result == 3
    
    def test_select_various_k(self, det_select):
        """Test selection with various k values."""
        arr = [9, 3, 6, 2, 8, 1, 5, 7, 4]
        sorted_arr = sorted(arr)
        
        for k in range(len(arr)):
            result, _ = det_select.select(arr, k)
            assert result == sorted_arr[k]
    
    def test_deterministic_vs_randomized(self):
        """Compare deterministic and randomized select."""
        det_select = DeterministicSelect(collect_metrics=True)
        quick_select = QuickSelect(collect_metrics=True)
        
        arr = list(range(100, 0, -1))  # Worst case for quickselect
        k = 50
        
        det_result, det_metrics = det_select.select(arr, k)
        quick_result, quick_metrics = quick_select.select(arr, k)
        
        assert det_result == quick_result
        assert det_result == sorted(arr)[k]
        
        # Deterministic should have predictable depth
        assert det_metrics['max_depth'] > 0
    
    def test_find_median_function(self):
        """Test median finding function."""
        # Odd length
        arr1 = [1, 2, 3, 4, 5]
        median1, _ = find_median(arr1, deterministic=True)
        assert median1 == 3
        
        # Even length (returns lower middle)
        arr2 = [1, 2, 3, 4, 5, 6]
        median2, _ = find_median(arr2, deterministic=True)
        assert median2 in [3, 4]  # Either middle element is acceptable
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        arr = [9, 3, 6, 2, 8, 1, 5, 7, 4]
        
        # Test deterministic select
        result1 = select(arr, 4, deterministic=True)
        assert result1 == sorted(arr)[4]
        
        # Test randomized select
        result2 = select(arr, 4, deterministic=False)
        assert result2 == sorted(arr)[4]
    
    def test_boundary_conditions(self, det_select):
        """Test boundary conditions."""
        arr = [5, 2, 8, 1, 9]
        
        # Test invalid k values
        with pytest.raises(ValueError):
            det_select.select(arr, -1)
        
        with pytest.raises(ValueError):
            det_select.select(arr, len(arr))


class TestClosestPair:
    """Test suite for closest pair algorithms."""
    
    @pytest.fixture
    def closest_pair_finder(self):
        return ClosestPair(collect_metrics=True)
    
    @pytest.fixture
    def brute_force_finder(self):
        return BruteForceClosestPair(collect_metrics=True)
    
    def test_empty_points(self, closest_pair_finder):
        """Test with empty point set."""
        result, _ = closest_pair_finder.find_closest_pair([])
        assert result.distance == float('inf')
    
    def test_single_point(self, closest_pair_finder):
        """Test with single point."""
        result, _ = closest_pair_finder.find_closest_pair([(0, 0)])
        assert result.distance == float('inf')
    
    def test_