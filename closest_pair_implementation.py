"""
Closest Pair of Points Algorithm (2D, O(n log n))

Divide and conquer approach:
1. Sort points by x-coordinate
2. Recursively find closest pairs in left and right halves
3. Check for closer pairs crossing the dividing line (strip check)
4. Classic 7-8 neighbor scan optimization in the strip
"""

import math
import time
from typing import List, Tuple, NamedTuple
from ..utils.metrics import MetricsCollector

class Point(NamedTuple):
    """Represents a 2D point."""
    x: float
    y: float

class PointPair(NamedTuple):
    """Represents a pair of points with their distance."""
    p1: Point
    p2: Point
    distance: float

class ClosestPair:
    def __init__(self, collect_metrics: bool = True):
        """
        Initialize Closest Pair algorithm.
        
        Args:
            collect_metrics: Whether to collect performance metrics
        """
        self.collector = MetricsCollector() if collect_metrics else None
    
    def euclidean_distance(self, p1: Point, p2: Point) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            p1: First point
            p2: Second point
            
        Returns:
            Euclidean distance
        """
        if self.collector:
            self.collector.increment_comparisons()  # Count distance calculations
        
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def brute_force_closest(self, points: List[Point]) -> PointPair:
        """
        Brute force closest pair for small point sets (n ≤ 3).
        
        Args:
            points: List of points
            
        Returns:
            Closest pair with distance
        """
        n = len(points)
        if n < 2:
            return PointPair(Point(0, 0), Point(0, 0), float('inf'))
        
        min_dist = float('inf')
        closest_pair = None
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.euclidean_distance(points[i], points[j])
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = PointPair(points[i], points[j], dist)
        
        return closest_pair
    
    def strip_closest(self, strip: List[Point], min_dist: float) -> PointPair:
        """
        Find closest pair in strip around dividing line.
        Uses the key optimization that we only need to check 7-8 neighbors.
        
        Args:
            strip: Points in strip, sorted by y-coordinate
            min_dist: Current minimum distance
            
        Returns:
            Closest pair in strip (or original min_dist pair if none closer)
        """
        strip_min = min_dist
        closest_pair = PointPair(Point(0, 0), Point(0, 0), min_dist)
        
        n = len(strip)
        
        # For each point, check at most 7-8 subsequent points
        for i in range(n):
            j = i + 1
            # Key insight: if points are more than min_dist apart in y,
            # they can't be closer than min_dist in total distance
            while j < n and (strip[j].y - strip[i].y) < strip_min:
                dist = self.euclidean_distance(strip[i], strip[j])
                if dist < strip_min:
                    strip_min = dist
                    closest_pair = PointPair(strip[i], strip[j], dist)
                j += 1
                
                # Optimization: at most 7-8 points need to be checked
                # This is guaranteed by geometric properties
                if j - i > 8:
                    break
        
        return closest_pair
    
    def _closest_pair_recursive(self, points_x: List[Point], points_y: List[Point], depth: int) -> PointPair:
        """
        Recursive divide and conquer closest pair algorithm.
        
        Args:
            points_x: Points sorted by x-coordinate
            points_y: Points sorted by y-coordinate  
            depth: Current recursion depth
            
        Returns:
            Closest pair in the point set
        """
        if self.collector:
            self.collector.update_max_depth(depth)
        
        n = len(points_x)
        
        # Base case: use brute force for small sets
        if n <= 3:
            return self.brute_force_closest(points_x)
        
        # Divide points by x-coordinate
        mid = n // 2
        midpoint = points_x[mid]
        
        # Split points by x-coordinate
        left_x = points_x[:mid]
        right_x = points_x[mid:]
        
        # Split points_y into left and right based on x-coordinate
        left_y = []
        right_y = []
        
        for point in points_y:
            if point.x <= midpoint.x:
                left_y.append(point)
            else:
                right_y.append(point)
        
        # Recursively find closest pairs in left and right halves
        left_closest = self._closest_pair_recursive(left_x, left_y, depth + 1)
        right_closest = self._closest_pair_recursive(right_x, right_y, depth + 1)
        
        # Find the minimum of the two halves
        if left_closest.distance <= right_closest.distance:
            min_dist = left_closest.distance
            current_closest = left_closest
        else:
            min_dist = right_closest.distance
            current_closest = right_closest
        
        # Create strip of points close to the dividing line
        strip = []
        for point in points_y:
            if abs(point.x - midpoint.x) < min_dist:
                strip.append(point)
        
        if self.collector:
            self.collector.record_allocation(len(strip))  # Strip allocation
        
        # Find closest pair in strip
        strip_closest = self.strip_closest(strip, min_dist)
        
        # Return overall closest pair
        if strip_closest.distance < current_closest.distance:
            return strip_closest
        else:
            return current_closest
    
    def find_closest_pair(self, points: List[Tuple[float, float]]) -> Tuple[PointPair, dict]:
        """
        Find closest pair of points in 2D plane.
        
        Args:
            points: List of (x, y) coordinate tuples
            
        Returns:
            Tuple of (closest_pair, metrics_dict)
        """
        if len(points) < 2:
            empty_pair = PointPair(Point(0, 0), Point(0, 0), float('inf'))
            return empty_pair, {}
        
        if self.collector:
            self.collector.reset()
            # Account for sorting allocations
            self.collector.record_allocation(len(points) * 2)  # Two sorted arrays
        
        # Convert to Point objects
        point_objects = [Point(x, y) for x, y in points]
        
        start_time = time.perf_counter()
        
        # Sort points by x and y coordinates
        points_x = sorted(point_objects, key=lambda p: p.x)
        points_y = sorted(point_objects, key=lambda p: p.y)
        
        # Find closest pair using divide and conquer
        result = self._closest_pair_recursive(points_x, points_y, 0)
        
        end_time = time.perf_counter()
        
        metrics = {}
        if self.collector:
            metrics = {
                'time': end_time - start_time,
                'comparisons': self.collector.comparisons,
                'allocations': self.collector.allocations,
                'max_depth': self.collector.max_depth,
                'algorithm': 'ClosestPair'
            }
        
        return result, metrics


class BruteForceClosestPair:
    """
    Brute force O(n²) closest pair algorithm for comparison.
    """
    
    def __init__(self, collect_metrics: bool = True):
        self.collector = MetricsCollector() if collect_metrics else None
    
    def euclidean_distance(self, p1: Point, p2: Point) -> float:
        """Calculate Euclidean distance between two points."""
        if self.collector:
            self.collector.increment_comparisons()
        
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def find_closest_pair(self, points: List[Tuple[float, float]]) -> Tuple[PointPair, dict]:
        """
        Find closest pair using brute force O(n²) algorithm.
        
        Args:
            points: List of (x, y) coordinate tuples
            
        Returns:
            Tuple of (closest_pair, metrics_dict)
        """
        if len(points) < 2:
            empty_pair = PointPair(Point(0, 0), Point(0, 0), float('inf'))
            return empty_pair, {}
        
        if self.collector:
            self.collector.reset()
            self.collector.record_allocation(len(points))
        
        point_objects = [Point(x, y) for x, y in points]
        
        start_time = time.perf_counter()
        
        min_dist = float('inf')
        closest_pair = None
        n = len(point_objects)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.euclidean_distance(point_objects[i], point_objects[j])
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = PointPair(point_objects[i], point_objects[j], dist)
        
        end_time = time.perf_counter()
        
        metrics = {}
        if self.collector:
            metrics = {
                'time': end_time - start_time,
                'comparisons': self.collector.comparisons,
                'allocations': self.collector.allocations,
                'max_depth': 1,
                'algorithm': 'BruteForceClosestPair'
            }
        
        return closest_pair, metrics


def closest_pair(points: List[Tuple[float, float]], use_divide_conquer: bool = True) -> PointPair:
    """
    Convenience function for finding closest pair.
    
    Args:
        points: List of (x, y) coordinate tuples
        use_divide_conquer: Use O(n log n) divide & conquer (True) or O(n²) brute force (False)
        
    Returns:
        Closest pair of points
    """
    if use_divide_conquer:
        finder = ClosestPair(collect_metrics=False)
    else:
        finder = BruteForceClosestPair(collect_metrics=False)
    
    result, _ = finder.find_closest_pair(points)
    return result


# Example usage and testing
if __name__ == "__main__":
    import random
    
    # Test cases
    test_cases = [
        # Simple cases
        [(0, 0), (1, 1), (2, 2)],
        [(0, 0), (3, 4), (1, 1)],  # Distance should be sqrt(2)
        [(0, 0), (1, 0), (0, 1), (1, 1)],  # Unit square
        
        # Edge cases
        [(0, 0), (1, 1)],  # Two points
        [(5, 5)],  # Single point
        [],  # Empty
        
        # Points with same coordinates
        [(1, 1), (1, 1), (2, 2)],  # Distance 0
        
        # Collinear points
        [(0, 0), (1, 0), (2, 0), (3, 0)],
    ]
    
    print("Testing Closest Pair Algorithm:")
    finder = ClosestPair(collect_metrics=True)
    brute_finder = BruteForceClosestPair(collect_metrics=True)
    
    for i, points in enumerate(test_cases):
        print(f"\nTest {i + 1}: {points}")
        
        if len(points) >= 2:
            # Test divide & conquer
            result_dc, metrics_dc = finder.find_closest_pair(points)
            print(f"  Divide & Conquer: {result_dc.distance:.6f}")
            print(f"    Points: ({result_dc.p1.x}, {result_dc.p1.y}) - ({result_dc.p2.x}, {result_dc.p2.y})")
            print(f"    Metrics: comparisons={metrics_dc['comparisons']}, depth={metrics_dc['max_depth']}")
            
            # Test brute force for comparison
            result_bf, metrics_bf = brute_finder.find_closest_pair(points)
            print(f"  Brute Force: {result_bf.distance:.6f}")
            print(f"    Comparisons: {metrics_bf['comparisons']}")
            
            # Verify results match
            assert abs(result_dc.distance - result_bf.distance) < 1e-10, f"Results don't match for test {i+1}!"
        else:
            print(f"  Skipping - insufficient points")
    
    # Performance comparison on larger datasets
    print("\n" + "="*60)
    print("Performance Comparison on Random Points:")
    
    sizes = [10, 50, 100, 500]
    random.seed(42)  # For reproducible results
    
    for n in sizes:
        # Generate random points
        points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
        
        # Divide & conquer
        result_dc, metrics_dc = finder.find_closest_pair(points)
        
        # Brute force (only for smaller sizes)
        if n <= 100:
            result_bf, metrics_bf = brute_finder.find_closest_pair(points)
            
            print(f"\nSize {n}:")
            print(f"  D&C: {metrics_dc['time']:.6f}s, {metrics_dc['comparisons']} comparisons")
            print(f"  Brute: {metrics_bf['time']:.6f}s, {metrics_bf['comparisons']} comparisons")
            print(f"  Speedup: {metrics_bf['time']/metrics_dc['time']:.2f}x")
            
            assert abs(result_dc.distance - result_bf.distance) < 1e-10, f"Results don't match for size {n}!"
        else:
            print(f"\nSize {n}:")
            print(f"  D&C: {metrics_dc['time']:.6f}s, {metrics_dc['comparisons']} comparisons")
            print(f"  (Brute force skipped - too slow)")
    
    print("\nAll tests passed! ✅")
    print("Divide & Conquer shows significant speedup for larger datasets!")