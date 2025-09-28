"""
Test data generators for algorithm benchmarking.
Generates various types of datasets to test different algorithm behaviors.
"""

import random
import math
from typing import List, Tuple, Any, Callable
from enum import Enum

class DataPattern(Enum):
    """Enumeration of different data patterns for testing."""
    RANDOM = "random"
    SORTED = "sorted"
    REVERSE_SORTED = "reverse_sorted"
    NEARLY_SORTED = "nearly_sorted"
    DUPLICATE_HEAVY = "duplicate_heavy"
    ALL_SAME = "all_same"
    ALTERNATING = "alternating"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"

class ArrayGenerator:
    """Generator for various array patterns."""
    
    def __init__(self, seed: int = None):
        """
        Initialize generator with optional seed for reproducibility.
        
        Args:
            seed: Random seed for reproducible generation
        """
        if seed is not None:
            random.seed(seed)
    
    def generate(self, size: int, pattern: DataPattern, **kwargs) -> List[int]:
        """
        Generate array of specified size and pattern.
        
        Args:
            size: Size of array to generate
            pattern: Type of pattern to generate
            **kwargs: Additional pattern-specific parameters
            
        Returns:
            Generated array
        """
        if size <= 0:
            return []
        
        generators = {
            DataPattern.RANDOM: self._generate_random,
            DataPattern.SORTED: self._generate_sorted,
            DataPattern.REVERSE_SORTED: self._generate_reverse_sorted,
            DataPattern.NEARLY_SORTED: self._generate_nearly_sorted,
            DataPattern.DUPLICATE_HEAVY: self._generate_duplicate_heavy,
            DataPattern.ALL_SAME: self._generate_all_same,
            DataPattern.ALTERNATING: self._generate_alternating,
            DataPattern.GAUSSIAN: self._generate_gaussian,
            DataPattern.EXPONENTIAL: self._generate_exponential
        }
        
        return generators[pattern](size, **kwargs)
    
    def _generate_random(self, size: int, min_val: int = 0, max_val: int = 1000) -> List[int]:
        """Generate uniformly random array."""
        return [random.randint(min_val, max_val) for _ in range(size)]
    
    def _generate_sorted(self, size: int, start: int = 0, step: int = 1) -> List[int]:
        """Generate sorted array."""
        return [start + i * step for i in range(size)]
    
    def _generate_reverse_sorted(self, size: int, start: int = None, step: int = 1) -> List[int]:
        """Generate reverse sorted array."""
        if start is None:
            start = size * step
        return [start - i * step for i in range(size)]
    
    def _generate_nearly_sorted(self, size: int, swap_percentage: float = 0.05, 
                               min_val: int = 0, max_val: int = 1000) -> List[int]:
        """Generate nearly sorted array with some random swaps."""
        arr = self._generate_sorted(size, min_val, (max_val - min_val) // size if size > 0 else 1)
        
        # Perform random swaps
        num_swaps = max(1, int(size * swap_percentage))
        for _ in range(num_swaps):
            i, j = random.randint(0, size-1), random.randint(0, size-1)
            arr[i], arr[j] = arr[j], arr[i]
        
        return arr
    
    def _generate_duplicate_heavy(self, size: int, num_unique: int = None, 
                                 min_val: int = 0, max_val: int = 100) -> List[int]:
        """Generate array with many duplicate values."""
        if num_unique is None:
            num_unique = max(1, size // 10)  # Default: 10% unique values
        
        unique_vals = [random.randint(min_val, max_val) for _ in range(num_unique)]
        return [random.choice(unique_vals) for _ in range(size)]
    
    def _generate_all_same(self, size: int, value: int = None) -> List[int]:
        """Generate array with all identical values."""
        if value is None:
            value = random.randint(0, 100)
        return [value] * size
    
    def _generate_alternating(self, size: int, val1: int = 0, val2: int = 1) -> List[int]:
        """Generate alternating pattern array."""
        return [val1 if i % 2 == 0 else val2 for i in range(size)]
    
    def _generate_gaussian(self, size: int, mean: float = 500, std: float = 100) -> List[int]:
        """Generate array with Gaussian distribution."""
        return [max(0, int(random.gauss(mean, std))) for _ in range(size)]
    
    def _generate_exponential(self, size: int, lambda_param: float = 0.01) -> List[int]:
        """Generate array with exponential distribution."""
        return [int(random.expovariate(lambda_param)) for _ in range(size)]


class PointGenerator:
    """Generator for 2D point datasets."""
    
    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
    
    def generate_random_points(self, count: int, x_range: Tuple[float, float] = (0, 100),
                              y_range: Tuple[float, float] = (0, 100)) -> List[Tuple[float, float]]:
        """Generate random 2D points within specified ranges."""
        return [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(count)]
    
    def generate_clustered_points(self, count: int, num_clusters: int = 3,
                                 cluster_radius: float = 10.0,
                                 area_size: float = 100.0) -> List[Tuple[float, float]]:
        """Generate points clustered around random centers."""
        # Generate cluster centers
        centers = [(random.uniform(0, area_size), random.uniform(0, area_size)) 
                  for _ in range(num_clusters)]
        
        points = []
        points_per_cluster = count // num_clusters
        remaining_points = count % num_clusters
        
        for i, (cx, cy) in enumerate(centers):
            # Number of points in this cluster
            cluster_size = points_per_cluster + (1 if i < remaining_points else 0)
            
            # Generate points around cluster center
            for _ in range(cluster_size):
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0, cluster_radius)
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                points.append((x, y))
        
        return points
    
    def generate_grid_points(self, grid_size: int, spacing: float = 10.0,
                           noise: float = 0.0) -> List[Tuple[float, float]]:
        """Generate points on a regular grid with optional noise."""
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = i * spacing
                y = j * spacing
                
                # Add noise if specified
                if noise > 0:
                    x += random.uniform(-noise, noise)
                    y += random.uniform(-noise, noise)
                
                points.append((x, y))
        
        return points
    
    def generate_circle_points(self, count: int, center: Tuple[float, float] = (50, 50),
                              radius: float = 30.0, noise: float = 0.0) -> List[Tuple[float, float]]:
        """Generate points on or near a circle."""
        points = []
        cx, cy = center
        
        for i in range(count):
            angle = 2 * math.pi * i / count
            r = radius + random.uniform(-noise, noise) if noise > 0 else radius
            
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            points.append((x, y))
        
        return points
    
    def generate_line_points(self, count: int, start: Tuple[float, float] = (0, 0),
                           end: Tuple[float, float] = (100, 100),
                           noise: float = 0.0) -> List[Tuple[float, float]]:
        """Generate points on or near a line."""
        points = []
        x1, y1 = start
        x2, y2 = end
        
        for i in range(count):
            t = i / (count - 1) if count > 1 else 0
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            # Add perpendicular noise
            if noise > 0:
                # Calculate perpendicular direction
                dx, dy = x2 - x1, y2 - y1
                length = math.sqrt(dx*dx + dy*dy)
                if length > 0:
                    perp_x, perp_y = -dy/length, dx/length
                    noise_amount = random.uniform(-noise, noise)
                    x += noise_amount * perp_x
                    y += noise_amount * perp_y
            
            points.append((x, y))
        
        return points


class TestDatasetGenerator:
    """High-level generator for comprehensive test datasets."""
    
    def __init__(self, seed: int = 42):
        self.array_gen = ArrayGenerator(seed)
        self.point_gen = PointGenerator(seed)
        self.seed = seed
    
    def generate_sorting_datasets(self, sizes: List[int]) -> dict:
        """
        Generate comprehensive datasets for sorting algorithm testing.
        
        Args:
            sizes: List of array sizes to generate
            
        Returns:
            Dictionary with dataset name as key and list of arrays as value
        """
        datasets = {}
        
        for pattern in DataPattern:
            datasets[pattern.value] = []
            for size in sizes:
                try:
                    arr = self.array_gen.generate(size, pattern)
                    datasets[pattern.value].append(arr)
                except Exception as e:
                    print(f"Failed to generate {pattern.value} of size {size}: {e}")
                    continue
        
        return datasets
    
    def generate_selection_datasets(self, sizes: List[int]) -> dict:
        """Generate datasets specifically for selection algorithm testing."""
        datasets = {
            'worst_case_quickselect': [],  # Sorted arrays (worst for randomized quickselect)
            'average_case': [],            # Random arrays
            'duplicate_heavy': [],         # Many duplicates
            'small_range': []              # Small value range
        }
        
        for size in sizes:
            # Worst case for quickselect (sorted)
            datasets['worst_case_quickselect'].append(self.array_gen.generate(size, DataPattern.SORTED))
            
            # Average case (random)
            datasets['average_case'].append(self.array_gen.generate(size, DataPattern.RANDOM))
            
            # Duplicate heavy
            datasets['duplicate_heavy'].append(
                self.array_gen.generate(size, DataPattern.DUPLICATE_HEAVY, num_unique=max(1, size//20))
            )
            
            # Small range (values 0-9)
            datasets['small_range'].append(
                self.array_gen.generate(size, DataPattern.RANDOM, min_val=0, max_val=9)
            )
        
        return datasets
    
    def generate_closest_pair_datasets(self, sizes: List[int]) -> dict:
        """Generate datasets for closest pair algorithm testing."""
        datasets = {
            'random_uniform': [],
            'clustered': [],
            'grid_based': [],
            'circular': [],
            'linear': []
        }
        
        for size in sizes:
            # Random uniform distribution
            datasets['random_uniform'].append(self.point_gen.generate_random_points(size))
            
            # Clustered points
            num_clusters = max(2, min(10, size // 10))
            datasets['clustered'].append(
                self.point_gen.generate_clustered_points(size, num_clusters=num_clusters)
            )
            
            # Grid-based (if size allows for reasonable grid)
            grid_size = int(math.sqrt(size))
            if grid_size * grid_size >= size:
                grid_points = self.point_gen.generate_grid_points(grid_size, noise=1.0)
                datasets['grid_based'].append(grid_points[:size])
            else:
                datasets['grid_based'].append(self.point_gen.generate_random_points(size))
            
            # Circular arrangement
            datasets['circular'].append(self.point_gen.generate_circle_points(size, noise=2.0))
            
            # Linear arrangement
            datasets['linear'].append(self.point_gen.generate_line_points(size, noise=1.0))
        
        return datasets
    
    def create_benchmark_suite(self, sizes: List[int] = None) -> dict:
        """
        Create comprehensive benchmark suite for all algorithms.
        
        Args:
            sizes: List of input sizes to test (default: [10, 50, 100, 500, 1000])
            
        Returns:
            Dictionary with algorithm categories and their test datasets
        """
        if sizes is None:
            sizes = [10, 50, 100, 500, 1000]
        
        benchmark_suite = {
            'sorting': self.generate_sorting_datasets(sizes),
            'selection': self.generate_selection_datasets(sizes),
            'closest_pair': self.generate_closest_pair_datasets(sizes)
        }
        
        return benchmark_suite


# Utility functions
def save_datasets_to_file(datasets: dict, filename: str):
    """Save datasets to file for reuse."""
    import json
    
    # Convert to JSON-serializable format
    json_datasets = {}
    for category, data in datasets.items():
        if isinstance(data, dict):
            json_datasets[category] = {}
            for subcategory, arrays in data.items():
                json_datasets[category][subcategory] = arrays
        else:
            json_datasets[category] = data
    
    with open(filename, 'w') as f:
        json.dump(json_datasets, f, indent=2)
    
    print(f"Datasets saved to {filename}")

def load_datasets_from_file(filename: str) -> dict:
    """Load datasets from file."""
    import json
    
    with open(filename, 'r') as f:
        return json.load(f)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Data Generators:")
    
    # Test array generator
    gen = ArrayGenerator(seed=42)
    
    print("\nArray Generation Examples:")
    for pattern in [DataPattern.RANDOM, DataPattern.SORTED, DataPattern.DUPLICATE_HEAVY]:
        arr = gen.generate(10, pattern)
        print(f"{pattern.value:15}: {arr}")
    
    # Test point generator
    point_gen = PointGenerator(seed=42)
    
    print("\nPoint Generation Examples:")
    random_points = point_gen.generate_random_points(5)
    print(f"Random points: {random_points}")
    
    clustered_points = point_gen.generate_clustered_points(8, num_clusters=2)
    print(f"Clustered points: {clustered_points}")
    
    # Test comprehensive dataset generation
    print("\nGenerating Comprehensive Test Suite:")
    dataset_gen = TestDatasetGenerator(seed=42)
    
    # Generate small test suite
    suite = dataset_gen.create_benchmark_suite(sizes=[5, 10, 20])
    
    print(f"Generated {len(suite)} algorithm categories:")
    for category, data in suite.items():
        print(f"  {category}: {len(data)} pattern types")
        for pattern_name, arrays in data.items():
            sizes = [len(arr) for arr in arrays]
            print(f"    {pattern_name}: sizes {sizes}")
    
    print("\nData generators test completed! ✅")