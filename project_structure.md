# Divide and Conquer Algorithms - Project Structure

## Complete File Structure

```
divide-and-conquer-algorithms/
├── README.md                           # Main project documentation
├── requirements.txt                    # Python dependencies
├── PROJECT_STRUCTURE.md               # This file
├── .gitignore                         # Git ignore rules
├── 
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── benchmark.py                   # Main benchmarking script
│   ├── plot_results.py               # Results visualization
│   │
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── mergesort.py              # MergeSort implementation
│   │   ├── quicksort.py              # QuickSort implementations
│   │   ├── select.py                 # Selection algorithms
│   │   └── closest_pair.py           # Closest pair algorithm
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py                # Performance metrics collection
│       └── generators.py             # Test data generators
│
├── tests/
│   ├── __init__.py
│   └── test_algorithms.py            # Comprehensive test suite
│
├── results/                          # Generated benchmark results
│   ├── benchmark_results.json        # Raw benchmark data
│   ├── benchmark_report.md          # Generated report
│   ├── benchmark_data.csv           # CSV data for plotting
│   └── plots/                       # Generated visualizations
│       ├── time_complexity.png
│       ├── recursion_depth.png
│       ├── comparisons_analysis.png
│       ├── pattern_comparison.png
│       └── performance_dashboard.png
│
├── docs/                            # Documentation
│   ├── algorithm_analysis.md        # Detailed algorithm analysis
│   ├── usage_guide.md              # Usage instructions
│   └── theoretical_background.md    # Theoretical foundations
│
└── examples/                       # Example usage scripts
    ├── basic_usage.py              # Basic algorithm usage examples
    ├── custom_benchmarks.py        # Custom benchmark examples
    └── interactive_demo.py         # Interactive demonstration
```

## Key Components

### Core Algorithms (`src/algorithms/`)

1. **MergeSort (`mergesort.py`)**
   - Stable O(n log n) sorting algorithm
   - Optimizations: reusable buffer, insertion sort cutoff
   - Master Theorem Case 2 example

2. **QuickSort (`quicksort.py`)**
   - Randomized pivot selection
   - Tail recursion optimization for bounded stack depth
   - Three-way partitioning variant for duplicate-heavy data

3. **Deterministic Select (`select.py`)**
   - Median-of-medians algorithm
   - Guaranteed O(n) worst-case performance
   - Comparison with randomized QuickSelect

4. **Closest Pair (`closest_pair.py`)**
   - 2D divide-and-conquer approach
   - Strip optimization with 7-point neighbor check
   - O(n log n) complexity with detailed geometric analysis

### Support Systems

#### Metrics Collection (`src/utils/metrics.py`)
- Thread-safe performance measurement
- Tracks: execution time, recursion depth, comparisons, memory allocations
- Benchmarking framework with statistical analysis

#### Data Generation (`src/utils/generators.py`)
- Comprehensive test data patterns
- Array generators: random, sorted, nearly-sorted, duplicate-heavy
- Point generators: random, clustered, grid-based, circular

#### Visualization (`src/plot_results.py`)
- Time complexity analysis plots
- Recursion depth visualization
- Performance comparison dashboards
- Pattern-based performance analysis

## Git Workflow Structure

### Branches
```
main                    # Stable releases only (v0.1, v1.0)
├── feature/mergesort   # MergeSort implementation
├── feature/quicksort   # QuickSort implementation
├── feature/select      # Selection algorithms
├── feature/closest     # Closest pair algorithm
├── feature/metrics     # Metrics and benchmarking
└── feature/docs        # Documentation updates
```

### Commit Message Format
```
<type>(<scope>): <description>

Types: feat, fix, docs, test, refactor, perf
Scopes: mergesort, quicksort, select, closest, metrics, tests
```

### Release Tags
- `v0.1`: Basic algorithm implementations
- `v1.0`: Complete implementation with metrics and analysis

## Testing Architecture

### Test Categories
1. **Correctness Tests**: Verify algorithm outputs
2. **Edge Case Tests**: Empty inputs, single elements, duplicates
3. **Performance Tests**: Basic complexity verification
4. **Integration Tests**: Cross-algorithm comparisons

### Test Data Patterns
- Random arrays of various sizes
- Pathological cases (sorted, reverse-sorted)
- Duplicate-heavy datasets
- Geometric point configurations

## Usage Examples

### Basic Algorithm Usage
```python
from src.algorithms.mergesort import merge_sort
from src.algorithms.select import select
from src.algorithms.closest_pair import closest_pair

# Sort array
sorted_arr = merge_sort([64, 34, 25, 12, 22, 11, 90])

# Find median
median = select([9, 3, 6, 2, 8, 1, 5, 7, 4], 4, deterministic=True)

# Find closest pair of points
points = [(0, 0), (1, 1), (2, 2), (10, 10)]
result = closest_pair(points)
```

### Running Benchmarks
```bash
# Full benchmark suite
python src/benchmark.py

# Specific algorithm categories
python src/benchmark.py --algorithm sorting
python src/benchmark.py --algorithm selection

# Custom sizes
python src/benchmark.py --sorting-sizes 100 500 1000 5000
```

### Generating Plots
```bash
# All plots
python src/plot_results.py

# Specific plot types
python src/plot_results.py --plot-type time
python src/plot_results.py --plot-type dashboard

# Sample plots (no benchmark data needed)
python src/plot_results.py --sample
```

## Development Setup

### Installation
```bash
# Clone repository
git clone <repository-url>
cd divide-and-conquer-algorithms

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Run benchmarks
python src/benchmark.py

# Generate plots
python src/plot_results.py --sample
```

### Dependencies
- **Core**: Python 3.8+
- **Analysis**: numpy, pandas, scipy
- **Visualization**: matplotlib, seaborn, plotly
- **Testing**: pytest, pytest-cov
- **Development**: black, flake8

## Performance Analysis Framework

### Complexity Analysis Methods
1. **Master Theorem**: Applied to divide-and-conquer recurrences
2. **Akra-Bazzi Method**: For more complex recurrence relations
3. **Empirical Analysis**: Growth rate measurement and validation

### Metrics Collected
- **Time**: High-precision execution timing
- **Space**: Memory allocation tracking
- **Operations**: Comparison and swap counts
- **Recursion**: Maximum stack depth measurement

### Visualization Types
- Log-log plots for complexity verification
- Linear plots for detailed analysis
- Heatmaps for pattern comparison
- Dashboard views for comprehensive overview

## Algorithm Optimizations Implemented

### MergeSort
- Reusable buffer to minimize allocations
- Insertion sort cutoff for small subarrays
- Bottom-up merge for iterative version

### QuickSort
- Randomized pivot selection
- Tail recursion elimination
- Three-way partitioning for duplicates
- Iterative processing of larger partition

### Deterministic Select
- Groups of 5 for median calculation
- Recursion only into needed partition
- Preference for smaller recursive calls

### Closest Pair
- Efficient strip construction
- Optimized neighbor checking (7-point rule)
- Preprocessing optimizations for sorted arrays

This structure provides a comprehensive framework for studying, implementing, and analyzing divide-and-conquer algorithms with rigorous performance measurement and visualization capabilities.