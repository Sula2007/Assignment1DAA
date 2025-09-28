# Divide and Conquer Algorithms Implementation

## Learning Objectives
- Implement classic divide-and-conquer algorithms with safe recursion patterns
- Time complexity analysis using the Main Theorem (3 cases) and Akra-Bazzi intuition
- Collect metrics (time, recursion depth, comparisons/memory allocations) and present results

## Project Architecture

### File Structure
```
src/
├── algorithms/
│ ├── mergesort.py # MergeSort with optimizations
│ ├── quicksort.py # Randomized QuickSort
│ ├── select.py # Deterministic Select (median of medians)
│ └── closest_pair.py # Closest pair points
├── utils/
│ ├── metrics.py # Collect performance metrics
│ └── generators.py # Test data generators
└── tests/
└── test_algorithms.py # Algorithm tests
```

### Recursion depth and memory allocation control
- **MergeSort**: Uses a reusable buffer and switches to insertion sort for small arrays (n < 10)
- **QuickSort**: Recurse only on the smaller part, iterate on the larger (depth ≈ O(log n))
- **Select**: Recurse only in the desired direction, preferring the smaller part
- **Closest Pair**: Standard dichotomy with band checking

## Recurrence analysis Relations

### 1. MergeSort - Master Theorem Case 2
**Recurrence**: T(n) = 2T(n/2) + O(n)
**Analysis**: a=2, b=2, f(n)=n, so n^(log_b(a)) = n^1 = n. Case 2: f(n) = Θ(n^(log_b(a))).
**Result**: T(n) = Θ(n log n)

### 2. QuickSort (Average Case)
**Recurrence**: T(n) = T(k) + T(n-k-1) + O(n), where k is a random position
**Analysis**: On average, k ≈ n/2, which gives T(n) = 2T(n/2) + O(n).
**Result**: T(n) = Θ(n log n) on average, O(n²) worst-case

### 3. Deterministic Select (Median-of-Medians)
**Recurrence**: T(n) = T(⌈n/5⌉) + T(7n/10) + O(n)
**Analysis**: Akra-Bazzi with p=1, since 1/5 + 7/10 < 1.
**Result**: T(n) = Θ(n)

### 4. Closest Pair of Points
**Recurrence**: T(n) = 2T(n/2) + O(n log n) for strip sort
**Analysis**: Master Theorem Case 3: f(n) = n log n > n^1.
**Result**: T(n) = Θ(n log n)

## Measurement Results

### Runtime vs. Input Size
![Time Complexity](plots/time_complexity.png)

The plots show compliance with the theoretical complexity:
- MergeSort: stable O(n log n) in all cases
- QuickSort: O(n log n) on average, with rare outliers
- Select: linear O(n)
- Closest Pair: O(n log n) with a high constant

### Recursion Depth vs. Input Size
![Recursion Depth](plots/recursion_depth.png)

Optimizations successfully limit the stack depth:
- QuickSort: ~log n thanks to tail recursion optimization
- MergeSort: exactly log n
- Select: ~log n thanks to one-way recursion

### Effect of constant factors
- **Cache Effects**: MergeSort performs better on large arrays due to locality of access
- **Garbage Collection**: Reusing buffers significantly reduces GC overhead
- **Cutoff Size**: Optimal value of 10 for switching to insertion sort

## Conclusions

### Correspondence between Theory and Practice
- **High Correspondence**: MergeSort and Select exhibit predictable behavior
- **Partial Correspondence**: QuickSort demonstrates theoretical complexity on average
- **Constant Factors**: Significantly impact practical performance

### Recommendations
1. MergeSort for stable performance
2. QuickSort for general use with good constants
3. Select for guaranteed linear search performance
4. Optimizing cutoffs and buffers is critical for practical use

## Usage

```bash
# Installing Dependencies
pip install -r requirements.txt

# Running tests
python -m pytest tests/

# Generating performance reports
python src/benchmark.py

# Creating plots
python src/plot_results.py
```

## Git Workflow

### Branches
- `main`: only production releases (tags v0.1, v1.0)
- `feature/mergesort`: MergeSort implementation
- `feature/quicksort`: QuickSort implementation
- `feature/select`: Deterministic Select implementation
- `feature/closest`: Closest Pair implementation
- `feature/metrics`: metrics collection system

### Tags
- v0.1: basic algorithm implementation
- v1.0: full implementation with metrics and reports
