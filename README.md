# Comparative Analysis of Sorting Algorithms: An Empirical Study

A comprehensive empirical evaluation of 19 sorting algorithms across 4 input distributions and 4 scale levels, reproducing and extending the work of Ala'anzy et al. (2024).

## Overview

This study presents transparent benchmarking of sorting algorithm performance with:
- **19 algorithms**: Bubble, Selection, Insertion, Merge, Quick, Heap, Counting, Radix, Bucket, Shell, Comb, Cocktail, Cycle, TimSort, Tree, Pancake, Flash, Tournament, and Strand Sort
- **4 input distributions**: Random, nearly-sorted, reverse-sorted, and few-unique values
- **4 scale levels**: n ∈ {100, 1,000, 10,000, 100,000}
- **304 total benchmarks**: 295 successful (97%), 9 recursion failures (3%)

## Key Findings

### Performance Differentials
- **47,400× speedup range** at n=100,000 (TimSort: 0.0086s vs Cycle Sort: 408s)
- O(n²) algorithms: 140–408s at large scale
- O(n log n) algorithms: 0.0086–0.035s at large scale

### Scalability Validation
- **Bubble Sort**: 1,327,000× growth from n=100 to n=100,000 (matches O(n²) theory)
- **TimSort**: 5,034× growth (matches O(n log n) theory)
- **Crossover threshold**: 58× range at n=100 (negligible) vs 2,215× range at n=100,000 (critical)

### Input Sensitivity
- **Insertion Sort**: 15.4× variation (16.3s nearly-sorted vs 251s reverse-sorted at n=100K)
- **TimSort**: 3.4× variation (demonstrates robust adaptivity)
- **Specialized algorithms**: Counting/Radix Sort achieve 6–16× speedups on few-unique data

### Implementation Constraints
- **9 recursion failures** (Quick Sort: 4, Tree Sort: 5) at n≥1,000 on adversarial inputs
- Python's ~3,000 stack frame limit causes failures despite O(n log n) average-case complexity

## Repository Structure

```
.
└── datasets/ 
├── algorithm_analysis.py          # Main benchmarking harness
├── visualize_results.py           # Heatmap generation script
├── algorithm_timings_random.csv   # Results: random distribution
├── algorithm_timings_nearly_sorted.csv
├── algorithm_timings_reverse_sorted.csv
├── algorithm_timings_few_unique.csv
└── plots/      # Column-normalized visualizations                  
```

## Methodology

### Experimental Protocol
- **Platform**: Apple M4, Python 3.12, macOS Sequoia
- **Timing**: `time.perf_counter_ns()` for nanosecond resolution
- **Measurement**: Median of 10 runs per configuration
- **Reproducibility**: Fixed seed (`random.seed(42)`)
- **Validation**: Pre-timing correctness checks on all implementations

### Visualization
Column-normalized heatmaps provide independent color scales per input size (100, 1K, 10K, 100K), exposing:
- **Light colors**: Fast relative performance at that scale
- **Dark colors**: Slow relative performance at that scale
- **White cells**: Recursion failures
- **Color progression**: Algorithms maintaining light colors across sizes scale well; light-to-dark progression indicates deteriorating competitiveness

## Running the Benchmarks

### Prerequisites
```bash
python3 -m pip install numpy matplotlib seaborn pandas
```

### Execute Benchmarking
```bash
python3 algorithm_analysis.py
```
Generates CSV files with median execution times for all 304 algorithm-dataset combinations.

### Generate Visualizations
```bash
python3 visualize_results.py
```
Creates column-normalized heatmaps from CSV results.

## Practical Implications

### Algorithm Selection Criteria

1. **Know your input distribution**
   - Quick Sort: Excellent on random data, fails on sorted inputs without pivot optimization
   - Insertion Sort: 15.4× faster on nearly-sorted vs reverse-sorted data

2. **Prioritize robustness**
   - TimSort: 3.4× variation across all distributions
   - Merge Sort/Heap Sort: Guaranteed O(n log n) regardless of input pattern

3. **Respect scale thresholds**
   - n < 1,000: Algorithm choice has <0.006s impact (negligible)
   - n ≥ 10,000: Poor choices cost 100–400× in runtime

4. **Leverage data properties**
   - Counting/Radix Sort: 6–16× speedups on few-unique data
   - Trade-off: 24,100× overhead at small scales due to auxiliary space


## References

Original study reproduced:
> Ala'anzy, M., Almalki, J., & Alqarni, M. (2024). Comparative Analysis of Sorting Algorithms: A Review. *International Journal of Advanced Computer Science and Applications*, 15(11).

## License

MIT License - See LICENSE file for details

## Contact

For questions or collaboration: evilkomir@gmail.com

---

**Note**: All measurements are single-machine results. Absolute times vary across platforms, but qualitative rankings remain stable. This study emphasizes reproducibility through transparent protocols, fixed seeds, and complete artifact sharing.
