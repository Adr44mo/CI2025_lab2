# TSP Solver Comparison - Lab 2

## Overview

This project implements and compares **three different approaches** to solve the Traveling Salesman Problem (TSP), with special focus on handling **asymmetric** and **negative-weight** distance matrices.

## Problem Types

- **G problems**: Symmetric distance matrices (D[i,j] = D[j,i])
- **R1 problems**: Asymmetric distance matrices (D[i,j] ≠ D[j,i])
- **R2 problems**: Asymmetric with negative edge weights

**Sizes tested**: 10, 20, 50, 100, 200, 500, 1000 cities

---

## Implemented Algorithms

### 1. K-means Clustering + 2-opt (Symmetric)
**File**: `kmeans_tsp_solver.py`

**Approach**: MDS embedding → K-means clustering → parallel 2-opt → greedy merge → boundary refinement

**Key optimization**: O(1) delta calculation for symmetric 2-opt
```python
delta = D[a,c] + D[b,d] - D[a,b] - D[c,d]  # 4 matrix accesses
```

**Best for**: Symmetric problems (G)

---

### 2. Hierarchical Clustering + Asymmetric 2-opt
**File**: `asymmetric_tsp_solver.py`

**Approach**: Hierarchical clustering → parallel solve (Held-Karp for n≤15) → asymmetric merge → asymmetric 2-opt

**Critical Optimizations**:

**A) D_sym Filtering** (eliminates 90% of bad swaps):
```python
D_sym = (D + D.T) / 2  # Precalculate symmetric approximation
delta_approx = D_sym[a,c] + D_sym[b,d] - D_sym[a,b] - D_sym[c,d]
if delta_approx >= 0: continue  # Skip expensive calculation
```

**B) Incremental Delta Calculation**:
```python
# Instead of recalculating entire segment sums (O(n))
for k in range(len(segment) - 1):
    delta += D[segment[-(k+1)], segment[-(k+2)]] - D[segment[k], segment[k+1]]
```

**Performance Impact**: 6-10x speedup over naive asymmetric 2-opt

**Best for**: Asymmetric problems (R1, R2)

---

### 3. Evolutionary Algorithm (Genetic Algorithm)
**File**: `evolutionary_tsp_solver.py`

**Approach**: Population init → tournament selection → order crossover (OX) → mutation → 2-opt local search → elitism

**Order Crossover (OX)**: Preserves relative city order (better for TSP than PMX)

**Adaptive Parameters** (critical for scalability):
```python
if n <= 50:  population=100, generations=500
elif n <= 500: population=50,  generations=200
else:          population=40,  generations=150
```

**Asymmetric 2-opt Integration**: Uses same D_sym filtering as asymmetric solver

**Best for**: Small problems (n≤200) and exploration-focused solving

---

## Performance Summary

See **[RESULTS.md](RESULTS.md)** for detailed benchmark results.

**Quick comparison** (n=1000):
- K-means: 60s, cost=12608
- Asymmetric: **15s**, cost=12550 ⭐ **fastest**
- Evolutionary: 135s, cost=12464 ⭐ **best quality**

**Key findings**:
- **Small problems (n≤200)**: Evolutionary fastest (no clustering overhead)
- **Large problems (n≥500)**: Asymmetric solver fastest (2-4x speedup)
- **R2 problems**: Asymmetric best overall (speed + quality)

---

## Project Structure

```
lab2/
├── README.md                      # This file
├── RESULTS.md                     # Detailed results and analysis
├── tsp_pb_comparaison.ipynb      # Main comparison notebook
├── kmeans_tsp_solver.py          # Symmetric solver
├── asymmetric_tsp_solver.py      # Asymmetric solver
├── evolutionary_tsp_solver.py    # Genetic algorithm
├── held_karp.py                  # Exact solver (small instances)
└── lab2/                         # Test problems
    ├── problem_g_*.npy
    ├── problem_r1_*.npy
    └── problem_r2_*.npy
```

---

## Key Algorithmic Insights

### Why Asymmetric TSP is Harder

In **symmetric TSP**, reversing a segment doesn't change its internal cost:
```
a → b → c → d    cost = D[b,c] + D[c,d]
a → d → c → b    cost = D[d,c] + D[c,b] = D[c,d] + D[b,c]  (same!)
```

In **asymmetric TSP**, reversal changes everything:
```
a → b → c → d    cost = D[b,c] + D[c,d]
a → d → c → b    cost = D[d,c] + D[c,b]  (completely different!)
```

**Impact**: For n=200, average segment ~100 cities → **100 matrix accesses vs 4 per swap**

### The D_sym Filtering Breakthrough

**Problem**: Asymmetric 2-opt was 20-30x slower than symmetric

**Insight**: Most swaps are bad even in symmetric approximation

**Solution**: 
1. Precalculate `D_sym = (D + D.T) / 2` once
2. Filter with O(1) symmetric delta
3. Only calculate expensive asymmetric delta for ~10% of swaps

**Result**: Achieved **6-10x speedup**, making asymmetric solver competitive with symmetric

---

## Dependencies

```bash
pip install numpy scipy scikit-learn matplotlib jupyter
```

**Python version**: 3.8+

---

## References

**Algorithms**:
- Held-Karp: Exact TSP solver (dynamic programming)
- 2-opt: Lin, S. (1965). "Computer solutions of the TSP"
- Order Crossover: Davis, L. (1985). "Applying Adaptive Algorithms to Epistatic Domains"

**Optimization techniques**:
- Candidate lists: Bentley, J. (1992). "Fast algorithms for geometric TSP"
- D_sym filtering: Custom innovation for asymmetric TSP

**Development tools**:
- GitHub Copilot: Code comments, documentation (README.md, RESULTS.md), and code clarity improvements

---

