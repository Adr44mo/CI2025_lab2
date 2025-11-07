"""
Complete K-means TSP solver with Davies-Bouldin clustering and efficient merging.

Pipeline:
1. Davies-Bouldin → optimal k
2. K-means clustering (MDS embedding)
3. Parallel TSP solving per cluster
4. Efficient cluster merging (nearest-neighbor stitching)
5. Cluster-aware 2-opt refinement (boundary optimization first)
"""
import numpy as np
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


# ============================================================================
# Local TSP utilities (no external dependencies)
# ============================================================================

def tour_cost(D: np.ndarray, perm: List[int]) -> float:
    """Calculate tour cost."""
    if not perm:
        return 0.0
    n = len(perm)
    cost = 0.0
    for i in range(n):
        a = perm[i]
        b = perm[(i + 1) % n]
        cost += D[a, b]
    return float(cost)


def nearest_neighbor_start(D: np.ndarray, start: int = 0) -> List[int]:
    """Greedy nearest neighbor TSP initialization."""
    n = D.shape[0]
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    
    while unvisited:
        current = tour[-1]
        nearest = min(unvisited, key=lambda x: D[current, x])
        tour.append(nearest)
        unvisited.remove(nearest)
    
    return tour


def build_candidate_lists(D: np.ndarray, k: int = 20) -> List[np.ndarray]:
    """Build k-nearest neighbor candidate lists."""
    n = D.shape[0]
    candidates = []
    
    for i in range(n):
        # Sort by distance (exclude self)
        distances = [(D[i, j], j) for j in range(n) if j != i]
        distances.sort()
        # Take k nearest
        nearest = [j for _, j in distances[:k]]
        candidates.append(np.array(nearest, dtype=np.int32))
    
    return candidates


def two_opt_with_candidates(D: np.ndarray, perm: List[int], 
                            candidates: List[np.ndarray] = None,
                            max_no_improve: int = 200) -> Tuple[List[int], float]:
    """2-opt local search with candidate lists."""
    n = len(perm)
    
    if candidates is None:
        k = min(20, max(5, n // 20))
        candidates = build_candidate_lists(D, k=k)
    
    # Position map
    pos = np.empty(n, dtype=int)
    for i, v in enumerate(perm):
        pos[v] = i
    
    best = perm[:]
    best_cost = tour_cost(D, best)
    no_improve = 0
    
    while no_improve < max_no_improve:
        improved = False
        
        for i in range(n):
            a = best[i]
            b = best[(i + 1) % n]
            
            for c in candidates[a]:
                j = pos[c]
                
                if j == i or j == (i + 1) % n:
                    continue
                
                d = best[(j + 1) % n]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                
                if delta < -1e-9:
                    # Perform reversal
                    if i < j:
                        best[i + 1 : j + 1] = best[i + 1 : j + 1][::-1]
                    else:
                        segment = best[i + 1 :] + best[: j + 1]
                        segment_reversed = segment[::-1]
                        len_tail = n - (i + 1)
                        best[i + 1 :] = segment_reversed[:len_tail]
                        best[: j + 1] = segment_reversed[len_tail:]
                    
                    # Update pos
                    for idx in range(n):
                        pos[best[idx]] = idx
                    
                    best_cost += delta
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
    
    return best, best_cost


def perturb_perm(perm: List[int], k: int = 5) -> List[int]:
    """Perturb permutation by reversing k random segments."""
    import random
    result = perm[:]
    n = len(result)
    
    for _ in range(k):
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        result[i:j+1] = result[i:j+1][::-1]
    
    return result


def multi_start_two_opt(D: np.ndarray, restarts: int = 10, 
                        initial_perm: List[int] = None,
                        verbose: bool = False) -> Tuple[List[int], float]:
    """Multi-start 2-opt."""
    n = D.shape[0]
    k_candidates = min(20, max(5, n // 20))
    candidates = build_candidate_lists(D, k=k_candidates)
    
    if initial_perm is None:
        best_perm = nearest_neighbor_start(D)
        best_perm, best_cost = two_opt_with_candidates(D, best_perm, candidates)
    else:
        best_perm = initial_perm[:]
        best_cost = tour_cost(D, best_perm)
    
    for r in range(restarts):
        if r == 0 and initial_perm is not None:
            perm0 = initial_perm[:]
        elif r == 0:
            perm0 = nearest_neighbor_start(D)
        else:
            perm0 = perturb_perm(best_perm, k=max(3, n // 50))
        
        perm_improved, cost = two_opt_with_candidates(D, perm0, candidates)
        
        if cost < best_cost:
            best_cost = cost
            best_perm = perm_improved[:]
    
    return best_perm, best_cost


# ============================================================================
# Clustering
# ============================================================================

def davies_bouldin_optimal_k(D: np.ndarray, max_clusters: int = 20, verbose: bool = True) -> int:
    """Find optimal number of clusters using Davies-Bouldin Index."""
    if verbose:
        print(f"[Davies-Bouldin] Finding optimal k (testing 2-{max_clusters})...")
    
    # Symmetrize distance matrix for MDS (required for asymmetric problems)
    D_sym = (D + D.T) / 2
    
    # MDS embedding
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(D_sym)
    
    db_scores = []
    K_range = range(2, min(max_clusters + 1, len(D) // 2))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        score = davies_bouldin_score(coords, labels)
        db_scores.append(score)
        if verbose:
            print(f"  k={k:2d}: DB index={score:.3f}")
    
    optimal_k = list(K_range)[np.argmin(db_scores)]
    best_score = np.min(db_scores)
    
    if verbose:
        print(f"✓ Optimal k: {optimal_k} (DB index={best_score:.3f})\n")
    
    return optimal_k


def cluster_tsp_problem(D: np.ndarray, n_clusters: int, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster TSP problem using K-means on MDS embedding."""
    if verbose:
        print(f"[K-means] Clustering into {n_clusters} clusters...")
    
    # Symmetrize distance matrix for MDS (required for asymmetric problems)
    D_sym = (D + D.T) / 2
    
    # MDS embedding
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(D_sym)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    
    if verbose:
        for c in range(n_clusters):
            size = np.sum(labels == c)
            print(f"  Cluster {c}: {size} nodes")
        print()
    
    return labels, coords


# ============================================================================
# Parallel cluster solving
# ============================================================================

def solve_cluster_tsp(args: Tuple[int, np.ndarray, np.ndarray, int]) -> Tuple[int, List[int], float]:
    """Solve TSP for a single cluster (for parallel execution)."""
    cluster_id, cluster_nodes, D, restarts = args
    
    if len(cluster_nodes) <= 1:
        return cluster_id, list(cluster_nodes), 0.0
    
    # Extract submatrix
    sub_D = D[np.ix_(cluster_nodes, cluster_nodes)]
    
    # Solve TSP (local function)
    perm_local, cost_local = multi_start_two_opt(sub_D, restarts=restarts, verbose=False)
    
    # Convert to global indices
    global_tour = [int(cluster_nodes[i]) for i in perm_local]
    
    return cluster_id, global_tour, cost_local


def solve_clusters_parallel(D: np.ndarray, labels: np.ndarray, n_clusters: int, 
                            restarts_per_cluster: int = 20, verbose: bool = True) -> Dict[int, Tuple[List[int], float]]:
    """Solve TSP for each cluster in parallel."""
    if verbose:
        print(f"[Parallel TSP] Solving {n_clusters} sub-problems...")
    
    tasks = []
    for c in range(n_clusters):
        cluster_nodes = np.where(labels == c)[0]
        tasks.append((c, cluster_nodes, D, restarts_per_cluster))
    
    cluster_tours = {}
    start_time = time.time()
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(solve_cluster_tsp, task): task[0] for task in tasks}
        
        for future in as_completed(futures):
            cluster_id, tour, cost = future.result()
            cluster_tours[cluster_id] = (tour, cost)
            if verbose:
                print(f"  Cluster {cluster_id}: {len(tour)} nodes, cost={cost:.2f}")
    
    elapsed = time.time() - start_time
    if verbose:
        total_cost = sum(c for _, c in cluster_tours.values())
        print(f"✓ Parallel solving done in {elapsed:.2f}s, total intra-cluster cost: {total_cost:.2f}\n")
    
    return cluster_tours


# ============================================================================
# Merging
# ============================================================================

def merge_cluster_tours_greedy(D: np.ndarray, cluster_tours: Dict[int, Tuple[List[int], float]], 
                               verbose: bool = True) -> Tuple[List[int], float]:
    """Merge cluster tours using greedy nearest-neighbor stitching."""
    if verbose:
        print("[Merge] Stitching cluster tours with greedy nearest-neighbor...")
    
    sorted_clusters = sorted(cluster_tours.items(), key=lambda x: len(x[1][0]), reverse=True)
    merged_tour = list(sorted_clusters[0][1][0])
    remaining = {cid: tour for cid, (tour, _) in sorted_clusters[1:]}
    
    while remaining:
        best_cost_increase = float('inf')
        best_insert_pos = None
        best_cluster_id = None
        best_cluster_orientation = None
        
        for cluster_id, cluster_tour in remaining.items():
            for orient in [cluster_tour, cluster_tour[::-1]]:
                for i in range(len(merged_tour)):
                    a = merged_tour[i]
                    b = merged_tour[(i + 1) % len(merged_tour)]
                    
                    start_node = orient[0]
                    end_node = orient[-1]
                    
                    cost_increase = D[a, start_node] + D[end_node, b] - D[a, b]
                    
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_insert_pos = i + 1
                        best_cluster_id = cluster_id
                        best_cluster_orientation = orient
        
        merged_tour = (merged_tour[:best_insert_pos] + 
                      list(best_cluster_orientation) + 
                      merged_tour[best_insert_pos:])
        
        del remaining[best_cluster_id]
        
        if verbose:
            print(f"  Attached cluster {best_cluster_id} (cost increase: {best_cost_increase:.2f})")
    
    merge_cost = tour_cost(D, merged_tour)
    
    if verbose:
        print(f"✓ Merge complete: {len(merged_tour)} nodes, cost={merge_cost:.2f}\n")
    
    return merged_tour, merge_cost


# ============================================================================
# Cluster-aware refinement
# ============================================================================

def refine_cluster_boundaries(D: np.ndarray, tour: List[int], labels: np.ndarray, 
                              max_iterations: int = 1000, verbose: bool = True) -> Tuple[List[int], float]:
    """
    Refine tour by focusing on inter-cluster edges (boundary optimization).
    
    Strategy:
    1. Identify edges that cross cluster boundaries
    2. Try 2-opt swaps specifically on boundary edges first
    3. These are the weakest connections from merging
    """
    if verbose:
        print(f"[Boundary Refinement] Optimizing inter-cluster connections...")
    
    start_time = time.time()
    n = len(tour)
    
    # Identify boundary edges
    boundary_edges = []
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        if labels[a] != labels[b]:
            boundary_edges.append(i)
    
    if verbose:
        print(f"  Found {len(boundary_edges)} boundary edges out of {n} total")
    
    # Position map
    pos = np.empty(n, dtype=int)
    for i, v in enumerate(tour):
        pos[v] = i
    
    best = tour[:]
    best_cost = tour_cost(D, best)
    initial_cost = best_cost
    
    improvements = 0
    iterations = 0
    no_improve = 0
    
    while no_improve < 100 and iterations < max_iterations:
        improved = False
        
        # Focus on boundary edges
        for i in boundary_edges:
            a = best[i]
            b = best[(i + 1) % n]
            
            # Try swapping with other boundary edges
            for j in boundary_edges:
                if j == i or j == (i + 1) % n:
                    continue
                
                c = best[j]
                d = best[(j + 1) % n]
                
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                
                if delta < -1e-9:
                    # Perform reversal
                    if i < j:
                        best[i + 1 : j + 1] = best[i + 1 : j + 1][::-1]
                    else:
                        segment = best[i + 1 :] + best[: j + 1]
                        segment_reversed = segment[::-1]
                        len_tail = n - (i + 1)
                        best[i + 1 :] = segment_reversed[:len_tail]
                        best[: j + 1] = segment_reversed[len_tail:]
                    
                    # Update pos
                    for idx in range(n):
                        pos[best[idx]] = idx
                    
                    best_cost += delta
                    improvements += 1
                    improved = True
                    break
            
            if improved:
                break
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
        
        iterations += 1
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"  Boundary optimization: {improvements} improvements in {elapsed:.2f}s")
        print(f"  Cost: {initial_cost:.2f} → {best_cost:.2f} ({(initial_cost - best_cost) / initial_cost * 100:.2f}%)\n")
    
    return best, best_cost


def refine_with_two_opt(D: np.ndarray, tour: List[int], labels: np.ndarray,
                        boundary_iterations: int = 1000,
                        general_restarts: int = 5,
                        verbose: bool = True) -> Tuple[List[int], float]:
    """
    Two-phase refinement:
    1. Focus on cluster boundary edges (most important)
    2. General multi-restart 2-opt
    """
    if verbose:
        print(f"[2-Opt Refinement] Phase 1: Boundary optimization, Phase 2: General refinement\n")
    
    start_time = time.time()
    initial_cost = tour_cost(D, tour)
    
    # Phase 1: Optimize cluster boundaries
    tour_after_boundary, cost_after_boundary = refine_cluster_boundaries(
        D, tour, labels, max_iterations=boundary_iterations, verbose=verbose
    )
    
    # Phase 2: Multi-restart general 2-opt
    if general_restarts > 0:
        if verbose:
            print(f"[Phase 2] Running {general_restarts} general 2-opt restarts...")
        
        best_tour, best_cost = multi_start_two_opt(
            D, 
            restarts=general_restarts,
            initial_perm=tour_after_boundary,
            verbose=False
        )
    else:
        best_tour = tour_after_boundary
        best_cost = cost_after_boundary
    
    elapsed = time.time() - start_time
    
    if verbose:
        total_improvement = (initial_cost - best_cost) / initial_cost * 100
        print(f"✓ Complete refinement done in {elapsed:.2f}s")
        print(f"  Initial: {initial_cost:.2f}")
        print(f"  After boundary: {cost_after_boundary:.2f}")
        print(f"  Final: {best_cost:.2f}")
        print(f"  Total improvement: {total_improvement:.2f}%\n")
    
    return best_tour, best_cost


# ============================================================================
# Main solver
# ============================================================================

def kmeans_tsp_solve(
    D: np.ndarray,
    max_clusters: int = 20,
    restarts_per_cluster: int = 20,
    boundary_iterations: int = 1000,
    general_restarts: int = 5,
    auto_clusters: bool = True,
    n_clusters: int = None,
    verbose: bool = True
) -> Tuple[List[int], Dict[str, float]]:
    """
    Complete K-means TSP solver with cluster-aware refinement.
    
    Args:
        D: distance matrix (n x n)
        max_clusters: max k for Davies-Bouldin search
        restarts_per_cluster: 2-opt restarts per cluster
        boundary_iterations: iterations for inter-cluster edge optimization
        general_restarts: multi-restart 2-opt after boundary refinement
        auto_clusters: use Davies-Bouldin to choose k
        n_clusters: manual cluster count (if auto_clusters=False)
        verbose: print progress
    
    Returns:
        (final_tour, cost_breakdown)
    """
    n = D.shape[0]
    
    if verbose:
        print(f"{'='*60}")
        print(f"K-means TSP Solver (n={n})")
        print(f"{'='*60}\n")
    
    start_total = time.time()
    
    # Step 1: Find optimal k
    if auto_clusters:
        k = davies_bouldin_optimal_k(D, max_clusters=max_clusters, verbose=verbose)
    else:
        k = n_clusters if n_clusters else max(2, int(np.sqrt(n / 2)))
        if verbose:
            print(f"[Manual] Using k={k} clusters\n")
    
    # Step 2: Cluster
    labels, coords = cluster_tsp_problem(D, k, verbose=verbose)
    
    # Step 3: Solve each cluster
    cluster_tours = solve_clusters_parallel(D, labels, k, restarts_per_cluster, verbose=verbose)
    intra_cluster_cost = sum(cost for _, cost in cluster_tours.values())
    
    # Step 4: Merge clusters
    merged_tour, merge_cost = merge_cluster_tours_greedy(D, cluster_tours, verbose=verbose)
    
    # Step 5: Cluster-aware refinement
    final_tour, final_cost = refine_with_two_opt(
        D, merged_tour, labels,
        boundary_iterations=boundary_iterations,
        general_restarts=general_restarts,
        verbose=verbose
    )
    
    total_time = time.time() - start_total
    
    # Cost breakdown
    cost_breakdown = {
        'intra_cluster': intra_cluster_cost,
        'after_merge': merge_cost,
        'final': final_cost,
        'improvement': (merge_cost - final_cost) / merge_cost * 100,
        'time': total_time,
        'n_clusters': k
    }
    
    if verbose:
        print(f"{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Number of clusters: {k}")
        print(f"Intra-cluster cost: {intra_cluster_cost:.2f}")
        print(f"After merging: {merge_cost:.2f}")
        print(f"Final (after refinement): {final_cost:.2f}")
        print(f"Improvement from merge: {cost_breakdown['improvement']:.2f}%")
        print(f"Total time: {total_time:.2f}s")
        print(f"{'='*60}\n")
    
    return final_tour, cost_breakdown


if __name__ == "__main__":
    test_files = [
        "lab2/problem_g_100.npy",
        "lab2/problem_g_500.npy",
        "lab2/problem_g_1000.npy",
    ]
    
    for filepath in test_files:
        try:
            print(f"\n{'#'*60}")
            print(f"Testing: {filepath}")
            print(f"{'#'*60}\n")
            
            D = np.load(filepath)
            
            tour, stats = kmeans_tsp_solve(
                D,
                max_clusters=15,
                restarts_per_cluster=20,
                boundary_iterations=1000,
                general_restarts=5,
                auto_clusters=True,
                verbose=True
            )
            
            print(f"\nFinal tour cost: {stats['final']:.2f}")
            print(f"Time: {stats['time']:.2f}s")
            
        except FileNotFoundError:
            print(f"Skipping {filepath} (not found)")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()