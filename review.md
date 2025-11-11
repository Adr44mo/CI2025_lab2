Critical fixes
===============

1) Enforce tour permutation invariants after merges and conversions
   - After any local→global mapping or cluster merge, assert the tour is a permutation of 0..n-1. If not, build a deterministic fallback: append missing nodes in ascending order and continue.
   - Impact: eliminates incorrect final tours (missing or duplicated cities) that invalidate results.

2) Centralize asymmetric 2‑opt delta
   - Isolate exact delta computation into a single helper used by 2‑opt. Keep the existing D_sym O(1) filter (approximate) but compute the exact delta in one place to avoid inconsistencies. Do not add extra instrumentation or tests now.
   - Impact: small code change with immediate correctness benefits for swap decisions and reproducible behavior.

Nits 
----------------------

1. In merge code, prefer `for idx, val in enumerate(merged_tour): pos[val] = idx` when updating positions — clearer and slightly faster than full-range loops.
2. When concatenating lists for wrap-around segments, prefer explicit list copies (`list(a) + list(b)`) to avoid accidental numpy/python type mixes.
3. Keep `random_state` optional but thread a derived integer seed to worker processes (deterministic).

