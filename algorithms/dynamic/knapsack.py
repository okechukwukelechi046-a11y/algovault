"""
0/1 Knapsack problem solutions with multiple DP approaches and optimizations.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass
from functools import lru_cache
import time

@dataclass
class Item:
    """Knapsack item with weight and value."""
    weight: int
    value: int
    name: Optional[str] = None
    
    @property
    def density(self) -> float:
        """Value per unit weight."""
        return self.value / self.weight if self.weight > 0 else float('inf')

class KnapsackSolver:
    """
    Solves 0/1 knapsack problem using multiple DP approaches.
    """
    
    def __init__(self, items: List[Item], capacity: int):
        """
        Initialize knapsack solver.
        
        Args:
            items: List of available items
            capacity: Maximum weight capacity
        """
        self.items = items
        self.n = len(items)
        self.capacity = capacity
        
    def solve_recursive(self) -> Tuple[int, List[Item]]:
        """
        Recursive solution with memoization.
        Time: O(n * capacity), Space: O(n * capacity)
        """
        @lru_cache(maxsize=None)
        def dp(i: int, remaining: int) -> int:
            """Recursive helper with memoization."""
            if i == 0 or remaining == 0:
                return 0
            
            item = self.items[i - 1]
            
            # Can't take this item
            if item.weight > remaining:
                return dp(i - 1, remaining)
            
            # Max of taking or not taking the item
            take = item.value + dp(i - 1, remaining - item.weight)
            skip = dp(i - 1, remaining)
            
            return max(take, skip)
        
        # Reconstruct solution
        max_value = dp(self.n, self.capacity)
        selected = self._reconstruct(dp)
        
        return max_value, selected
    
    def solve_dp_basic(self) -> Tuple[int, List[Item]]:
        """
        Basic 2D DP solution.
        Time: O(n * capacity), Space: O(n * capacity)
        """
        dp = [[0] * (self.capacity + 1) for _ in range(self.n + 1)]
        
        # Build DP table
        for i in range(1, self.n + 1):
            item = self.items[i - 1]
            for w in range(1, self.capacity + 1):
                if item.weight > w:
                    dp[i][w] = dp[i - 1][w]
                else:
                    dp[i][w] = max(
                        dp[i - 1][w],
                        item.value + dp[i - 1][w - item.weight]
                    )
        
        max_value = dp[self.n][self.capacity]
        selected = self._reconstruct_from_dp(dp)
        
        return max_value, selected
    
    def solve_dp_optimized(self) -> Tuple[int, List[Item]]:
        """
        Space-optimized DP using 1D array.
        Time: O(n * capacity), Space: O(capacity)
        """
        dp = [0] * (self.capacity + 1)
        decisions = [[False] * (self.capacity + 1) for _ in range(self.n)]
        
        for i, item in enumerate(self.items):
            # Iterate backwards to avoid reusing items
            for w in range(self.capacity, item.weight - 1, -1):
                if item.value + dp[w - item.weight] > dp[w]:
                    dp[w] = item.value + dp[w - item.weight]
                    decisions[i][w] = True
        
        max_value = dp[self.capacity]
        selected = self._reconstruct_from_decisions(decisions, dp)
        
        return max_value, selected
    
    def solve_branch_and_bound(self) -> Tuple[int, List[Item]]:
        """
        Branch and bound with best-first search.
        More efficient for large capacity/values.
        """
        # Sort by value density for better bounds
        sorted_items = sorted(
            enumerate(self.items),
            key=lambda x: x[1].density,
            reverse=True
        )
        indices, sorted_items_list = zip(*sorted_items)
        
        # Calculate upper bounds using fractional knapsack
        prefix_weights = [0]
        prefix_values = [0]
        for item in sorted_items_list:
            prefix_weights.append(prefix_weights[-1] + item.weight)
            prefix_values.append(prefix_values[-1] + item.value)
        
        best_value = 0
        best_solution = []
        
        # Priority queue of states (negative upper bound for max heap)
        import heapq
        # Upper bound calculation using fractional knapsack
        def upper_bound(idx: int, weight: int, value: int) -> float:
            """Calculate upper bound using fractional knapsack."""
            if weight > self.capacity:
                return -float('inf')
            
            bound = value
            remaining = self.capacity - weight
            
            # Greedily add items by density
            j = idx
            while j < len(sorted_items_list) and remaining > 0:
                item = sorted_items_list[j]
                if item.weight <= remaining:
                    bound += item.value
                    remaining -= item.weight
                else:
                    # Take fraction of next item
                    bound += item.value * (remaining / item.weight)
                    remaining = 0
                j += 1
            
            return bound
        
        # Start state (index, weight, value, selected mask)
        start_ub = upper_bound(0, 0, 0)
        pq = [(-start_ub, 0, 0, 0, 0)]  # (negative_ub, idx, weight, value, mask)
        
        while pq:
            neg_ub, idx, weight, value, mask = heapq.heappop(pq)
            ub = -neg_ub
            
            # Prune if upper bound <= best
            if ub <= best_value:
                continue
            
            # Update best if leaf node
            if idx == len(sorted_items_list):
                if value > best_value and weight <= self.capacity:
                    best_value = value
                    best_solution = mask
                continue
            
            # Branch: take or don't take current item
            item = sorted_items_list[idx]
            
            # Don't take
            ub_skip = upper_bound(idx + 1, weight, value)
            if ub_skip > best_value:
                heapq.heappush(pq, (-ub_skip, idx + 1, weight, value, mask))
            
            # Take
            new_weight = weight + item.weight
            new_value = value + item.value
            if new_weight <= self.capacity:
                ub_take = upper_bound(idx + 1, new_weight, new_value)
                if ub_take > best_value:
                    new_mask = mask | (1 << idx)
                    heapq.heappush(pq, (-ub_take, idx + 1, new_weight, new_value, new_mask))
        
        # Convert mask to selected items
        selected = []
        for i in range(len(sorted_items_list)):
            if best_solution & (1 << i):
                original_idx = indices[i]
                selected.append(self.items[original_idx])
        
        return best_value, selected
    
    def solve_meet_in_middle(self) -> Tuple[int, List[Item]]:
        """
        Meet-in-the-middle for moderate n (up to ~40).
        Time: O(2^(n/2) * n), Space: O(2^(n/2))
        """
        def generate_subsets(items: List[Item]) -> List[Tuple[int, int, int]]:
            """Generate all subsets with total weight and value."""
            n = len(items)
            subsets = []
            
            for mask in range(1 << n):
                weight = 0
                value = 0
                for i in range(n):
                    if mask & (1 << i):
                        weight += items[i].weight
                        value += items[i].value
                subsets.append((weight, value, mask))
            
            return subsets
        
        # Split items into two halves
        mid = self.n // 2
        left = self.items[:mid]
        right = self.items[mid:]
        
        # Generate subsets for both halves
        left_subsets = generate_subsets(left)
        right_subsets = generate_subsets(right)
        
        # Sort right subsets by weight and keep best value for each weight
        right_by_weight = {}
        for weight, value, mask in right_subsets:
            if weight <= self.capacity:
                if weight not in right_by_weight or value > right_by_weight[weight][0]:
                    right_by_weight[weight] = (value, mask)
        
        # Convert to sorted list for binary search
        sorted_right = sorted(right_by_weight.items())
        right_weights = [w for w, _ in sorted_right]
        right_values = [v for _, (v, _) in sorted_right]
        right_masks = [m for _, (_, m) in sorted_right]
        
        # Prefix max values for binary search
        prefix_max = [0]
        for v in right_values:
            prefix_max.append(max(prefix_max[-1], v))
        
        best_value = 0
        best_left_mask = 0
        best_right_mask = 0
        
        # Try all left subsets, find best matching right subset
        for l_weight, l_value, l_mask in left_subsets:
            if l_weight > self.capacity:
                continue
            
            remaining = self.capacity - l_weight
            
            # Binary search for best right subset with weight <= remaining
            import bisect
            idx = bisect.bisect_right(right_weights, remaining) - 1
            
            if idx >= 0:
                r_value = prefix_max[idx + 1]
                total_value = l_value + r_value
                
                if total_value > best_value:
                    best_value = total_value
                    best_left_mask = l_mask
                    # Find which right mask gives this value
                    for i in range(idx + 1):
                        if right_values[i] == r_value:
                            best_right_mask = right_masks[i]
                            break
        
        # Reconstruct selected items
        selected = []
        for i in range(mid):
            if best_left_mask & (1 << i):
                selected.append(left[i])
        
        for i in range(len(right)):
            if best_right_mask & (1 << i):
                selected.append(right[i])
        
        return best_value, selected
    
    def _reconstruct(self, dp_func) -> List[Item]:
        """Reconstruct solution from memoized function."""
        selected = []
        i, w = self.n, self.capacity
        
        while i > 0 and w > 0:
            item = self.items[i - 1]
            
            if item.weight > w:
                i -= 1
            else:
                take = item.value + dp_func(i - 1, w - item.weight)
                skip = dp_func(i - 1, w)
                
                if take > skip:
                    selected.append(item)
                    w -= item.weight
                i -= 1
        
        return selected[::-1]
    
    def _reconstruct_from_dp(self, dp: List[List[int]]) -> List[Item]:
        """Reconstruct solution from 2D DP table."""
        selected = []
        i, w = self.n, self.capacity
        
        while i > 0 and w > 0:
            item = self.items[i - 1]
            
            if item.weight > w:
                i -= 1
            elif dp[i][w] == dp[i - 1][w]:
                i -= 1
            else:
                selected.append(item)
                w -= item.weight
                i -= 1
        
        return selected[::-1]
    
    def _reconstruct_from_decisions(self, decisions: List[List[bool]], 
                                   dp: List[int]) -> List[Item]:
        """Reconstruct solution from decision matrix."""
        selected = []
        w = self.capacity
        
        for i in reversed(range(self.n)):
            if decisions[i][w]:
                selected.append(self.items[i])
                w -= self.items[i].weight
        
        return selected[::-1]

class KnapsackBenchmark:
    """Benchmark different knapsack algorithms."""
    
    @staticmethod
    def compare_algorithms(n: int = 20, capacity: int = 50) -> Dict:
        """
        Compare all algorithms on random problem instance.
        
        Returns:
            Dictionary with results and timings
        """
        import random
        
        # Generate random items
        items = []
        for i in range(n):
            weight = random.randint(1, capacity // 2)
            value = random.randint(1, 100)
            items.append(Item(weight, value, f"Item_{i}"))
        
        solver = KnapsackSolver(items, capacity)
        results = {}
        
        algorithms = [
            ('recursive', solver.solve_recursive),
            ('dp_basic', solver.solve_dp_basic),
            ('dp_optimized', solver.solve_dp_optimized),
            ('branch_bound', solver.solve_branch_and_bound),
            ('meet_middle', solver.solve_meet_in_middle)
        ]
        
        for name, func in algorithms:
            start_time = time.perf_counter()
            try:
                value, selected = func()
                end_time = time.perf_counter()
                
                results[name] = {
                    'value': value,
                    'selected_count': len(selected),
                    'time_ms': (end_time - start_time) * 1000,
                    'valid': sum(item.weight for item in selected) <= capacity,
                    'success': True
                }
            except Exception as e:
                results[name] = {
                    'value': 0,
                    'selected_count': 0,
                    'time_ms': 0,
                    'error': str(e),
                    'success': False
                }
        
        return results
