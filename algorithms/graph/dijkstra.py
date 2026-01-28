"""
Dijkstra's algorithm for finding shortest paths in weighted graphs.
Implements both lazy and eager versions with performance optimizations.
"""

import heapq
import math
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import time

class PriorityQueueType(Enum):
    """Type of priority queue implementation."""
    LAZY = "lazy"      # Simple heap-based (lazy deletion)
    EAGER = "eager"    # Indexed priority queue
    FIBONACCI = "fibonacci"  # Fibonacci heap (optimal theoretical)

@dataclass
class Edge:
    """Represents a directed edge in a graph."""
    to: int
    weight: float
    data: Optional[dict] = field(default_factory=dict)

class Graph:
    """Adjacency list representation of a weighted graph."""
    
    def __init__(self, n: int, directed: bool = True):
        """
        Initialize graph with n vertices.
        
        Args:
            n: Number of vertices (0 to n-1)
            directed: Whether the graph is directed
        """
        self.n = n
        self.directed = directed
        self.adjacency: List[List[Edge]] = [[] for _ in range(n)]
        self.edges: List[Tuple[int, int, float]] = []
        
    def add_edge(self, u: int, v: int, weight: float, **kwargs) -> None:
        """
        Add an edge from u to v with given weight.
        
        Args:
            u: Source vertex
            v: Destination vertex
            weight: Edge weight (must be non-negative for Dijkstra)
            **kwargs: Additional edge metadata
        """
        if weight < 0:
            raise ValueError("Dijkstra requires non-negative weights")
        
        self.adjacency[u].append(Edge(to=v, weight=weight, data=kwargs))
        self.edges.append((u, v, weight))
        
        if not self.directed:
            self.adjacency[v].append(Edge(to=u, weight=weight, data=kwargs))
    
    def neighbors(self, u: int) -> List[Edge]:
        """Get all outgoing edges from vertex u."""
        return self.adjacency[u]
    
    @property
    def vertices(self) -> range:
        """Get range of vertices."""
        return range(self.n)

class IndexedPriorityQueue:
    """
    Indexed priority queue (minimum) supporting O(log n) updates.
    Essential for efficient Dijkstra implementation.
    """
    
    def __init__(self, size: int):
        """
        Initialize indexed priority queue.
        
        Args:
            size: Maximum number of elements (vertices)
        """
        self.size = size
        self.values: List[float] = [float('inf')] * size
        self.pm: List[int] = [-1] * size  # Position map: vertex -> heap index
        self.im: List[int] = [-1] * size  # Inverse map: heap index -> vertex
        self.heap_size = 0
    
    def is_empty(self) -> bool:
        """Check if priority queue is empty."""
        return self.heap_size == 0
    
    def contains(self, vertex: int) -> bool:
        """Check if vertex exists in the queue."""
        return self.pm[vertex] != -1
    
    def insert(self, vertex: int, value: float) -> None:
        """
        Insert vertex with given value.
        
        Args:
            vertex: Vertex index
            value: Priority value
        """
        if self.contains(vertex):
            raise ValueError(f"Vertex {vertex} already in queue")
        
        self.values[vertex] = value
        self.pm[vertex] = self.heap_size
        self.im[self.heap_size] = vertex
        self._swim(self.heap_size)
        self.heap_size += 1
    
    def update(self, vertex: int, value: float) -> None:
        """
        Update priority of existing vertex.
        
        Args:
            vertex: Vertex index
            value: New priority value
        """
        if not self.contains(vertex):
            raise ValueError(f"Vertex {vertex} not in queue")
        
        old_value = self.values[vertex]
        self.values[vertex] = value
        
        # Swim if improved, sink if worsened
        if value < old_value:
            self._swim(self.pm[vertex])
        else:
            self._sink(self.pm[vertex])
    
    def decrease_key(self, vertex: int, value: float) -> None:
        """
        Decrease priority of vertex (optimization for Dijkstra).
        
        Args:
            vertex: Vertex index
            value: New priority value (must be less than current)
        """
        if not self.contains(vertex):
            raise ValueError(f"Vertex {vertex} not in queue")
        if value >= self.values[vertex]:
            raise ValueError(f"New value must be less than current")
        
        self.values[vertex] = value
        self._swim(self.pm[vertex])
    
    def extract_min(self) -> Tuple[int, float]:
        """
        Remove and return vertex with minimum value.
        
        Returns:
            Tuple of (vertex, value)
        """
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        
        min_vertex = self.im[0]
        min_value = self.values[min_vertex]
        
        self._swap(0, self.heap_size - 1)
        self.heap_size -= 1
        self._sink(0)
        
        # Clean up
        self.pm[min_vertex] = -1
        self.im[self.heap_size] = -1
        
        return min_vertex, min_value
    
    def peek(self) -> Tuple[int, float]:
        """
        Get vertex with minimum value without removal.
        
        Returns:
            Tuple of (vertex, value)
        """
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        
        min_vertex = self.im[0]
        return min_vertex, self.values[min_vertex]
    
    def _swap(self, i: int, j: int) -> None:
        """Swap two elements in the heap."""
        self.im[i], self.im[j] = self.im[j], self.im[i]
        self.pm[self.im[i]] = i
        self.pm[self.im[j]] = j
    
    def _swim(self, k: int) -> None:
        """Bubble element up the heap."""
        while k > 0 and self._less(k, (k - 1) // 2):
            parent = (k - 1) // 2
            self._swap(k, parent)
            k = parent
    
    def _sink(self, k: int) -> None:
        """Bubble element down the heap."""
        while 2 * k + 1 < self.heap_size:
            j = 2 * k + 1
            if j + 1 < self.heap_size and self._less(j + 1, j):
                j += 1
            if not self._less(j, k):
                break
            self._swap(k, j)
            k = j
    
    def _less(self, i: int, j: int) -> bool:
        """Compare two heap elements."""
        return self.values[self.im[i]] < self.values[self.im[j]]

class DijkstraSolver:
    """
    Dijkstra's algorithm implementation with multiple optimizations.
    Supports finding shortest paths from single source to all vertices.
    """
    
    def __init__(self, graph: Graph, pq_type: PriorityQueueType = PriorityQueueType.EAGER):
        """
        Initialize Dijkstra solver.
        
        Args:
            graph: Weighted graph (non-negative edges)
            pq_type: Type of priority queue to use
        """
        self.graph = graph
        self.pq_type = pq_type
        self.dist: List[float] = []
        self.prev: List[Optional[int]] = []
        self.visited: List[bool] = []
        
    def solve(self, start: int, end: Optional[int] = None) -> Dict:
        """
        Find shortest paths from start vertex.
        
        Args:
            start: Source vertex
            end: Optional destination vertex (for early termination)
            
        Returns:
            Dictionary containing distances, paths, and metrics
        """
        n = self.graph.n
        self.dist = [float('inf')] * n
        self.prev = [-1] * n
        self.visited = [False] * n
        
        self.dist[start] = 0
        
        start_time = time.perf_counter()
        
        if self.pq_type == PriorityQueueType.EAGER:
            result = self._solve_eager(start, end)
        elif self.pq_type == PriorityQueueType.LAZY:
            result = self._solve_lazy(start, end)
        else:
            raise ValueError(f"Unsupported priority queue type: {self.pq_type}")
        
        end_time = time.perf_counter()
        
        result['execution_time_ms'] = (end_time - start_time) * 1000
        result['vertices_relaxed'] = sum(self.visited)
        
        return result
    
    def _solve_eager(self, start: int, end: Optional[int]) -> Dict:
        """Eager Dijkstra using indexed priority queue."""
        n = self.graph.n
        ipq = IndexedPriorityQueue(n)
        ipq.insert(start, 0)
        
        iterations = 0
        
        while not ipq.is_empty():
            iterations += 1
            u, dist_u = ipq.extract_min()
            
            # Early termination if we found the target
            if end is not None and u == end:
                break
            
            self.visited[u] = True
            
            # Optimization: Skip stale entries
            if dist_u > self.dist[u]:
                continue
            
            for edge in self.graph.neighbors(u):
                v = edge.to
                
                # Skip if already visited (not necessary but good practice)
                if self.visited[v]:
                    continue
                
                new_dist = self.dist[u] + edge.weight
                
                if new_dist < self.dist[v]:
                    self.dist[v] = new_dist
                    self.prev[v] = u
                    
                    if ipq.contains(v):
                        ipq.decrease_key(v, new_dist)
                    else:
                        ipq.insert(v, new_dist)
        
        return {
            'distances': self.dist,
            'predecessors': self.prev,
            'iterations': iterations,
            'queue_type': 'eager'
        }
    
    def _solve_lazy(self, start: int, end: Optional[int]) -> Dict:
        """Lazy Dijkstra using standard heap (simpler but slower)."""
        import heapq
        
        n = self.graph.n
        pq = [(0, start)]  # (distance, vertex)
        iterations = 0
        
        while pq:
            iterations += 1
            dist_u, u = heapq.heappop(pq)
            
            # Skip stale entries
            if dist_u > self.dist[u]:
                continue
            
            # Early termination
            if end is not None and u == end:
                break
            
            self.visited[u] = True
            
            for edge in self.graph.neighbors(u):
                v = edge.to
                
                if self.visited[v]:
                    continue
                
                new_dist = self.dist[u] + edge.weight
                
                if new_dist < self.dist[v]:
                    self.dist[v] = new_dist
                    self.prev[v] = u
                    heapq.heappush(pq, (new_dist, v))
        
        return {
            'distances': self.dist,
            'predecessors': self.prev,
            'iterations': iterations,
            'queue_type': 'lazy'
        }
    
    def get_path(self, target: int) -> List[int]:
        """
        Reconstruct shortest path to target vertex.
        
        Args:
            target: Destination vertex
            
        Returns:
            List of vertices from source to target (inclusive)
            
        Raises:
            ValueError: If no path exists
        """
        if self.prev[target] == -1 and self.dist[target] == float('inf'):
            raise ValueError(f"No path to vertex {target}")
        
        path = []
        current = target
        
        while current != -1:
            path.append(current)
            current = self.prev[current]
        
        return path[::-1]
    
    def get_all_paths(self) -> Dict[int, List[int]]:
        """
        Get shortest paths to all reachable vertices.
        
        Returns:
            Dictionary mapping vertex to path from source
        """
        paths = {}
        
        for v in self.graph.vertices:
            if self.dist[v] < float('inf'):
                try:
                    paths[v] = self.get_path(v)
                except ValueError:
                    pass
        
        return paths

class DijkstraBenchmark:
    """Benchmark different Dijkstra implementations."""
    
    @staticmethod
    def run_benchmark(graph_sizes: List[int], density: float = 0.3) -> pd.DataFrame:
        """
        Run comprehensive benchmark on different graph sizes.
        
        Args:
            graph_sizes: List of graph sizes to test
            density: Edge density (0 to 1)
            
        Returns:
            DataFrame with benchmark results
        """
        results = []
        
        for n in graph_sizes:
            # Generate random graph
            graph = Graph(n, directed=True)
            edges_added = 0
            max_edges = int(n * (n - 1) * density)
            
            while edges_added < max_edges:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    weight = random.uniform(0.1, 100.0)
                    try:
                        graph.add_edge(u, v, weight)
                        edges_added += 1
                    except ValueError:
                        pass
            
            # Test both implementations
            for pq_type in [PriorityQueueType.LAZY, PriorityQueueType.EAGER]:
                solver = DijkstraSolver(graph, pq_type)
                
                # Warm up
                solver.solve(0)
                
                # Time multiple runs
                times = []
                memory_usage = []
                
                for _ in range(5):
                    start_mem = memory_profiler.memory_usage()[0]
                    result = solver.solve(0)
                    end_mem = memory_profiler.memory_usage()[0]
                    
                    times.append(result['execution_time_ms'])
                    memory_usage.append(end_mem - start_mem)
                
                results.append({
                    'vertices': n,
                    'edges': edges_added,
                    'queue_type': pq_type.value,
                    'avg_time_ms': np.mean(times),
                    'std_time_ms': np.std(times),
                    'avg_memory_mb': np.mean(memory_usage),
                    'iterations': result['iterations']
                })
        
        return pd.DataFrame(results)

# Example usage and demonstration
if __name__ == "__main__":
    # Create a sample graph
    graph = Graph(6, directed=True)
    
    # Add edges (u, v, weight)
    edges = [
        (0, 1, 4), (0, 2, 2),
        (1, 2, 1), (1, 3, 5),
        (2, 3, 8), (2, 4, 10),
        (3, 4, 2), (3, 5, 6),
        (4, 5, 3)
    ]
    
    for u, v, w in edges:
        graph.add_edge(u, v, w)
    
    # Solve using eager implementation
    solver = DijkstraSolver(graph, PriorityQueueType.EAGER)
    result = solver.solve(0)
    
    print("Distances from vertex 0:")
    for i, dist in enumerate(result['distances']):
        print(f"  to {i}: {dist:.2f}")
    
    print(f"\nPath to vertex 5: {solver.get_path(5)}")
    print(f"Execution time: {result['execution_time_ms']:.2f} ms")
    print(f"Queue type: {result['queue_type']}")
