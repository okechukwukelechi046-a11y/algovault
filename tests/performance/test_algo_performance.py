"""
Performance benchmarking for algorithms.
"""
import time
import tracemalloc
import statistics
from contextlib import contextmanager
from typing import Dict, List, Callable, Any
import json

@contextmanager
def measure_performance():
    """Context manager to measure time and memory usage."""
    tracemalloc.start()
    start_time = time.perf_counter()
    start_memory = tracemalloc.get_traced_memory()[0]
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        elapsed_time = end_time - start_time
        memory_used = current - start_memory
        peak_memory = peak
        
        print(f"Time: {elapsed_time:.4f} seconds")
        print(f"Memory: {memory_used / 1024:.2f} KB")
        print(f"Peak Memory: {peak_memory / 1024:.2f} KB")

class AlgorithmBenchmark:
    """Framework for benchmarking algorithms."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_sorting(self, array_sizes: List[int] = None):
        """Benchmark sorting algorithms."""
        if array_sizes is None:
            array_sizes = [100, 1000, 10000, 50000]
        
        from algorithms.sorting.quicksort import quick_sort
        from algorithms.sorting.mergesort import merge_sort
        from algorithms.sorting.heapsort import heap_sort
        
        algorithms = {
            'quick_sort': quick_sort,
            'merge_sort': merge_sort,
            'heap_sort': heap_sort,
            'timsort': sorted  # Python's built-in
        }
        
        results = {}
        
        for size in array_sizes:
            print(f"\nBenchmarking array size: {size}")
            results[size] = {}
            
            # Generate test data
            import random
            test_data = [random.randint(0, 1000000) for _ in range(size)]
            
            for name, algo in algorithms.items():
                # Create copy for each algorithm
                data_copy = test_data.copy()
                
                # Measure performance
                tracemalloc.start()
                start_time = time.perf_counter()
                
                if name == 'timsort':
                    result = algo(data_copy)
                else:
                    algo(data_copy)
                
                end_time = time.perf_counter()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                elapsed = end_time - start_time
                memory = current / 1024  # KB
                
                results[size][name] = {
                    'time_ms': elapsed * 1000,
                    'memory_kb': memory,
                    'peak_kb': peak / 1024
                }
                
                print(f"  {name}: {elapsed:.4f}s, {memory:.2f} KB")
        
        self.results['sorting'] = results
        return results
    
    def benchmark_graph_algorithms(self):
        """Benchmark graph algorithms."""
        from algorithms.graph.dijkstra import Graph, DijkstraSolver, PriorityQueueType
        
        results = {}
        
        # Test different graph sizes
        graph_sizes = [100, 500, 1000, 2000]
        densities = [0.1, 0.3, 0.5]
        
        for size in graph_sizes:
            for density in densities:
                print(f"\nGraph: {size} vertices, density: {density}")
                
                # Generate random graph
                graph = self._generate_random_graph(size, density)
                
                # Benchmark Dijkstra
                for pq_type in [PriorityQueueType.LAZY, PriorityQueueType.EAGER]:
                    solver = DijkstraSolver(graph, pq_type)
                    
                    times = []
                    for _ in range(5):  # Multiple runs
                        start = time.perf_counter()
                        solver.solve(0)
                        times.append(time.perf_counter() - start)
                    
                    avg_time = statistics.mean(times)
                    std_time = statistics.stdev(times) if len(times) > 1 else 0
                    
                    key = f"n={size}_d={density}_{pq_type.value}"
                    results[key] = {
                        'avg_time_ms': avg_time * 1000,
                        'std_time_ms': std_time * 1000,
                        'vertices': size,
                        'edges': len(graph.edges),
                        'density': density,
                        'queue_type': pq_type.value
                    }
                    
                    print(f"  {pq_type.value}: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        
        self.results['graph'] = results
        return results
    
    def _generate_random_graph(self, n: int, density: float) -> Graph:
        """Generate random weighted graph."""
        from algorithms.graph.dijkstra import Graph
        import random
        
        graph = Graph(n, directed=True)
        max_edges = int(n * (n - 1) * density)
        edges_added = 0
        
        while edges_added < max_edges:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                weight = random.uniform(0.1, 10.0)
                try:
                    graph.add_edge(u, v, weight)
                    edges_added += 1
                except ValueError:
                    pass
        
        return graph
    
    def save_results(self, filename: str):
        """Save benchmark results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def plot_results(self):
        """Plot benchmark results."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Plot sorting results
        if 'sorting' in self.results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            sorting_data = self.results['sorting']
            sizes = list(sorting_data.keys())
            
            for algo in ['quick_sort', 'merge_sort', 'heap_sort', 'timsort']:
                times = [sorting_data[size][algo]['time_ms'] for size in sizes]
                ax1.plot(sizes, times, marker='o', label=algo)
            
            ax1.set_xlabel('Array Size')
            ax1.set_ylabel('Time (ms)')
            ax1.set_title('Sorting Algorithm Performance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Memory usage
            for algo in ['quick_sort', 'merge_sort', 'heap_sort']:
                memory = [sorting_data[size][algo]['memory_kb'] for size in sizes]
                ax2.plot(sizes, memory, marker='s', label=algo)
            
            ax2.set_xlabel('Array Size')
            ax2.set_ylabel('Memory (KB)')
            ax2.set_title('Sorting Algorithm Memory Usage')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        # Plot graph results
        if 'graph' in self.results:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            graph_data = self.results['graph']
            
            # Group by density
            densities = set()
            for key in graph_data:
                if 'density' in graph_data[key]:
                    densities.add(graph_data[key]['density'])
            
            for density in sorted(densities):
                lazy_times = []
                eager_times = []
                sizes = []
                
                for key, data in graph_data.items():
                    if data['density'] == density:
                        sizes.append(data['vertices'])
                        if 'lazy' in key:
                            lazy_times.append(data['avg_time_ms'])
                        elif 'eager' in key:
                            eager_times.append(data['avg_time_ms'])
                
                # Sort by size
                sizes, lazy_times, eager_times = zip(*sorted(
                    zip(sizes, lazy_times, eager_times)
                ))
                
                ax.plot(sizes, lazy_times, marker='o', 
                       label=f'Lazy (d={density})')
                ax.plot(sizes, eager_times, marker='s', 
                       label=f'Eager (d={density})')
            
            ax.set_xlabel('Number of Vertices')
            ax.set_ylabel('Time (ms)')
            ax.set_title('Dijkstra Algorithm Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    benchmark = AlgorithmBenchmark()
    
    print("=" * 60)
    print("ALGORITHM PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Run benchmarks
    print("\n1. Benchmarking Sorting Algorithms...")
    benchmark.benchmark_sorting([100, 1000, 5000, 10000])
    
    print("\n2. Benchmarking Graph Algorithms...")
    benchmark.benchmark_graph_algorithms()
    
    # Save and plot results
    benchmark.save_results("benchmark_results.json")
    benchmark.plot_results()
    
    print("\nBenchmark completed! Results saved to benchmark_results.json")
