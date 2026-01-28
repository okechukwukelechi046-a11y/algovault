import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from algorithms.graph.dijkstra import (
    Graph, DijkstraSolver, PriorityQueueType, 
    IndexedPriorityQueue, Edge
)
import numpy as np

class TestIndexedPriorityQueue:
    """Test Indexed Priority Queue implementation."""
    
    def test_basic_operations(self):
        """Test insert, extract_min, and update operations."""
        ipq = IndexedPriorityQueue(5)
        
        # Insert elements
        ipq.insert(0, 10.0)
        ipq.insert(1, 5.0)
        ipq.insert(2, 15.0)
        
        assert ipq.contains(0) == True
        assert ipq.contains(3) == False
        
        # Extract minimum
        vertex, value = ipq.extract_min()
        assert vertex == 1
        assert value == 5.0
        
        # Update and extract next
        ipq.update(0, 3.0)
        vertex, value = ipq.extract_min()
        assert vertex == 0
        assert value == 3.0
    
    def test_decrease_key(self):
        """Test decrease_key operation."""
        ipq = IndexedPriorityQueue(3)
        ipq.insert(0, 10.0)
        ipq.insert(1, 20.0)
        ipq.insert(2, 30.0)
        
        ipq.decrease_key(2, 5.0)
        
        vertex, value = ipq.extract_min()
        assert vertex == 2
        assert value == 5.0
    
    def test_heap_property(self):
        """Verify heap property is maintained."""
        ipq = IndexedPriorityQueue(10)
        values = [23, 17, 31, 5, 12, 19, 8, 42, 3, 11]
        
        for i, val in enumerate(values):
            ipq.insert(i, val)
        
        extracted = []
        while not ipq.is_empty():
            vertex, value = ipq.extract_min()
            extracted.append(value)
        
        # Should be extracted in sorted order
        assert extracted == sorted(values)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        ipq = IndexedPriorityQueue(3)
        
        # Empty queue
        with pytest.raises(IndexError):
            ipq.extract_min()
        
        # Duplicate insert
        ipq.insert(0, 10.0)
        with pytest.raises(ValueError):
            ipq.insert(0, 20.0)
        
        # Update non-existent
        with pytest.raises(ValueError):
            ipq.update(1, 10.0)

class TestDijkstraSolver:
    """Test Dijkstra's algorithm implementation."""
    
    @pytest.fixture
    def simple_graph(self):
        """Create a simple test graph."""
        graph = Graph(5, directed=True)
        
        edges = [
            (0, 1, 4),
            (0, 2, 2),
            (1, 2, 1),
            (1, 3, 5),
            (2, 3, 8),
            (2, 4, 10),
            (3, 4, 2)
        ]
        
        for u, v, w in edges:
            graph.add_edge(u, v, w)
        
        return graph
    
    @pytest.fixture
    def complex_graph(self):
        """Create a more complex graph for testing."""
        graph = Graph(7, directed=False)
        
        # Create a small world network
        edges = [
            (0, 1, 7), (0, 2, 9), (0, 5, 14),
            (1, 2, 10), (1, 3, 15),
            (2, 3, 11), (2, 5, 2),
            (3, 4, 6),
            (4, 5, 9)
        ]
        
        for u, v, w in edges:
            graph.add_edge(u, v, w)
        
        return graph
    
    def test_simple_shortest_path(self, simple_graph):
        """Test shortest path in simple graph."""
        solver = DijkstraSolver(simple_graph, PriorityQueueType.EAGER)
        result = solver.solve(0)
        
        # Verify distances
        expected_distances = [0, 4, 2, 9, 11]
        
        for i, expected in enumerate(expected_distances):
            assert result['distances'][i] == pytest.approx(expected, rel=1e-9)
        
        # Verify path reconstruction
        path_to_4 = solver.get_path(4)
        assert path_to_4 == [0, 1, 3, 4] or path_to_4 == [0, 2, 4]
    
    def test_early_termination(self, complex_graph):
        """Test early termination when target is reached."""
        solver = DijkstraSolver(complex_graph, PriorityQueueType.LAZY)
        
        # Solve with early termination at vertex 4
        result = solver.solve(0, 4)
        
        # Distance to vertex 4 should be correct
        assert result['distances'][4] == pytest.approx(20, rel=1e-9)
        
        # Vertex 6 should still be unreachable
        assert result['distances'][6] == float('inf')
    
    def test_unreachable_vertex(self, simple_graph):
        """Test behavior with unreachable vertices."""
        # Add an isolated vertex
        simple_graph.add_edge(5, 5, 0)  # Self-loop to create vertex 5
        
        solver = DijkstraSolver(simple_graph, PriorityQueueType.EAGER)
        result = solver.solve(0)
        
        # Vertex 5 should be unreachable
        assert result['distances'][5] == float('inf')
        
        # Attempting to get path should raise error
        with pytest.raises(ValueError):
            solver.get_path(5)
    
    def test_negative_weight_error(self):
        """Test that negative weights raise error."""
        graph = Graph(3, directed=True)
        graph.add_edge(0, 1, 5)
        
        with pytest.raises(ValueError):
            graph.add_edge(1, 2, -3)
    
    def test_compare_implementations(self, complex_graph):
        """Compare lazy and eager implementations give same results."""
        # Lazy implementation
        solver_lazy = DijkstraSolver(complex_graph, PriorityQueueType.LAZY)
        result_lazy = solver_lazy.solve(0)
        
        # Eager implementation
        solver_eager = DijkstraSolver(complex_graph, PriorityQueueType.EAGER)
        result_eager = solver_eager.solve(0)
        
        # Both should give same distances
        for i in range(complex_graph.n):
            assert result_lazy['distances'][i] == pytest.approx(
                result_eager['distances'][i], rel=1e-9
            )
        
        # Both should find same paths
        for i in range(complex_graph.n):
            if result_lazy['distances'][i] < float('inf'):
                path_lazy = solver_lazy.get_path(i)
                path_eager = solver_eager.get_path(i)
                
                # Both paths should have same total weight
                def path_weight(path):
                    total = 0
                    for j in range(len(path) - 1):
                        u, v = path[j], path[j + 1]
                        # Find edge weight
                        for edge in complex_graph.adjacency[u]:
                            if edge.to == v:
                                total += edge.weight
                                break
                    return total
                
                assert path_weight(path_lazy) == pytest.approx(
                    path_weight(path_eager), rel=1e-9
                )
    
    def test_performance_benchmark(self, benchmark):
        """Benchmark Dijkstra performance."""
        # Create larger graph for benchmarking
        n = 100
        graph = Graph(n, directed=True)
        
        # Create random edges
        import random
        for _ in range(n * 5):  # 5 edges per vertex on average
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                weight = random.uniform(0.1, 10.0)
                try:
                    graph.add_edge(u, v, weight)
                except ValueError:
                    pass
        
        solver = DijkstraSolver(graph, PriorityQueueType.EAGER)
        
        # Benchmark
        def run_dijkstra():
            return solver.solve(0)
        
        result = benchmark(run_dijkstra)
        
        # Verify basic properties
        assert result['distances'][0] == 0
        assert all(d >= 0 for d in result['distances'] if d < float('inf'))
        assert result['iterations'] > 0

class TestGraphStructure:
    """Test Graph data structure."""
    
    def test_adjacency_list(self):
        """Test adjacency list representation."""
        graph = Graph(4, directed=True)
        
        graph.add_edge(0, 1, 5)
        graph.add_edge(0, 2, 3)
        graph.add_edge(1, 3, 2)
        graph.add_edge(2, 3, 7)
        
        # Test neighbors
        assert len(graph.neighbors(0)) == 2
        assert len(graph.neighbors(3)) == 0
        
        # Test edge properties
        edge = graph.neighbors(0)[0]
        assert edge.to == 1
        assert edge.weight == 5
    
    def test_undirected_graph(self):
        """Test undirected graph edges."""
        graph = Graph(3, directed=False)
        graph.add_edge(0, 1, 5)
        
        # Should create both directions
        assert len(graph.neighbors(0)) == 1
        assert len(graph.neighbors(1)) == 1
        
        edge_0 = graph.neighbors(0)[0]
        edge_1 = graph.neighbors(1)[0]
        
        assert edge_0.to == 1
        assert edge_1.to == 0
        assert edge_0.weight == edge_1.weight == 5
    
    def test_edge_list(self):
        """Test edge list tracking."""
        graph = Graph(3, directed=True)
        
        edges = [(0, 1, 3), (1, 2, 4), (0, 2, 7)]
        for u, v, w in edges:
            graph.add_edge(u, v, w)
        
        assert len(graph.edges) == 3
        
        # Verify edges are stored correctly
        for (u, v, w), stored in zip(edges, graph.edges):
            assert stored == (u, v, w)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
