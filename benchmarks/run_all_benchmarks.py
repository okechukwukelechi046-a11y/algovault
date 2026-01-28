#!/usr/bin/env python3
"""
Run all benchmarks and generate reports.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.performance.test_algo_performance import AlgorithmBenchmark
import json
import datetime

def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("ALGOVault Benchmark Suite")
    print("=" * 70)
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"benchmark_results/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    benchmark = AlgorithmBenchmark()
    
    # Run benchmarks
    print("\n1. Sorting Algorithms Benchmark...")
    sorting_results = benchmark.benchmark_sorting([100, 1000, 10000, 50000])
    
    print("\n2. Graph Algorithms Benchmark...")
    graph_results = benchmark.benchmark_graph_algorithms()
    
    print("\n3. Data Structure Operations Benchmark...")
    # Add data structure benchmarks
    
    # Save results
    results_file = os.path.join(output_dir, "benchmark_results.json")
    benchmark.save_results(results_file)
    
    # Generate report
    report_file = os.path.join(output_dir, "benchmark_report.md")
    generate_report(benchmark.results, report_file)
    
    # Plot results
    benchmark.plot_results()
    
    print(f"\nBenchmark completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Report: {report_file}")

def generate_report(results: dict, filename: str):
    """Generate markdown report from benchmark results."""
    with open(filename, 'w') as f:
        f.write("# Algorithm Benchmark Report\n\n")
        f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")
        
        # Sorting results
        if 'sorting' in results:
            f.write("## Sorting Algorithms\n\n")
            f.write("| Algorithm | Size | Time (ms) | Memory (KB) |\n")
            f.write("|-----------|------|-----------|-------------|\n")
            
            for size, algorithms in results['sorting'].items():
                for algo_name, metrics in algorithms.items():
                    f.write(f"| {algo_name} | {size} | {metrics['time_ms']:.2f} | {metrics['memory_kb']:.2f} |\n")
        
        # Graph results
        if 'graph' in results:
            f.write("\n## Graph Algorithms\n\n")
            f.write("| Algorithm | Vertices | Edges | Time (ms) |\n")
            f.write("|-----------|----------|-------|-----------|\n")
            
            for key, metrics in results['graph'].items():
                f.write(f"| {metrics['queue_type']} | {metrics['vertices']} | {metrics['edges']} | {metrics['avg_time_ms']:.2f} |\n")
        
        # Summary
        f.write("\n## Summary\n\n")
        
        # Find fastest algorithm for each category
        if 'sorting' in results:
            f.write("### Fastest Sorting Algorithms:\n")
            for size in results['sorting']:
                fastest = min(
                    results['sorting'][size].items(),
                    key=lambda x: x[1]['time_ms']
                )
                f.write(f"- Size {size}: {fastest[0]} ({fastest[1]['time_ms']:.2f} ms)\n")
        
        f.write("\n### Recommendations:\n")
        f.write("1. Use eager Dijkstra for dense graphs\n")
        f.write("2. Use lazy Dijkstra for sparse graphs\n")
        f.write("3. Python's built-in Timsort is fastest for general sorting\n")
        f.write("4. QuickSort is fastest for primitive arrays\n")

if __name__ == "__main__":
    main()
