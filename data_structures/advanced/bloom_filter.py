"""
Bloom Filter implementation with optimal hash functions and false positive analysis.
"""

import math
import mmap
import struct
from typing import Callable, List, Optional
import hashlib
from bitarray import bitarray
import numpy as np

class BloomFilter:
    """
    Space-efficient probabilistic data structure for membership testing.
    
    Attributes:
        n: Expected number of elements
        p: Acceptable false positive probability
        m: Optimal number of bits
        k: Optimal number of hash functions
    """
    
    def __init__(self, n: int, p: float = 0.01):
        """
        Initialize Bloom filter with optimal parameters.
        
        Args:
            n: Expected number of elements to store
            p: Acceptable false positive probability (0 < p < 1)
        """
        if not (0 < p < 1):
            raise ValueError("False positive probability must be between 0 and 1")
        if n <= 0:
            raise ValueError("Number of elements must be positive")
        
        self.n = n
        self.p = p
        
        # Calculate optimal parameters
        self.m = self._optimal_bits(n, p)
        self.k = self._optimal_hashes(self.m, n)
        
        # Initialize bit array
        self.bits = bitarray(self.m)
        self.bits.setall(0)
        
        # Initialize hash functions
        self.hash_functions = self._create_hash_functions()
        
        # Track actual number of insertions
        self.count = 0
    
    @staticmethod
    def _optimal_bits(n: int, p: float) -> int:
        """Calculate optimal number of bits."""
        m = - (n * math.log(p)) / (math.log(2) ** 2)
        return int(math.ceil(m))
    
    @staticmethod
    def _optimal_hashes(m: int, n: int) -> int:
        """Calculate optimal number of hash functions."""
        k = (m / n) * math.log(2)
        return max(1, int(math.ceil(k)))
    
    def _create_hash_functions(self) -> List[Callable[[bytes], int]]:
        """Create k independent hash functions using double hashing."""
        functions = []
        
        # Use different seeds for hash functions
        for i in range(self.k):
            def make_hash(seed: int) -> Callable[[bytes], int]:
                def hash_func(data: bytes) -> int:
                    # Create hash using double hashing technique
                    h1 = hashlib.sha256(data + struct.pack('>I', seed)).digest()
                    h2 = hashlib.md5(data + struct.pack('>I', seed ^ 0xFFFFFFFF)).digest()
                    
                    # Combine hashes and map to [0, m-1]
                    combined = int.from_bytes(h1 + h2, 'big')
                    return combined % self.m
                return hash_func
            
            functions.append(make_hash(i))
        
        return functions
    
    def add(self, item: bytes) -> None:
        """
        Add an item to the Bloom filter.
        
        Args:
            item: Bytes to add
        """
        for hash_func in self.hash_functions:
            index = hash_func(item)
            self.bits[index] = 1
        
        self.count += 1
    
    def add_string(self, s: str) -> None:
        """Add a string to the Bloom filter."""
        self.add(s.encode('utf-8'))
    
    def contains(self, item: bytes) -> bool:
        """
        Check if item is probably in the Bloom filter.
        
        Returns:
            True if item is probably in the set (may have false positives)
            False if item is definitely not in the set (no false negatives)
        """
        for hash_func in self.hash_functions:
            index = hash_func(item)
            if not self.bits[index]:
                return False
        return True
    
    def contains_string(self, s: str) -> bool:
        """Check if string is probably in the Bloom filter."""
        return self.contains(s.encode('utf-8'))
    
    def clear(self) -> None:
        """Clear all items from the Bloom filter."""
        self.bits.setall(0)
        self.count = 0
    
    @property
    def estimated_false_positive_rate(self) -> float:
        """Estimate current false positive probability."""
        if self.count == 0:
            return 0.0
        
        # Theoretical false positive rate
        return (1 - math.exp(-self.k * self.count / self.m)) ** self.k
    
    @property
    def load_factor(self) -> float:
        """Get current load factor (fraction of bits set)."""
        return sum(self.bits) / self.m
    
    def union(self, other: 'BloomFilter') -> 'BloomFilter':
        """
        Compute union of two Bloom filters (must have same parameters).
        
        Returns:
            New Bloom filter containing union
        """
        if self.m != other.m or self.k != other.k:
            raise ValueError("Bloom filters must have same parameters for union")
        
        result = BloomFilter(self.n, self.p)
        result.bits = self.bits | other.bits
        result.count = max(self.count, other.count)  # Approximation
        return result
    
    def intersection(self, other: 'BloomFilter') -> 'BloomFilter':
        """
        Compute intersection of two Bloom filters.
        
        Note: Result may have higher false positive rate.
        """
        if self.m != other.m or self.k != other.k:
            raise ValueError("Bloom filters must have same parameters for intersection")
        
        result = BloomFilter(self.n, self.p)
        result.bits = self.bits & other.bits
        result.count = min(self.count, other.count)  # Approximation
        return result
    
    def save(self, filename: str) -> None:
        """Save Bloom filter to file."""
        with open(filename, 'wb') as f:
            # Save metadata
            f.write(struct.pack('QQQdQ', 
                               self.n, self.m, self.k, 
                               self.p, self.count))
            # Save bit array
            self.bits.tofile(f)
    
    @classmethod
    def load(cls, filename: str) -> 'BloomFilter':
        """Load Bloom filter from file."""
        with open(filename, 'rb') as f:
            # Read metadata
            metadata = f.read(struct.calcsize('QQQdQ'))
            n, m, k, p, count = struct.unpack('QQQdQ', metadata)
            
            # Create Bloom filter
            bf = cls(n, p)
            
            # Verify parameters match
            if bf.m != m or bf.k != k:
                raise ValueError("Stored parameters don't match calculated ones")
            
            # Read bit array
            bf.bits = bitarray()
            bf.bits.fromfile(f)
            bf.count = count
        
        return bf
    
    def __str__(self) -> str:
        """String representation of Bloom filter."""
        return (f"BloomFilter(n={self.n}, p={self.p}, "
                f"m={self.m}, k={self.k}, "
                f"load={self.load_factor:.3f}, "
                f"estimated_fp={self.estimated_false_positive_rate:.6f})")

class CountingBloomFilter:
    """
    Counting Bloom Filter allows deletions by using counters instead of bits.
    """
    
    def __init__(self, n: int, p: float = 0.01, counter_bits: int = 4):
        """
        Initialize Counting Bloom Filter.
        
        Args:
            n: Expected number of elements
            p: False positive probability
            counter_bits: Bits per counter (default 4, max 255)
        """
        self.n = n
        self.p = p
        self.counter_bits = counter_bits
        self.max_counter = (1 << counter_bits) - 1
        
        # Calculate parameters
        self.m = self._optimal_bits(n, p)
        self.k = self._optimal_hashes(self.m, n)
        
        # Initialize counters
        self.counters = bytearray(self.m)
        
        # Hash functions
        self.hash_functions = self._create_hash_functions()
    
    def add(self, item: bytes) -> None:
        """Add item to the filter."""
        for hash_func in self.hash_functions:
            idx = hash_func(item)
            if self.counters[idx] < self.max_counter:
                self.counters[idx] += 1
    
    def remove(self, item: bytes) -> bool:
        """
        Remove item from the filter.
        
        Returns:
            True if removal was successful, False if item might not exist
        """
        # First check if item might be in the filter
        if not self.contains(item):
            return False
        
        # Decrement counters
        for hash_func in self.hash_functions:
            idx = hash_func(item)
            if self.counters[idx] > 0:
                self.counters[idx] -= 1
        
        return True
    
    def contains(self, item: bytes) -> bool:
        """Check if item is probably in the filter."""
        for hash_func in self.hash_functions:
            idx = hash_func(item)
            if self.counters[idx] == 0:
                return False
        return True
    
    @property
    def load_factor(self) -> float:
        """Get current load factor."""
        nonzero = sum(1 for c in self.counters if c > 0)
        return nonzero / self.m

# Benchmarking and analysis
class BloomFilterAnalyzer:
    """Analyze Bloom filter performance characteristics."""
    
    @staticmethod
    def analyze_false_positives(bf: BloomFilter, test_items: List[bytes], 
                               negative_items: List[bytes]) -> dict:
        """
        Analyze false positive rate empirically.
        
        Returns:
            Dictionary with analysis results
        """
        # Add all test items
        for item in test_items:
            bf.add(item)
        
        # Check for false positives
        false_positives = 0
        for item in negative_items:
            if bf.contains(item):
                false_positives += 1
        
        empirical_fp_rate = false_positives / len(negative_items)
        theoretical_fp_rate = bf.estimated_false_positive_rate
        
        return {
            'empirical_fp_rate': empirical_fp_rate,
            'theoretical_fp_rate': theoretical_fp_rate,
            'false_positives': false_positives,
            'total_tests': len(negative_items),
            'load_factor': bf.load_factor,
            'bits_per_element': bf.m / len(test_items)
        }
    
    @staticmethod
    def plot_memory_vs_error(n_range: range, p_values: List[float]):
        """
        Plot memory usage vs error rate for different parameters.
        
        This helps visualize the space/accuracy trade-off.
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        for p in p_values:
            bits_per_element = []
            for n in n_range:
                m = BloomFilter._optimal_bits(n, p)
                bits_per_element.append(m / n)
            
            plt.plot(n_range, bits_per_element, 
                    label=f'p={p}', marker='o')
        
        plt.xlabel('Number of Elements (n)')
        plt.ylabel('Bits per Element')
        plt.title('Bloom Filter Space Efficiency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
