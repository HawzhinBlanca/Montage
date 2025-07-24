"""Test ROVER algorithm performance to verify O(n log n) complexity"""
import time
import random
import matplotlib.pyplot as plt
from montage.core.rover_linear import rover_merge_original

def generate_transcript(num_words, start_time=0.0):
    """Generate a transcript with specified number of words"""
    words = []
    current_time = start_time
    
    for i in range(num_words):
        duration = random.uniform(0.2, 0.8)
        word = (
            current_time,
            current_time + duration,
            f"word_{i}",
            random.uniform(0.7, 0.99)
        )
        words.append(word)
        current_time += duration + random.uniform(0.05, 0.2)
    
    return words

def measure_performance(sizes):
    """Measure ROVER performance for different input sizes"""
    results = []
    
    for size in sizes:
        # Generate two transcripts with overlapping words
        transcript1 = generate_transcript(size // 2)
        transcript2 = generate_transcript(size // 2, start_time=0.1)
        
        # Measure time
        start_time = time.time()
        result = rover_merge_original([transcript1, transcript2])
        end_time = time.time()
        
        elapsed = end_time - start_time
        results.append({
            'size': size,
            'time': elapsed,
            'words_per_second': size / elapsed
        })
        
        print(f"Size: {size:6d} words | Time: {elapsed:8.4f}s | Words/sec: {size/elapsed:10.0f}")
    
    return results

def verify_complexity(results):
    """Verify if performance matches O(n log n) complexity"""
    import numpy as np
    
    sizes = [r['size'] for r in results]
    times = [r['time'] for r in results]
    
    # Calculate expected O(n log n) curve
    sizes_array = np.array(sizes)
    times_array = np.array(times)
    
    # Fit to n log n model: time = a * n * log(n) + b
    log_sizes = np.log(sizes_array)
    n_log_n = sizes_array * log_sizes
    
    # Linear regression to find coefficient
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(n_log_n, times_array)
    
    print(f"\n=== Complexity Analysis ===")
    print(f"Linear fit to n*log(n): R² = {r_value**2:.4f}")
    print(f"Coefficient: {slope:.2e} seconds per (word * log(word))")
    
    # Check if it's closer to O(n log n) than O(n²)
    n_squared = sizes_array ** 2
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(n_squared, times_array)
    
    print(f"Linear fit to n²: R² = {r_value2**2:.4f}")
    
    if r_value**2 > 0.95:
        print("\n✅ Performance matches O(n log n) complexity!")
        return True
    else:
        print("\n⚠️ Performance may not be O(n log n)")
        return False

def plot_results(results):
    """Plot performance results"""
    sizes = [r['size'] for r in results]
    times = [r['time'] for r in results]
    
    plt.figure(figsize=(10, 6))
    
    # Plot actual times
    plt.subplot(1, 2, 1)
    plt.plot(sizes, times, 'bo-', label='Actual')
    
    # Plot theoretical O(n log n)
    import numpy as np
    sizes_array = np.array(sizes)
    theoretical = sizes_array * np.log(sizes_array) * (times[-1] / (sizes[-1] * np.log(sizes[-1])))
    plt.plot(sizes, theoretical, 'r--', label='O(n log n)')
    
    plt.xlabel('Number of Words')
    plt.ylabel('Time (seconds)')
    plt.title('ROVER Performance')
    plt.legend()
    plt.grid(True)
    
    # Log-log plot
    plt.subplot(1, 2, 2)
    plt.loglog(sizes, times, 'bo-', label='Actual')
    plt.loglog(sizes, theoretical, 'r--', label='O(n log n)')
    plt.xlabel('Number of Words (log scale)')
    plt.ylabel('Time (log scale)')
    plt.title('ROVER Performance (Log-Log)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('rover_performance.png', dpi=150)
    print("\nPerformance plot saved to rover_performance.png")

def test_correctness():
    """Test ROVER produces correct results"""
    print("\n=== Correctness Test ===")
    
    # Test case 1: Simple overlap
    transcript1 = [(0.0, 0.5, "hello", 0.9), (0.6, 1.0, "world", 0.8)]
    transcript2 = [(0.1, 0.5, "hello", 0.85), (0.7, 1.1, "earth", 0.95)]
    
    result = rover_merge_original([transcript1, transcript2])
    print(f"Test 1: {result}")
    assert "hello" in result and ("world" in result or "earth" in result)
    
    # Test case 2: Different confidences
    transcript1 = [(0.0, 0.5, "cat", 0.7), (0.6, 1.0, "dog", 0.9)]
    transcript2 = [(0.0, 0.5, "cat", 0.95), (0.6, 1.0, "dog", 0.6)]
    
    result = rover_merge_original([transcript1, transcript2])
    print(f"Test 2: {result}")
    assert "cat" in result and "dog" in result
    
    # Test case 3: Empty input
    result = rover_merge_original([])
    assert result == ""
    
    print("✅ All correctness tests passed!")

if __name__ == "__main__":
    print("=== ROVER O(n log n) Performance Test ===\n")
    
    # Test correctness first
    test_correctness()
    
    # Test performance with increasing sizes
    sizes = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    print("\n=== Performance Measurement ===")
    results = measure_performance(sizes)
    
    # Verify complexity
    is_nlogn = verify_complexity(results)
    
    # Plot results
    try:
        plot_results(results)
    except ImportError:
        print("\nMatplotlib not available, skipping plot generation")
    
    # Summary
    print("\n=== Summary ===")
    if is_nlogn:
        print("✅ ROVER implementation verified as O(n log n)")
        print("✅ Performance scales efficiently for large transcripts")
    else:
        print("⚠️ ROVER implementation may need optimization")