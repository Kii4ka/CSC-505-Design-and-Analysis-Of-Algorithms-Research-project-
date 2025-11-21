import csv
import time
import statistics
import os
from algorithm_analysis import (
    insertion_sort, selection_sort, bubble_sort, merge_sort, quick_sort,
    heap_sort, shell_sort, counting_sort, radix_sort, bucket_sort,
    cocktail_shaker_sort, gnome_sort, cycle_sort, tim_sort, tree_sort,
    pancake_sort, flash_sort, tournament_sort, strand_sort, load_dataset
)

def measure_time(sort_func, arr, runs=10):
    times = []
    # Do correctness check only once
    arr_test = arr.copy()
    sort_func(arr_test)
    expected_sorted = sorted(arr)
    assert arr_test == expected_sorted, f"Sort failed for {sort_func.__name__}"
    
    # Time the actual runs
    for _ in range(runs):
        arr_copy = arr.copy()
        start = time.perf_counter_ns()
        sort_func(arr_copy)
        end = time.perf_counter_ns()
        times.append((end - start) / 1e9)
    return statistics.median(times)

def main():
    dataset_dir = 'datasets'
    base_result_file = 'algorithm_timings'
    
    # Define missing data points
    # Format: (dataset_type, size, algorithm_name, algorithm_function)
    missing_runs = [
        # Nearly sorted 100K
        # ('nearly_sorted', 100000, 'Tree Sort', tree_sort),  # COMPLETED - Error
        
        # Reverse sorted 100K
        # ('reverse_sorted', 100000, 'Bubble Sort', bubble_sort),  # COMPLETED - 406.04s
        # ('reverse_sorted', 100000, 'Bucket Sort', bucket_sort),  # COMPLETED - 0.025s
        # ('reverse_sorted', 100000, 'Cocktail-Shaker Sort', cocktail_shaker_sort),  # COMPLETED - 413.63s
        # ('reverse_sorted', 100000, 'Counting Sort', counting_sort),  # COMPLETED - 0.018s
        # ('reverse_sorted', 100000, 'Heap Sort', heap_sort),  # COMPLETED - 0.275s
        # ('reverse_sorted', 100000, 'Insertion Sort', insertion_sort),  # COMPLETED - 250.85s
        # ('reverse_sorted', 100000, 'Merge Sort', merge_sort),  # COMPLETED - 0.188s
        # ('reverse_sorted', 100000, 'Radix Sort', radix_sort),  # COMPLETED - 0.102s
        # ('reverse_sorted', 100000, 'Selection Sort', selection_sort),  # COMPLETED - 123.16s
        # ('reverse_sorted', 100000, 'Shell Sort', shell_sort),  # COMPLETED - 0.132s
        
        # Reverse sorted Quick Sort (1K, 10K, 100K)
        # ('reverse_sorted', 1000, 'Quick Sort', quick_sort),  # COMPLETED - Error
        # ('reverse_sorted', 10000, 'Quick Sort', quick_sort),  # COMPLETED - Error
        # ('reverse_sorted', 100000, 'Quick Sort', quick_sort),  # COMPLETED - Error
        
        # Reverse sorted Tree Sort (1K, 10K, 100K)
        # ('reverse_sorted', 1000, 'Tree Sort', tree_sort),  # COMPLETED - Error
        # ('reverse_sorted', 10000, 'Tree Sort', tree_sort),  # COMPLETED - Error
        # ('reverse_sorted', 100000, 'Tree Sort', tree_sort),  # COMPLETED - Error
        
        # Few unique 100K (all except flash, pancake, strand, tournament)
        # ('few_unique', 100000, 'Insertion Sort', insertion_sort),  # COMPLETED - 117.99s
        # ('few_unique', 100000, 'Selection Sort', selection_sort),  # COMPLETED - 149.56s
        ('few_unique', 100000, 'Bubble Sort', bubble_sort),
        ('few_unique', 100000, 'Merge Sort', merge_sort),
        ('few_unique', 100000, 'Quick Sort', quick_sort),
        ('few_unique', 100000, 'Heap Sort', heap_sort),
        ('few_unique', 100000, 'Shell Sort', shell_sort),
        ('few_unique', 100000, 'Counting Sort', counting_sort),
        ('few_unique', 100000, 'Radix Sort', radix_sort),
        ('few_unique', 100000, 'Bucket Sort', bucket_sort),
        ('few_unique', 100000, 'Cocktail-Shaker Sort', cocktail_shaker_sort),
        ('few_unique', 100000, 'Gnome Sort', gnome_sort),
        ('few_unique', 100000, 'Cycle Sort', cycle_sort),
        ('few_unique', 100000, 'TimSort (Python built-in)', tim_sort),
        ('few_unique', 100000, 'Tree Sort', tree_sort),
    ]
    
    print(f"Total missing benchmarks to run: {len(missing_runs)}")
    print("="*70)
    
    completed = 0
    failed = 0
    
    for dataset_type, size, algo_name, algo_func in missing_runs:
        dataset_filename = f"dataset_{size}_{dataset_type}.txt"
        dataset_path = os.path.join(dataset_dir, dataset_filename)
        result_file = f"{base_result_file}_{dataset_type}.csv"
        
        # Check if dataset exists
        if not os.path.exists(dataset_path):
            print(f"⚠️  Dataset {dataset_filename} not found. Skipping.")
            failed += 1
            continue
        
        # Load dataset
        print(f"\n[{completed + failed + 1}/{len(missing_runs)}] Running {algo_name} on {dataset_filename}...")
        arr = load_dataset(dataset_path)
        
        try:
            t = measure_time(algo_func, arr, runs=10)
            print(f"✓ {algo_name} completed: {t:.6f} seconds")
            
            # Append result to CSV
            with open(result_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([algo_name, t, dataset_filename.replace('dataset_', '', 1)])
            
            completed += 1
            
        except Exception as e:
            print(f"✗ Error running {algo_name}: {e}")
            
            # Write error to CSV
            with open(result_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([algo_name, 'Error', dataset_filename.replace('dataset_', '', 1)])
            
            failed += 1
    
    print("\n" + "="*70)
    print(f"Summary: {completed} completed, {failed} failed/skipped")
    print("="*70)

if __name__ == "__main__":
    main()
