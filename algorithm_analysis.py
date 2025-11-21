import time
import csv
import os
import statistics
#1 Bubble Sort 
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
#2 Insertion sort
def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j = j - 1
        arr[j + 1] = key

#3 Selection sort
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[min_idx] > arr[j]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]


#4 Cocktail-Shaker sort
def cocktail_shaker_sort(arr):
    n = len(arr)
    swapped = True
    start = 0
    end = n - 1
    while swapped:
        swapped = False
        for i in range(start, end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                swapped = True
        if not swapped:
            break
        swapped = False
        end = end - 1
        for i in range(end, start, -1):
            if arr[i - 1] > arr[i]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
                swapped = True
        start = start + 1

#5 Shell sort
def shell_sort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2


#6 Cycle sort
def cycle_sort(arr):
    n = len(arr)
    for cycle_start in range(0, n - 1):
        item = arr[cycle_start]
        pos = cycle_start
        for i in range(cycle_start + 1, n):
            if arr[i] < item:
                pos += 1
        if pos == cycle_start:
            continue
        while item == arr[pos]:
            pos += 1
        arr[pos], item = item, arr[pos]
        while pos != cycle_start:
            pos = cycle_start
            for i in range(cycle_start + 1, n):
                if arr[i] < item:
                    pos += 1
            while item == arr[pos]:
                pos += 1
            arr[pos], item = item, arr[pos]

#7 Gnome sort
def gnome_sort(arr):
    n = len(arr)
    index = 0
    while index < n:
        if index == 0 or arr[index] >= arr[index - 1]:
            index += 1
        else:
            arr[index], arr[index - 1] = arr[index - 1], arr[index]
            index -= 1
            
#8 Counting sort
def counting_sort(arr):
    N = len(arr)
    if N == 0:
        return
    M = max(arr)
    countArray = [0] * (M + 1)
    for i in range(N):
        countArray[arr[i]] += 1
    for i in range(1, M + 1):
        countArray[i] += countArray[i - 1]
    outputArray = [0] * N
    for i in range(N - 1, -1, -1):
        outputArray[countArray[arr[i]] - 1] = arr[i]
        countArray[arr[i]] -= 1
    for i in range(N):
        arr[i] = outputArray[i]
        
#9 Radix sort
def radix_sort(arr):
    def count_sort(arr, exp1):
        n = len(arr)
        output = [0] * n
        count = [0] * 10
        for i in range(n):
            index = arr[i] // exp1
            count[index % 10] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]
        i = n - 1
        while i >= 0:
            index = arr[i] // exp1
            output[count[index % 10] - 1] = arr[i]
            count[index % 10] -= 1
            i -= 1
        for i in range(n):
            arr[i] = output[i]

    if len(arr) == 0:
        return
    max1 = max(arr)
    exp = 1
    while max1 // exp > 0:
        count_sort(arr, exp)
        exp *= 10

#10 Bucket sort
def bucket_sort(arr):
    n = len(arr)
    if n <= 0:
        return
    bucket = [[] for _ in range(n)]
    max_value = max(arr) if arr else 0
    for i in range(n):
        # Normalize to [0,1) for bucket index, avoid division by zero
        bucketIndex = int(arr[i] * n / (max_value + 1)) if max_value > 0 else 0
        bucket[bucketIndex].append(arr[i])
    for i in range(n):
        bucket[i].sort()
    index = 0
    for i in range(n):
        for j in bucket[i]:
            arr[index] = j
            index += 1

#11 Heap sort Algorithm
def heap_sort(arr):
    def heapify(A, N, i):
        left = 2 * i + 1
        right = 2 * i + 2
        largest = i
        if left < N and A[left] > A[largest]:
            largest = left
        if right < N and A[right] > A[largest]:
            largest = right
        if largest != i:
            A[i], A[largest] = A[largest], A[i]
            heapify(A, N, largest)

    N = len(arr)
    for i in range(N // 2 - 1, -1, -1):
        heapify(arr, N, i)
    for i in range(N - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)


#12 Merge sort (index-based, matches pseudocode)
def merge_sort(arr):
    def _merge_sort(A, left, right):
        if left < right:
            middle = left + (right - left) // 2
            _merge_sort(A, left, middle)
            _merge_sort(A, middle + 1, right)
            merge(A, left, middle, right)

    def merge(A, left, middle, right):
        n1 = middle - left + 1
        n2 = right - middle
        L = [0] * n1
        R = [0] * n2
        for i in range(n1):
            L[i] = A[left + i]
        for j in range(n2):
            R[j] = A[middle + 1 + j]
        i = 0
        j = 0
        k = left
        while i < n1 and j < n2:
            if L[i] <= R[j]:
                A[k] = L[i]
                i += 1
            else:
                A[k] = R[j]
                j += 1
            k += 1
        while i < n1:
            A[k] = L[i]
            i += 1
            k += 1
        while j < n2:
            A[k] = R[j]
            j += 1
            k += 1

    _merge_sort(arr, 0, len(arr) - 1)


#13 Quick sort (matches pseudocode)
def quick_sort(arr):
    def partition(A, low, high):
        pivot = A[high]
        i = low - 1
        for j in range(low, high):
            if A[j] <= pivot:
                i += 1
                A[i], A[j] = A[j], A[i]
        A[i + 1], A[high] = A[high], A[i + 1]
        return i + 1

    def _quick_sort(A, low, high):
        if low < high:
            pi = partition(A, low, high)
            _quick_sort(A, low, pi - 1)
            _quick_sort(A, pi + 1, high)

    _quick_sort(arr, 0, len(arr) - 1)


#14 Timsort (Python built-in sort)
def tim_sort(arr):
    arr.sort()

#15 Tree Sort
def tree_sort(arr):
    class Node:
        def __init__(self, key):
            self.left = None
            self.right = None
            self.val = key
    def insert(root, key):
        if root is None:
            return Node(key)
        if key < root.val:
            root.left = insert(root.left, key)
        else:
            root.right = insert(root.right, key)
        return root
    def inorder(root, res):
        if root:
            inorder(root.left, res)
            res.append(root.val)
            inorder(root.right, res)
    root = None
    for key in arr:
        root = insert(root, key)
    res = []
    inorder(root, res)
    for i in range(len(arr)):
        arr[i] = res[i]

#16 Pancake Sort
def pancake_sort(arr):
    def flip(sub, k):
        sub[:k+1] = sub[:k+1][::-1]
    n = len(arr)
    for curr_size in range(n, 1, -1):
        mi = arr.index(max(arr[:curr_size]))
        if mi != curr_size - 1:
            flip(arr, mi)
            flip(arr, curr_size - 1)

#17 Flash Sort
def flash_sort(arr):
    n = len(arr)
    if n == 0:
        return
    m = int(0.43 * n) if n > 1 else 1
    min_val = min(arr)
    max_val = max(arr)
    if min_val == max_val:
        return
    l = [0] * m
    c1 = (m - 1) / (max_val - min_val)
    for i in range(n):
        k = int(c1 * (arr[i] - min_val))
        l[k] += 1
    for i in range(1, m):
        l[i] += l[i - 1]
    hold = arr[0]
    move = 0
    j = 0
    k = m - 1
    while move < n:
        while j > l[k] - 1:
            j += 1
            k = int(c1 * (arr[j] - min_val))
        ev = arr[j]
        while j != l[k]:
            k = int(c1 * (ev - min_val))
            l[k] -= 1
            arr[l[k]], ev = ev, arr[l[k]]
            move += 1
    arr.sort()

#18 Tournament Sort
def tournament_sort(arr):
    import heapq
    n = len(arr)
    if n == 0:
        return
    heap = [(val, i) for i, val in enumerate(arr)]
    heapq.heapify(heap)
    res = []
    while heap:
        res.append(heapq.heappop(heap)[0])
    for i in range(n):
        arr[i] = res[i]

#19 Strand Sort
def strand_sort(arr):
    def merge(a, b):
        result = []
        while a and b:
            if a[0] < b[0]:
                result.append(a.pop(0))
            else:
                result.append(b.pop(0))
        result.extend(a or b)
        return result
    output = []
    while arr:
        i, sub = 0, [arr.pop(0)]
        while i < len(arr):
            if arr[i] > sub[-1]:
                sub.append(arr.pop(i))
            else:
                i += 1
        output = merge(sub, output)
    arr.extend(output)
    for i in range(len(arr) - len(output)):
        arr.pop(0)


def load_dataset(file_path):
    with open(file_path) as f:
        return [int(line.strip()) for line in f]

def measure_time(sort_func, arr, runs=10):
    times = []
    # Do correctness check only once (not in every run) to avoid overhead
    arr_test = arr.copy()
    sort_func(arr_test)
    expected_sorted = sorted(arr)
    assert arr_test == expected_sorted, f"Sort failed for {sort_func.__name__}"
    
    # Now time the actual runs without correctness check overhead
    for _ in range(runs):
        arr_copy = arr.copy()
        start = time.perf_counter_ns()
        sort_func(arr_copy)
        end = time.perf_counter_ns()
        times.append((end - start) / 1e9)  # Convert ns to seconds
    return statistics.median(times)

def main():
    dataset_dir = 'datasets'
    base_result_file = 'algorithm_timings'
    algorithms = [
        ('Insertion Sort', insertion_sort, 'O(n) (best) O(n²) (avg, worst)', 'O(1)', 'Stable', 'Yes', 'O(n) to O(n²)', 'O(n) to O(n²)'),
        ('Selection Sort', selection_sort, 'O(n²) (best, avg, worst)', 'O(1)', 'Unstable', 'No', 'O(n²)', 'O(n)'),
        ('Bubble Sort', bubble_sort, 'O(n) (best) O(n²) (avg, worst)', 'O(1)', 'Stable', 'Yes', 'O(n) to O(n²)', 'O(n²)'),
        ('Merge Sort', merge_sort, 'O(n log n) (best, avg, worst)', 'O(n)', 'Stable', 'No', 'O(n log n)', 'O(n log n)'),
        ('Quick Sort', quick_sort, 'O(n log n) (avg, best) O(n²) (worst)', 'O(log n)', 'Unstable', 'No', 'O(n log n)', 'O(n log n)'),
        ('Heap Sort', heap_sort, 'O(n log n) (best, avg, worst)', 'O(1)', 'Unstable', 'No', 'O(n log n)', 'O(n log n)'),
        ('Shell Sort', shell_sort, 'O(n log² n) (avg) O(n²) (worst)', 'O(1)', 'Unstable', 'No', 'Depends', 'Depends'),
        ('Counting Sort', counting_sort, 'O(n + k) (avg, best, worst)', 'O(k)', 'Stable', 'No', 'O(n + k)', 'O(n + k)'),
        ('Radix Sort', radix_sort, 'O(nk) (avg, best, worst)', 'O(n + k)', 'Stable', 'No', 'O(nk)', 'O(n + k)'),
        ('Bucket Sort', bucket_sort, 'O(n + k) (avg) O(n²) (worst)', 'O(n + k)', 'Stable', 'No', 'O(n²)', 'O(n + k)'),
        ('Cocktail-Shaker Sort', cocktail_shaker_sort, 'O(n) (best) O(n²) (avg, worst)', 'O(1)', 'Stable', 'Yes', 'O(n) to O(n²)', 'O(n²)'),
        ('Gnome Sort', gnome_sort, 'O(n) (best) O(n²) (avg, worst)', 'O(1)', 'Stable', 'Yes', 'O(n) to O(n²)', 'O(n²)'),
        ('Cycle Sort', cycle_sort, 'O(n²) (best, avg, worst)', 'O(1)', 'Stable', 'No', 'O(n²)', 'O(n²)'),
        ('TimSort (Python built-in)', tim_sort, 'O(n log n) (best, avg, worst)', 'O(n)', 'Stable', 'Yes', 'O(n log n)', 'O(n log n)'),
        ('Tree Sort', tree_sort, 'O(n log n) (avg, best) O(n²) (worst)', 'O(n)', 'Stable', 'No', 'O(n log n)', 'O(n)'),
        ('Pancake Sort', pancake_sort, 'O(n²)', 'O(1)', 'Unstable', 'No', 'O(n²)', 'O(n²)'),
        ('Flash Sort', flash_sort, 'O(n) (best) O(n log n) (avg, worst)', 'O(n)', 'Unstable', 'No', 'O(n)', 'O(n)'),
        ('Tournament Sort', tournament_sort, 'O(n log n)', 'O(n)', 'Stable', 'No', 'O(n log n)', 'O(n log n)'),
        ('Strand Sort', strand_sort, 'O(n²)', 'O(n)', 'Stable', 'No', 'O(n²)', 'O(n²)')
    ]
    sizes = [100, 1000, 10000, 100000]
    
    # RESUME MODE: Skip header initialization to preserve existing results
    # Comment out the next 5 lines if resuming from a previous run
    # Uncomment them for a fresh start
    # for suffix in ['random', 'nearly_sorted', 'reverse_sorted', 'few_unique']:
    #     result_file = f"{base_result_file}_{suffix}.csv"
    #     with open(result_file, 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(['Algorithm', 'CPU Time (s)', 'Dataset'])
    
    # Start from specific size and dataset type to resume
    # Modify these to skip already completed work:
    start_size = 100000  # Set to the size where you want to resume
    start_suffix = 'few_unique'  # Set to the dataset type where you want to resume
    start_algorithm_index = 15  # 15=Pancake Sort (currently running)
    
    # Algorithm index reference:
    # 0=Insertion, 1=Selection, 2=Bubble, 3=Merge, 4=Quick, 5=Heap, 6=Shell,
    # 7=Counting, 8=Radix, 9=Bucket, 10=Cocktail-Shaker, 11=Gnome, 12=Cycle,
    # 13=TimSort, 14=Tree, 15=Pancake, 16=Flash, 17=Tournament, 18=Strand
    
    resume_mode = True  # Set to False for fresh start
    
    for size in sizes:
        if resume_mode and size < start_size:
            continue  # Skip completed sizes
            
        for suffix in ['random', 'nearly_sorted', 'reverse_sorted', 'few_unique']:
            if resume_mode and size == start_size:
                # Skip completed dataset types for the current size
                if suffix != start_suffix:
                    suffix_order = ['random', 'nearly_sorted', 'reverse_sorted', 'few_unique']
                    if suffix_order.index(suffix) < suffix_order.index(start_suffix):
                        continue
            
            result_file = f"{base_result_file}_{suffix}.csv"
            dataset_filename = f"dataset_{size}_{suffix}.txt"
            dataset_path = os.path.join(dataset_dir, dataset_filename)
            
            # Check if dataset file exists
            if not os.path.exists(dataset_path):
                print(f"Warning: Dataset file {dataset_filename} not found. Skipping.")
                continue
            
            # Load dataset once for all algorithms
            print(f"\nLoading dataset: {dataset_filename}")
            arr = load_dataset(dataset_path)
            
            # Batch write results for this dataset to minimize file I/O
            results = []
            for idx, (name, func, *rest) in enumerate(algorithms):
                # Skip algorithms that were already completed
                if resume_mode and size == start_size and suffix == start_suffix and idx < start_algorithm_index:
                    print(f"Skipping {name} on {dataset_filename} (already completed)")
                    continue
                
                # Turn off resume mode after processing the first algorithm in resume position
                if resume_mode and idx >= start_algorithm_index:
                    resume_mode = False
                
                print(f"Running {name} on {dataset_filename}...")
                try:
                    t = measure_time(func, arr, runs=10)
                    print(f"{name} completed: {t} seconds")
                except Exception as e:
                    print(f"Error running {name} on {dataset_filename}: {e}")
                    t = 'Error'
                results.append([name, t, dataset_filename.replace('dataset_', '', 1)])
            
            # Write all results for this dataset at once
            with open(result_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(results)
                
        print(f"\n{'='*60}")
        print(f"Completed all dataset types for size {size}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
