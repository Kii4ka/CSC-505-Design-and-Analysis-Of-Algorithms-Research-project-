import random
import os

def generate_dataset(size, file_path):
    """Generate a dataset of positive integers and save to a file."""
    with open(file_path, 'w') as f:
        for _ in range(size):
            f.write(f"{random.randint(1, 10**6)}\n")

def generate_nearly_sorted(size, file_path, shuffle_percent=0.05):
    arr = list(range(1, size + 1))
    num_shuffle = int(size * shuffle_percent)
    for _ in range(num_shuffle):
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)
        arr[i], arr[j] = arr[j], arr[i]
    with open(file_path, 'w') as f:
        for num in arr:
            f.write(f"{num}\n")

def generate_reverse_sorted(size, file_path):
    arr = list(range(size, 0, -1))
    with open(file_path, 'w') as f:
        for num in arr:
            f.write(f"{num}\n")

def generate_few_unique(size, file_path, unique_count=32):
    arr = [random.randint(1, unique_count) for _ in range(size)]
    with open(file_path, 'w') as f:
        for num in arr:
            f.write(f"{num}\n")

def main():
    # Set seed for reproducibility
    random.seed(42)
    
    sizes = [100, 1000, 10000, 100000]
    dataset_dir = 'datasets'
    os.makedirs(dataset_dir, exist_ok=True)
    for size in sizes:
        # Random
        file_path = os.path.join(dataset_dir, f"dataset_{size}_random.txt")
        print(f"Generating random {size} integers in {file_path}...")
        generate_dataset(size, file_path)
        # Nearly sorted
        file_path = os.path.join(dataset_dir, f"dataset_{size}_nearly_sorted.txt")
        print(f"Generating nearly sorted {size} integers in {file_path}...")
        generate_nearly_sorted(size, file_path)
        # Reverse sorted
        file_path = os.path.join(dataset_dir, f"dataset_{size}_reverse_sorted.txt")
        print(f"Generating reverse sorted {size} integers in {file_path}...")
        generate_reverse_sorted(size, file_path)
        # Few unique
        file_path = os.path.join(dataset_dir, f"dataset_{size}_few_unique.txt")
        print(f"Generating few-unique {size} integers in {file_path}...")
        generate_few_unique(size, file_path)
    print("All datasets generated.")

if __name__ == "__main__":
    main()
