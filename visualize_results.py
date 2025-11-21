import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_data():
    """Load all CSV files and combine them."""
    dataset_types = ['random', 'nearly_sorted', 'reverse_sorted', 'few_unique']
    all_data = []
    
    for dtype in dataset_types:
        csv_file = f'algorithm_timings_{dtype}.csv'
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df['Dataset Type'] = dtype.replace('_', ' ').title()
            all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Extract size from dataset filename
    combined_df['Size'] = combined_df['Dataset'].str.extract(r'(\d+)').astype(int)
    
    # Clean up dataset column
    combined_df['Dataset'] = combined_df['Dataset'].str.replace(r'^\d+_', '', regex=True)
    combined_df['Dataset'] = combined_df['Dataset'].str.replace('.txt', '')
    
    # Remove "Sort" from algorithm names
    combined_df['Algorithm'] = combined_df['Algorithm'].str.replace(' Sort', '', regex=False)
    # Remove "(Python built-in)" from algorithm names
    combined_df['Algorithm'] = combined_df['Algorithm'].str.replace(' (Python built-in)', '', regex=False)
    
    # Convert time to numeric, handle 'Error' and 'Skipped'
    combined_df['CPU Time (s)'] = pd.to_numeric(combined_df['CPU Time (s)'], errors='coerce')
    
    return combined_df

def plot_time_vs_algorithms_by_size(df, output_dir='plots'):
    """Create one bar graph per size showing time vs algorithms for all dataset types."""
    os.makedirs(output_dir, exist_ok=True)
    
    sizes = [100, 1000, 10000, 100000]
    dataset_types = ['Random', 'Nearly Sorted', 'Reverse Sorted', 'Few Unique']
    
    for size in sizes:
        fig, axes = plt.subplots(4, 1, figsize=(14, 20))
        
        for idx, (ax, dtype) in enumerate(zip(axes.flat, dataset_types)):
            subset = df[(df['Size'] == size) & (df['Dataset Type'] == dtype)].dropna(subset=['CPU Time (s)'])
            
            if not subset.empty:
                subset = subset.sort_values('CPU Time (s)')
                algorithms = subset['Algorithm'].values
                times = subset['CPU Time (s)'].values
                
                # Create colorful bars
                colors = sns.color_palette("husl", len(algorithms))
                bars = ax.bar(range(len(algorithms)), times, color=colors)
                
                ax.set_ylabel('Time (seconds)', fontsize=16, fontweight='bold')
                ax.set_title(f'{dtype} Dataset', fontsize=18, fontweight='bold')
                ax.set_xticks(range(len(algorithms)))
                ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=13, fontweight='bold')
                ax.tick_params(axis='y', labelsize=14)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                
                # Add value labels on bars
                for i, (bar, time) in enumerate(zip(bars, times)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{time:.4f}s',
                           ha='center', va='bottom', fontsize=11, fontweight='bold', rotation=0)
        
        plt.tight_layout()
        filename = f'{output_dir}/algorithms_size_{size}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: algorithms_size_{size}.png")

def plot_heatmap_table(df, output_dir='plots'):
    """Create heatmap table showing algorithm times across all sizes and dataset types."""
    os.makedirs(output_dir, exist_ok=True)
    
    import numpy as np
    
    dataset_types = ['Random', 'Nearly Sorted', 'Reverse Sorted', 'Few Unique']
    
    for dtype in dataset_types:
        subset = df[df['Dataset Type'] == dtype].dropna(subset=['CPU Time (s)'])
        
        # Pivot table: algorithms (rows) vs sizes (columns)
        pivot = subset.pivot_table(values='CPU Time (s)', 
                                   index='Algorithm', 
                                   columns='Size', 
                                   aggfunc='mean')
        
        # Ensure columns are in correct order
        pivot = pivot[[100, 1000, 10000, 100000]]
        
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(6, 16))
            
            # Normalize by column (each size independently) to show relative ranking
            # This allows us to see if algorithms maintain their competitive position across sizes
            pivot_normalized = pivot.copy()
            for col in pivot.columns:
                col_min = pivot[col].min()
                col_max = pivot[col].max()
                if col_max > col_min:  # Avoid division by zero
                    pivot_normalized[col] = (pivot[col] - col_min) / (col_max - col_min)
                else:
                    pivot_normalized[col] = 0
            
            # Create heatmap with column-normalized colors but original values as annotations
            sns.heatmap(pivot_normalized, annot=pivot.values, fmt='.6f', cmap='YlOrRd', 
                       linewidths=2, linecolor='black', ax=ax, 
                       cbar_kws={'label': 'Relative Performance (per column)'},
                       annot_kws={'fontsize': 10, 'weight': 'bold'},
                       square=True, cbar=True, vmin=0, vmax=1)
            
            ax.set_xlabel('Input Size (n)', fontweight='bold', fontsize=14)
            ax.set_ylabel('', fontweight='bold', fontsize=14)
            
            # Format column labels - make them bold and bigger, compact
            ax.set_xticklabels(['100', '1K', '10K', '100K'], 
                              rotation=0, fontsize=11, fontweight='bold')
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=13, fontweight='bold', rotation=0)
            
            plt.tight_layout()
            filename = f'{output_dir}/heatmap_table_{dtype.replace(" ", "_").lower()}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: heatmap_table_{dtype.replace(' ', '_').lower()}.png")

def main():
    print("Loading data...")
    df = load_data()
    
    if df.empty:
        print("Error: No data found. Make sure CSV files exist in the current directory.")
        return
    
    print(f"Loaded {len(df)} records")
    print(f"Algorithms: {df['Algorithm'].nunique()}")
    print(f"Dataset types: {df['Dataset Type'].nunique()}")
    print(f"Sizes: {sorted(df['Size'].unique())}")
    print("\nGenerating visualizations...")
    
    # Generate the time vs algorithms plots (one per size)
    plot_time_vs_algorithms_by_size(df)
    
    # Generate heatmap tables (one per dataset type)
    plot_heatmap_table(df)
    
    print("\n✓ All visualizations generated successfully!")
    print("Check the 'plots' directory for output files.")

if __name__ == "__main__":
    main()
