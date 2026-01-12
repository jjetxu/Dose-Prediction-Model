import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV file."""
    df = pd.read_csv(filepath)
    return df


def generate_statistics(df: pd.DataFrame) -> dict:
    """Generate descriptive statistics for the dataset."""
    stats = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "data_types": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
        "descriptive_stats": df.describe().to_dict(),
        "categorical_summary": {}
    }
    
    # Add categorical summaries
    categorical_cols = df.select_dtypes(include=['object', 'int']).columns
    for col in categorical_cols:
        if df[col].nunique() <= 10:
            stats["categorical_summary"][col] = df[col].value_counts().to_dict()
    
    return stats


def print_statistics(stats: dict) -> None:
    """Print statistics in a readable format."""
    print("=" * 80)
    print("DATA PIPELINE STATISTICS")
    print("=" * 80)
    
    print(f"\nDataset Shape: {stats['shape'][0]} rows × {stats['shape'][1]} columns")
    
    print("\nData Types:")
    for col, dtype in stats['data_types'].items():
        print(f"  {col}: {dtype}")
    
    print("\nMissing Values:")
    for col, count in stats['missing_values'].items():
        pct = stats['missing_percentage'][col]
        print(f"  {col}: {count} ({pct:.2f}%)")
    
    print("\nDescriptive Statistics:")
    for col, col_stats in stats['descriptive_stats'].items():
        print(f"\n  {col}:")
        for stat_name, value in col_stats.items():
            if isinstance(value, float):
                print(f"    {stat_name}: {value:.4f}")
            else:
                print(f"    {stat_name}: {value}")
    
    print("\nCategorical Summaries:")
    for col, value_counts in stats['categorical_summary'].items():
        print(f"  {col}:")
        for value, count in sorted(value_counts.items()):
            print(f"    {value}: {count}")
    
    print("\n" + "=" * 80)


def create_visualizations(df: pd.DataFrame, output_dir: str = "io/plots") -> None:
    """Create and save various plots for data exploration."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Distribution of numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    fig, axes = plt.subplots(len(numerical_cols), 1, figsize=(10, 3*len(numerical_cols)))
    if len(numerical_cols) == 1:
        axes = [axes]
    
    for idx, col in enumerate(numerical_cols):
        axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_distributions.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/01_distributions.png")
    
    # 2. Boxplots for numerical features
    fig, axes = plt.subplots(1, len(numerical_cols), figsize=(4*len(numerical_cols), 5))
    if len(numerical_cols) == 1:
        axes = [axes]
    
    for idx, col in enumerate(numerical_cols):
        axes[idx].boxplot(df[col].dropna())
        axes[idx].set_title(f'Boxplot of {col}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(col)
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_boxplots.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/02_boxplots.png")
    
    # 3. Correlation heatmap
    if len(numerical_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/03_correlation_heatmap.png", dpi=100, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_dir}/03_correlation_heatmap.png")
    
    # 4. Pairplot (if dataset is not too large)
    if len(df) <= 5000 and len(numerical_cols) <= 6:
        fig = plt.figure(figsize=(12, 10))
        pd.plotting.scatter_matrix(df[numerical_cols], alpha=0.6, figsize=(12, 10))
        plt.suptitle('Pairplot of Numerical Features', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/04_pairplot.png", dpi=100, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_dir}/04_pairplot.png")
    
    # 5. Missing data visualization
    if df.isnull().sum().sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        missing_data.plot(kind='barh', ax=ax, color='coral')
        ax.set_title('Missing Values by Column', fontsize=12, fontweight='bold')
        ax.set_xlabel('Count of Missing Values')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/05_missing_data.png", dpi=100, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_dir}/05_missing_data.png")


def main():
    # Define paths
    data_path = "io/anaesthetic_data.csv"
    
    # Load data
    print("Loading data...")
    df = load_data(data_path)
    
    # Generate statistics
    print("Generating statistics...")
    stats = generate_statistics(df)
    
    # Print statistics
    print_statistics(stats)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df)
    
    print("\n✓ Data pipeline completed successfully!")


if __name__ == "__main__":
    main()
