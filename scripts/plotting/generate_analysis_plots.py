#!/usr/bin/env python3
"""
Master Plotting Script for Temporal Aliasing Analysis

Generates all necessary plots and statistical analyses for a given model and dataset.
This consolidates multiple individual plotting scripts into one comprehensive analysis.

Usage:
    python scripts/plotting/plot_all.py --model videomae --dataset ucf101
    python scripts/plotting/plot_all.py --model all --dataset kinetics400
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def run_statistical_analysis(csv_path, per_class_csv_path, output_dir):
    """Run comprehensive statistical analysis."""
    print("🔬 Running statistical analysis...")

    import subprocess
    try:
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "statistical_analysis.py"),
            "--csv", str(csv_path),
            "--per-class-csv", str(per_class_csv_path),
            "--out-dir", str(output_dir)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Statistical analysis completed")
        else:
            print(f"⚠️  Statistical analysis failed: {result.stderr}")
    except Exception as e:
        print(f"⚠️  Could not run statistical analysis: {e}")

def generate_distribution_plot(per_class_csv, output_dir, model_name, stride=8):
    """Generate per-class accuracy distribution plots."""
    print("📊 Generating distribution plots...")

    df = pd.read_csv(per_class_csv)

    # Filter by stride
    df = df[df['stride'] == stride]

    # Prepare data
    coverages = sorted(df['coverage'].unique())
    class_list = sorted(df['class'].unique())

    acc_matrix = np.zeros((len(class_list), len(coverages)))
    for i, cls in enumerate(class_list):
        for j, cov in enumerate(coverages):
            acc = df[(df['class'] == cls) & (df['coverage'] == cov)]['accuracy']
            acc_matrix[i, j] = acc.values[0] if not acc.empty else np.nan

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Boxplot
    sns.boxplot(data=pd.DataFrame(acc_matrix, columns=coverages), ax=axes[0])
    axes[0].set_xlabel('Frame Coverage (%)', fontsize=14)
    axes[0].set_ylabel('Per-Class Accuracy', fontsize=14)
    axes[0].set_title('Boxplot: Per-Class Accuracy by Coverage', fontsize=16)
    axes[0].set_xticklabels([str(int(c)) for c in coverages])
    axes[0].grid(True, linestyle=':', alpha=0.5)

    # Violin plot
    sns.violinplot(data=pd.DataFrame(acc_matrix, columns=coverages), ax=axes[1], inner='quartile')
    axes[1].set_xlabel('Frame Coverage (%)', fontsize=14)
    axes[1].set_title('Violin: Per-Class Accuracy by Coverage', fontsize=16)
    axes[1].set_xticklabels([str(int(c)) for c in coverages])
    axes[1].grid(True, linestyle=':', alpha=0.5)

    plt.suptitle(f'Distribution of Per-Class Accuracies at Stride-{stride} Across Coverage Levels ({model_name.capitalize()})', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = output_dir / f"per_class_distribution_by_coverage.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

def generate_representative_plot(per_class_csv, output_dir, model_name):
    """Generate representative classes sensitivity analysis."""
    print("🎯 Generating representative classes plot...")

    df = pd.read_csv(per_class_csv)

    # Find best stride
    stride_performance = df.groupby('stride')['accuracy'].mean()
    best_stride = stride_performance.idxmax()
    print(f"Best stride: {best_stride} (mean accuracy: {stride_performance[best_stride]:.4f})")

    # Use best stride
    df = df[df['stride'] == best_stride]

    # Aggregate
    df = df.groupby(['class', 'coverage'], as_index=False)['accuracy'].mean()
    pivot = df.pivot(index='class', columns='coverage', values='accuracy')

    coverages = sorted(df['coverage'].unique())

    # Compute sensitive/robust classes
    drop_100_to_25 = pivot[100] - pivot[25]
    sensitive_classes = drop_100_to_25.sort_values(ascending=False).head(5).index.tolist()
    variance = pivot.var(axis=1)
    robust_classes = variance.sort_values().head(5).index.tolist()

    print(f"Most sensitive classes: {sensitive_classes}")
    print(f"Most robust classes: {robust_classes}")

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, 10))
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

    for i, cls in enumerate(sensitive_classes + robust_classes):
        if cls in sensitive_classes:
            label = f"{cls} (sensitive)"
            color = colors[0]
            style = '--'
        else:
            label = f"{cls} (robust)"
            color = colors[1]
            style = '-'

        accuracies = [pivot.loc[cls, cov] for cov in coverages]
        ax.plot(coverages, accuracies, label=label, color=color, linestyle=style, linewidth=2, marker='o')

    ax.set_xlabel('Frame Coverage (%)', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title(f'Representative Classes: Sensitivity Analysis ({model_name.capitalize()})', fontsize=16, fontweight='bold')
    ax.set_xticks(coverages)
    ax.set_xticklabels([f"{int(c)}%" for c in coverages])
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    output_path = output_dir / "per_class_representative.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

def generate_heatmap_plot(temporal_csv, output_dir, model_name):
    """Generate accuracy heatmap across coverage and stride."""
    print("🔥 Generating accuracy heatmap...")

    df = pd.read_csv(temporal_csv)

    # Pivot to create heatmap
    heatmap_data = df.pivot_table(values='accuracy', index='coverage', columns='stride', aggfunc='mean')

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Accuracy'})

    plt.title(f'Accuracy Heatmap: Coverage vs Stride ({model_name.capitalize()})', fontsize=16, fontweight='bold')
    plt.xlabel('Stride', fontsize=14)
    plt.ylabel('Coverage (%)', fontsize=14)

    output_path = output_dir / "accuracy_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

def generate_accuracy_curves(temporal_csv, output_dir, model_name):
    """Generate accuracy vs coverage curves for different strides."""
    print("📈 Generating accuracy curves...")

    df = pd.read_csv(temporal_csv)

    plt.figure(figsize=(12, 8))

    strides = sorted(df['stride'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(strides)))

    for i, stride in enumerate(strides):
        stride_data = df[df['stride'] == stride].sort_values('coverage')
        plt.plot(stride_data['coverage'], stride_data['accuracy'],
                label=f'Stride {stride}', color=colors[i], linewidth=2, marker='o')

    plt.xlabel('Frame Coverage (%)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title(f'Accuracy vs Coverage by Stride ({model_name.capitalize()})', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = output_dir / "accuracy_vs_coverage.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate all analysis plots for a model and dataset")
    parser.add_argument('--model', required=True, help='Model name (videomae, vivit, timesformer, or all)')
    parser.add_argument('--dataset', required=True, choices=['ucf101', 'kinetics400'], help='Dataset name')
    parser.add_argument('--stride', type=int, default=8, help='Stride for distribution analysis (default: 8)')

    args = parser.parse_args()

    # Set up paths
    base_dir = Path(__file__).parent.parent.parent
    evaluations_dir = base_dir / "evaluations" / args.dataset

    if args.model == 'all':
        models = ['videomae', 'vivit', 'timesformer']
    else:
        models = [args.model]

    for model in models:
        print(f"\n{'='*60}")
        print(f"🎨 Generating plots for {model.upper()} on {args.dataset.upper()}")
        print(f"{'='*60}")

        model_dir = evaluations_dir / model
        if not model_dir.exists():
            print(f"⚠️  No results found for {model} on {args.dataset}")
            continue

        # Find CSV files
        if args.dataset == 'kinetics400':
            per_class_csv = model_dir / f"{model}-base-finetuned-kinetics_per_class.csv"
            temporal_csv = model_dir / f"{model}-base-finetuned-kinetics_temporal_sampling.csv"
        else:  # ucf101
            per_class_csv = model_dir / f"fine_tuned_{model}_{args.dataset}_per_class_testset.csv"
            temporal_csv = model_dir / f"fine_tuned_{model}_{args.dataset}_temporal_sampling.csv"

        if not per_class_csv.exists():
            print(f"❌ Per-class CSV not found: {per_class_csv}")
            continue

        if not temporal_csv.exists():
            print(f"❌ Temporal CSV not found: {temporal_csv}")
            continue

        # Generate all plots
        try:
            generate_distribution_plot(per_class_csv, model_dir, model, args.stride)
            generate_representative_plot(per_class_csv, model_dir, model)
            generate_heatmap_plot(temporal_csv, model_dir, model)
            generate_accuracy_curves(temporal_csv, model_dir, model)

            # Run statistical analysis
            run_statistical_analysis(str(temporal_csv), str(per_class_csv), model_dir)

            print(f"✅ All plots generated for {model}")

        except Exception as e:
            print(f"❌ Error generating plots for {model}: {e}")

    print(f"\n🎉 Plot generation complete!")

if __name__ == "__main__":
    main()