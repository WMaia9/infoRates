import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load per-class results
csv_path = "data/UCF101_data/results/videomae/fine_tuned_videomae_ucf101_per_class_testset.csv"
df = pd.read_csv(csv_path)

# Aggregate duplicates
df_agg = df.groupby(['class', 'coverage', 'stride'], as_index=False)['accuracy'].mean()

# Use stride=8
df_stride8 = df_agg[df_agg['stride'] == 8]

# Pivot
pivot = df_stride8.pivot(index='class', columns='coverage', values='accuracy')

# Compute drop from 100% to 25%
if 100 in pivot.columns and 25 in pivot.columns:
    drop = pivot[100] - pivot[25]
    
    # Categorize into tiers
    quantiles = drop.quantile([0.33, 0.67])
    low_sens = drop[drop <= quantiles[0.33]]
    med_sens = drop[(drop > quantiles[0.33]) & (drop <= quantiles[0.67])]
    high_sens = drop[drop > quantiles[0.67]]
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    low_sens.head(10).plot(kind='bar', ax=axes[0], color='green', title='Low Sensitivity (Top 10)')
    med_sens.head(10).plot(kind='bar', ax=axes[1], color='orange', title='Medium Sensitivity (Top 10)')
    high_sens.head(10).plot(kind='bar', ax=axes[2], color='red', title='High Sensitivity (Top 10)')
    
    for ax in axes:
        ax.set_ylabel('Accuracy Drop (100% → 25%)')
        ax.set_xlabel('Class')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Per-Class Sensitivity Tiers to Aliasing (VideoMAE)', fontsize=16)
    plt.tight_layout()
    plt.savefig("data/UCF101_data/results/videomae/per_class_sensitivity_tiers.png", dpi=300)
    plt.close()
    print("Saved: data/UCF101_data/results/videomae/per_class_sensitivity_tiers.png")
else:
    print("Missing coverage levels for sensitivity analysis")