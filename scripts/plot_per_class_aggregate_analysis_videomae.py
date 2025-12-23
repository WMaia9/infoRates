import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load per-class results
csv_path = "data/UCF101_data/results/videomae/fine_tuned_videomae_ucf101_per_class_testset.csv"
df = pd.read_csv(csv_path)

# Aggregate duplicates by taking mean accuracy
df_agg = df.groupby(['class', 'coverage', 'stride'], as_index=False)['accuracy'].mean()

# Compute mean accuracy per class across all configurations
mean_per_class = df_agg.groupby('class')['accuracy'].mean().sort_values(ascending=False)

# Plot top 20 classes
plt.figure(figsize=(12, 8))
mean_per_class.head(20).plot(kind='bar', color='skyblue')
plt.title('Top 20 Classes by Mean Accuracy Across All Configurations (VideoMAE)', fontsize=16)
plt.ylabel('Mean Accuracy', fontsize=14)
plt.xlabel('Class', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("data/UCF101_data/results/videomae/per_class_aggregate_analysis.png", dpi=300)
plt.close()
print("Saved: data/UCF101_data/results/videomae/per_class_aggregate_analysis.png")