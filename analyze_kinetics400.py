#!/usr/bin/env python3
"""
Kinetics400 Dataset Video Analysis
Analyzes video properties like FPS, duration, resolution, etc.
"""

import os
import json
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

def get_video_info(video_path):
    """Get video information using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)

        # Extract video stream info
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break

        if not video_stream:
            return None

        # Extract properties
        width = video_stream.get('width', 0)
        height = video_stream.get('height', 0)
        duration = float(data.get('format', {}).get('duration', 0))

        # FPS calculation
        fps_parts = video_stream.get('r_frame_rate', '0/1').split('/')
        if len(fps_parts) == 2 and fps_parts[1] != '0':
            fps = float(fps_parts[0]) / float(fps_parts[1])
        else:
            fps = 0

        # File size
        size_bytes = int(data.get('format', {}).get('size', 0))

        return {
            'filename': os.path.basename(video_path),
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'resolution': f"{width}x{height}",
            'size_bytes': size_bytes,
            'size_mb': size_bytes / (1024 * 1024)
        }

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

def analyze_kinetics400_videos(video_dir, sample_size=1000):
    """Analyze a sample of Kinetics400 videos."""
    video_dir = Path(video_dir)
    video_files = list(video_dir.glob('*.mp4'))

    print(f"Found {len(video_files)} videos in {video_dir}")
    print(f"Analyzing sample of {min(sample_size, len(video_files))} videos...")

    # Sample videos (take every Nth video to get good distribution)
    step = max(1, len(video_files) // sample_size)
    sample_files = video_files[::step][:sample_size]

    results = []
    for video_path in tqdm(sample_files, desc="Analyzing videos"):
        info = get_video_info(str(video_path))
        if info:
            results.append(info)

    return pd.DataFrame(results)

def main():
    video_dir = "data/Kinetics400_data/k4testset/videos_val"
    output_file = "data/Kinetics400_data/kinetics400_video_analysis.csv"

    print("🎬 Kinetics400 Video Analysis")
    print("=" * 50)

    # Analyze videos
    df = analyze_kinetics400_videos(video_dir, sample_size=1000)

    if df.empty:
        print("❌ No video data collected!")
        return

    # Save raw data
    df.to_csv(output_file, index=False)
    print(f"✅ Saved detailed results to {output_file}")

    # Calculate statistics
    print("\n📊 VIDEO STATISTICS (Sample of 1,000 videos)")
    print("=" * 50)

    print("\n🎬 Duration (seconds):")
    print(f"  Mean:   {df['duration'].mean():.2f}")
    print(f"  Median: {df['duration'].median():.2f}")
    print(f"  Min:    {df['duration'].min():.2f}")
    print(f"  Max:    {df['duration'].max():.2f}")
    print(f"  Std:    {df['duration'].std():.2f}")

    print("\n🎥 FPS (frames per second):")
    print(f"  Mean:   {df['fps'].mean():.2f}")
    print(f"  Median: {df['fps'].median():.2f}")
    print(f"  Min:    {df['fps'].min():.2f}")
    print(f"  Max:    {df['fps'].max():.2f}")
    print(f"  Std:    {df['fps'].std():.2f}")

    # FPS distribution
    fps_bins = [0, 15, 20, 25, 30, 35, 40, 100]
    fps_labels = ['<15', '15-20', '20-25', '25-30', '30-35', '35-40', '40+']
    df['fps_range'] = pd.cut(df['fps'], bins=fps_bins, labels=fps_labels, right=False)
    print("\n📈 FPS Distribution:")
    fps_dist = df['fps_range'].value_counts().sort_index()
    for range_name, count in fps_dist.items():
        pct = (count / len(df)) * 100
        print(f"  {range_name}: {count} videos ({pct:.1f}%)")

    print("\n🖼️  Resolution:")
    resolution_counts = df['resolution'].value_counts().head(10)
    for res, count in resolution_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {res}: {count} videos ({pct:.1f}%)")

    print("\n💾 File Size (MB):")
    print(f"  Mean:   {df['size_mb'].mean():.2f}")
    print(f"  Median: {df['size_mb'].median():.2f}")
    print(f"  Min:    {df['size_mb'].min():.2f}")
    print(f"  Max:    {df['size_mb'].max():.2f}")

    # Common resolutions
    print("\n🔍 Most Common Resolutions:")
    res_stats = df.groupby('resolution').agg({
        'duration': ['count', 'mean'],
        'fps': 'mean',
        'size_mb': 'mean'
    }).round(2)
    res_stats.columns = ['count', 'avg_duration', 'avg_fps', 'avg_size_mb']
    res_stats = res_stats.sort_values('count', ascending=False).head(5)

    for res, row in res_stats.iterrows():
        print(f"  {res}: {int(row['count'])} videos, {row['avg_duration']:.1f}s, {row['avg_fps']:.1f}fps, {row['avg_size_mb']:.1f}MB")

    print(f"\n📋 Total videos analyzed: {len(df)}")
    print(f"📄 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()