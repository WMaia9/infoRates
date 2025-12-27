from pathlib import Path
import pandas as pd
from typing import Iterable, Optional


def replace_manifest_paths(old_root: str, new_root: str, manifests: Iterable[Path], column: Optional[str] = "video_path", autodetect: bool = True) -> None:
    """Replace path prefixes in manifest CSV files.

    Parameters
    - old_root: prefix to replace (string match)
    - new_root: replacement prefix
    - manifests: iterable of Path objects pointing to CSV manifests
    - column: column name to target; if not present and autodetect=True, will try to find a candidate column

    Behavior: overwrites each manifest in-place and prints a short status.
    """
    for p in manifests:
        p = Path(p)
        if not p.exists():
            print(f"! Skipped {p}: file not found")
            continue
        df = pd.read_csv(p)
        target_col = None
        if column and column in df.columns:
            target_col = column
        elif autodetect:
            # find first object-type column that contains the old_root in at least one entry
            for col in df.columns:
                if df[col].dtype == object:
                    sample = df[col].astype(str).head(20)
                    if any(old_root in s for s in sample):
                        target_col = col
                        break
        if not target_col:
            print(f"! Skipped {p}: no suitable column found (tried '{column}')")
            continue
        df[target_col] = df[target_col].astype(str).str.replace(old_root, new_root, regex=False)
        df.to_csv(p, index=False)
        print(f"✓ Updated {p} ({old_root} -> {new_root})")
