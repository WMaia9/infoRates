import tempfile
from pathlib import Path
import pandas as pd
from info_rates.data_utils import replace_manifest_paths


def test_replace_manifest_paths_basic(tmp_path):
    csv = tmp_path / "test_manifest.csv"
    df = pd.DataFrame({
        'video_path': [
            '/oldroot/video1.mp4',
            '/oldroot/video2.mp4',
            '/other/path/video3.mp4'
        ],
        'class': ['a','b','c']
    })
    df.to_csv(csv, index=False)

    replace_manifest_paths('/oldroot', '/newroot', [csv])

    df2 = pd.read_csv(csv)
    assert all('/newroot/video' in p or '/other/path' in p for p in df2['video_path'].astype(str))


def test_autodetect_column(tmp_path):
    csv = tmp_path / "test_manifest2.csv"
    df = pd.DataFrame({
        'path': ['/oldroot/1.mp4', '/oldroot/2.mp4'],
        'label': [0,1]
    })
    df.to_csv(csv, index=False)

    # call without specifying column; autodetect should find 'path'
    replace_manifest_paths('/oldroot', '/newroot', [csv])
    df2 = pd.read_csv(csv)
    assert all('/newroot/' in p for p in df2['path'].astype(str))
