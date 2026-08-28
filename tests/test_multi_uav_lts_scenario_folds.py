import csv, subprocess, sys
from pathlib import Path
from PIL import Image

def test_fold_builder_keeps_prefix_together(tmp_path: Path):
    root = tmp_path / 'images'
    root.mkdir()
    for name in ('alpha_1', 'alpha_2', 'beta_1'):
        seq = root / name
        seq.mkdir()
        Image.new('L', (4, 4)).save(seq / '0001.png')
    output = tmp_path / 'folds.csv'
    subprocess.run([sys.executable, 'scripts/build_multi_uav_lts_scenario_folds.py', str(root), '--output', str(output), '--fold-count', '2'], check=True)
    rows = list(csv.DictReader(output.open()))
    alpha = {r['fold'] for r in rows if r['scenario_prefix'] == 'alpha'}
    assert len(alpha) == 2
