#!/usr/bin/env python3
"""Build deterministic scenario-prefix folds balanced by frame count."""
from __future__ import annotations
import argparse, csv, re
from collections import defaultdict
from pathlib import Path

def image_count(path: Path) -> int:
    return sum((p.is_file() and p.suffix.lower() in {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'} for p in path.iterdir()))

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('image_root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fold-count', type=int, default=5)
    a = p.parse_args(argv)
    groups = defaultdict(list)
    for seq in sorted((p for p in a.image_root.iterdir() if p.is_dir())):
        match = re.fullmatch('(.+)_\\d+', seq.name)
        prefix = match.group(1) if match else seq.name
        groups[prefix].append((seq.name, image_count(seq)))
    loads = [0] * a.fold_count
    rows = []
    for prefix_index, (prefix, items) in enumerate(sorted(groups.items())):
        used = set()
        for item_index, (name, frames) in enumerate(sorted(items, key=lambda row: (-row[1], row[0]))):
            available = [i for i in range(a.fold_count) if i not in used] or list(range(a.fold_count))
            rotation = (prefix_index + item_index) % a.fold_count
            fold = min(available, key=lambda i: (loads[i], (i - rotation) % a.fold_count))
            rows.append({'sequence': name, 'fold': fold, 'scenario_prefix': prefix, 'frame_count': frames})
            loads[fold] += frames
            used.add(fold)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sequence', 'fold', 'scenario_prefix', 'frame_count'])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r['sequence']))
    print({'sequence_count': len(rows), 'fold_loads': loads})
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
