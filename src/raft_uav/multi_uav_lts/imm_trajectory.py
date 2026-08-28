"""Robust interacting-multiple-model trajectory and box smoother for LTS tracks."""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
from ._full_stack_io import group_id, prediction_files, read_rows, write_json, write_rows
from ._records import Detection

@dataclass(frozen=True)
class ImmConfig:
    process_noise: float = 2.0
    measurement_noise: float = 4.0
    size_noise: float = 0.08
    student_df: float = 5.0
    transition_stay: float = 0.94
    min_probability: float = 0.0001
_MODEL_NAMES = ('cv', 'ca', 'damped')

def model_matrix(name: str, dt: float) -> np.ndarray:
    f = np.eye(8)
    f[0, 2] = dt
    f[1, 3] = dt
    if name == 'ca':
        f[0, 4] = 0.5 * dt * dt
        f[1, 5] = 0.5 * dt * dt
        f[2, 4] = dt
        f[3, 5] = dt
    elif name == 'cv':
        f[4, 4] = 0
        f[5, 5] = 0
    elif name == 'damped':
        f[2, 2] = 0.72
        f[3, 3] = 0.72
        f[4, 4] = 0
        f[5, 5] = 0
    else:
        raise ValueError(name)
    return f

def process_cov(name: str, dt: float, q: float) -> np.ndarray:
    scale = {'cv': 1.0, 'ca': 1.8, 'damped': 0.7}[name]
    diag = np.array([dt ** 3, dt ** 3, dt ** 2, dt ** 2, dt, dt, 0.02, 0.02]) * q * scale
    return np.diag(np.maximum(diag, 1e-06))

def observation(row: Detection) -> np.ndarray:
    return np.array([row.center_x, row.center_y, np.log(max(row.width, 1e-06)), np.log(max(row.height, 1e-06))])

def _transition(config: ImmConfig) -> np.ndarray:
    n = len(_MODEL_NAMES)
    off = (1 - config.transition_stay) / (n - 1)
    return np.full((n, n), off) + np.eye(n) * (config.transition_stay - off)

def smooth_track(rows: list[Detection], config: ImmConfig=ImmConfig()) -> list[Detection]:
    rows = sorted(rows, key=lambda r: r.frame_id)
    if len(rows) < 2:
        return rows
    h = np.zeros((4, 8))
    h[0, 0] = h[1, 1] = h[2, 6] = h[3, 7] = 1
    rbase = np.diag([config.measurement_noise ** 2, config.measurement_noise ** 2, config.size_noise ** 2, config.size_noise ** 2])
    trans = _transition(config)
    n = 3
    z0 = observation(rows[0])
    x = np.tile(np.array([z0[0], z0[1], 0, 0, 0, 0, z0[2], z0[3]]), (n, 1))
    p = np.tile(np.diag([16, 16, 25, 25, 25, 25, 0.2, 0.2]), (n, 1, 1))
    probs = np.full(n, 1 / n)
    filtered = []
    predicted = []
    mix_probs = []
    fs = []
    previous = rows[0].frame_id
    for row in rows:
        dt = max(1, row.frame_id - previous)
        previous = row.frame_id
        c = probs @ trans
        c = np.maximum(c, config.min_probability)
        c /= c.sum()
        mixed_x = []
        mixed_p = []
        for j in range(n):
            weights = probs * trans[:, j] / c[j]
            mean = np.sum(weights[:, None] * x, axis=0)
            cov = np.zeros((8, 8))
            for i in range(n):
                d = x[i] - mean
                cov += weights[i] * (p[i] + np.outer(d, d))
            mixed_x.append(mean)
            mixed_p.append(cov)
        new_x = []
        new_p = []
        likelihood = []
        pred_store = []
        f_store = []
        z = observation(row)
        for j, name in enumerate(_MODEL_NAMES):
            f = model_matrix(name, dt)
            xp = f @ mixed_x[j]
            pp = f @ mixed_p[j] @ f.T + process_cov(name, dt, config.process_noise)
            innovation = z - h @ xp
            s = h @ pp @ h.T + rbase
            inv = np.linalg.pinv(s)
            mahal = float(innovation @ inv @ innovation)
            robust = (config.student_df + 4) / (config.student_df + mahal)
            reff = rbase / max(robust, 0.1)
            s = h @ pp @ h.T + reff
            inv = np.linalg.pinv(s)
            k = pp @ h.T @ inv
            xu = xp + k @ (z - h @ xp)
            pu = (np.eye(8) - k @ h) @ pp
            sign, logdet = np.linalg.slogdet(s)
            loglike = -0.5 * (mahal + (logdet if sign > 0 else 50) + 4 * np.log(2 * np.pi))
            likelihood.append(loglike)
            new_x.append(xu)
            new_p.append(pu)
            pred_store.append((xp, pp))
            f_store.append(f)
        logs = np.log(c) + np.asarray(likelihood)
        logs -= logs.max()
        probs = np.exp(logs)
        probs = np.maximum(probs, config.min_probability)
        probs /= probs.sum()
        x = np.asarray(new_x)
        p = np.asarray(new_p)
        filtered.append((x.copy(), p.copy(), probs.copy()))
        predicted.append(pred_store)
        mix_probs.append(c.copy())
        fs.append(f_store)
    xs = [[None] * len(rows) for _ in range(n)]
    ps = [[None] * len(rows) for _ in range(n)]
    for j in range(n):
        xs[j][-1] = filtered[-1][0][j]
        ps[j][-1] = filtered[-1][1][j]
    for t in range(len(rows) - 2, -1, -1):
        for j in range(n):
            xf, pf = (filtered[t][0][j], filtered[t][1][j])
            f = fs[t + 1][j]
            xp, pp = predicted[t + 1][j]
            gain = pf @ f.T @ np.linalg.pinv(pp)
            xs[j][t] = xf + gain @ (xs[j][t + 1] - xp)
            ps[j][t] = pf + gain @ (ps[j][t + 1] - pp) @ gain.T
    out = []
    for t, row in enumerate(rows):
        weights = filtered[t][2]
        state = sum((weights[j] * xs[j][t] for j in range(n)))
        width = float(np.exp(np.clip(state[6], -5, 12)))
        height = float(np.exp(np.clip(state[7], -5, 12)))
        out.append(replace(row, x1=max(0.0, float(state[0] - width / 2)), y1=max(0.0, float(state[1] - height / 2)), width=width, height=height))
    return out

def smooth_directory(input_dir: Path, output_dir: Path, sequence_names: list[str] | None=None, config: ImmConfig=ImmConfig()) -> dict:
    allowed = None if sequence_names is None else set(sequence_names)
    summary = {}
    for file in prediction_files(input_dir):
        if allowed is not None and file.stem not in allowed:
            continue
        tracks = group_id(read_rows(file))
        rows = [r for identity in sorted(tracks) for r in smooth_track(tracks[identity], config)]
        write_rows(output_dir / file.name, rows)
        summary[file.stem] = len(rows)
    return {'sequences': summary, 'config': config.__dict__}

def _names(path):
    return None if path is None else [x.strip() for x in path.read_text().splitlines() if x.strip()]

def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input_dir', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--sequence-list', type=Path)
    p.add_argument('--process-noise', type=float, default=2)
    p.add_argument('--measurement-noise', type=float, default=4)
    p.add_argument('--student-df', type=float, default=5)
    p.add_argument('--summary-json', type=Path)
    return p

def main(argv=None):
    a = build_parser().parse_args(argv)
    result = smooth_directory(a.input_dir, a.output_dir, _names(a.sequence_list), ImmConfig(process_noise=a.process_noise, measurement_noise=a.measurement_noise, student_df=a.student_df))
    if a.summary_json:
        write_json(a.summary_json, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
