"""Central registry for named RaFT-UAV experiment methods.

Several research scripts build commands manually, which makes SOTA rows hard to
compare when a method relies on runtime flags. This registry gives each named
row a single identifier, a command template, and the environment variables that
must be active for that row.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class MethodSpec:
    """A named, reproducible experiment method."""

    method_id: str
    description: str
    command: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    leakage_safe: bool = True
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["command"] = list(self.command)
        record["env"] = dict(self.env)
        record["tags"] = list(self.tags)
        return record

    def shell_command(self) -> str:
        """Return a copy/paste shell command with required env prefixes."""

        env_prefix = " ".join(f"{key}={_shell_quote(value)}" for key, value in self.env.items())
        command = " ".join(_shell_quote(part) for part in self.command)
        return f"{env_prefix} {command}".strip()


METHOD_REGISTRY: dict[str, MethodSpec] = {
    "paper_strict_table2": MethodSpec(
        method_id="paper_strict_table2",
        description=(
            "Strict Table-II paper reproduction path with 800 m range gate "
            "and NIS validation."
        ),
        command=(
            "raft-uav-paper-strict",
            "{dataset_root}",
            "--flight",
            "{flight}",
            "--output-dir",
            "{output_dir}",
            "--enu-origin",
            "lw1",
        ),
        tags=("paper", "diagnostic", "strict"),
        leakage_safe=False,
        notes="Uses same-flight empirical covariance by default for paper reproduction.",
    ),
    "radar_geometry_audit": MethodSpec(
        method_id="radar_geometry_audit",
        description=(
            "Fortem LLA versus native polar backprojection coordinate audit."
        ),
        command=(
            "raft-uav-radar-geometry-audit",
            "{dataset_root}",
            "--flight",
            "{flight}",
            "--output-dir",
            "{output_dir}",
        ),
        tags=("diagnostic", "geometry", "radar"),
    ),
    "nis_reliability": MethodSpec(
        method_id="nis_reliability",
        description=(
            "NIS reliability and covariance-scale report from tracker diagnostics."
        ),
        command=(
            "raft-uav-nis-reliability",
            "{output_dir}",
            "--output-csv",
            "{output_dir}/nis_reliability_report.csv",
            "--output-json",
            "{output_dir}/nis_reliability_report.json",
        ),
        tags=("diagnostic", "calibration", "nis"),
    ),
    "tracklet_feature_store": MethodSpec(
        method_id="tracklet_feature_store",
        description=(
            "Candidate feature store with oracle ranks and selected-vs-best regret rows."
        ),
        command=(
            "raft-uav-tracklet-feature-store",
            "{dataset_root}",
            "--flight",
            "{flight}",
            "--output-dir",
            "{output_dir}",
        ),
        tags=("diagnostic", "association", "features"),
        leakage_safe=False,
        notes="Oracle ranks and regret use truth for post-run diagnostics only.",
    ),
    "cv_catprob": MethodSpec(
        method_id="cv_catprob",
        description="Asynchronous CV Kalman baseline using catProb radar association.",
        command=(
            "raft-uav-legacy",
            "{dataset_root}",
            "--flight",
            "{flight}",
            "--output-dir",
            "{output_dir}",
            "--radar-selection",
            "catprob",
        ),
        tags=("baseline", "cv", "catprob"),
    ),
    "cv_prediction_nis_fixed_lag": MethodSpec(
        method_id="cv_prediction_nis_fixed_lag",
        description="Prediction-NIS radar association with fixed-lag smoothing wrapper.",
        command=(
            "raft-uav",
            "run-baseline",
            "{dataset_root}",
            "--flight",
            "{flight}",
            "--output-dir",
            "{output_dir}",
            "--radar-association",
            "prediction-nis",
            "--enable-gating",
            "--rf-gate-prob",
            "0.99",
            "--radar-gate-prob",
            "0.99",
            "--smoother",
            "fixed-lag",
            "--smoother-lag-s",
            "20",
        ),
        tags=("baseline", "cv", "nis", "fixed-lag"),
    ),
    "imm_tracklet_viterbi_fixed_lag": MethodSpec(
        method_id="imm_tracklet_viterbi_fixed_lag",
        description=(
            "Canonical fixed-lag tracklet Viterbi association replayed through "
            "the IMM tracker."
        ),
        command=(
            "raft-uav",
            "run-baseline",
            "{dataset_root}",
            "--flight",
            "{flight}",
            "--output-dir",
            "{output_dir}",
            "--radar-association",
            "tracklet-viterbi",
            "--tracklet-variant",
            "range-covariance",
            "--tracklet-replay-tracker",
            "imm",
            "--smoother",
            "fixed-lag",
            "--smoother-lag-s",
            "20",
        ),
        tags=("sota", "imm", "tracklet", "fixed-lag"),
    ),
    "imm_tracklet_viterbi_fixed_lag_softk": MethodSpec(
        method_id="imm_tracklet_viterbi_fixed_lag_softk",
        description=(
            "IMM fixed-lag tracklet Viterbi with soft top-k path retention "
            "enabled."
        ),
        command=(
            "raft-uav",
            "run-baseline",
            "{dataset_root}",
            "--flight",
            "{flight}",
            "--output-dir",
            "{output_dir}",
            "--radar-association",
            "tracklet-viterbi",
            "--tracklet-variant",
            "range-covariance",
            "--tracklet-replay-tracker",
            "imm",
            "--smoother",
            "fixed-lag",
            "--smoother-lag-s",
            "20",
        ),
        env={
            "RAFT_UAV_TRACKLET_SOFT_TOP_K_PATHS": "3",
            "RAFT_UAV_TRACKLET_SOFT_PATH_TEMPERATURE": "1.5",
        },
        tags=("sota", "imm", "tracklet", "soft-top-k", "fixed-lag"),
    ),
    "imm_tracklet_viterbi_fixed_lag_dnh": MethodSpec(
        method_id="imm_tracklet_viterbi_fixed_lag_dnh",
        description=(
            "IMM fixed-lag tracklet Viterbi with conservative do-no-harm "
            "radar updates."
        ),
        command=(
            "raft-uav",
            "run-baseline",
            "{dataset_root}",
            "--flight",
            "{flight}",
            "--output-dir",
            "{output_dir}",
            "--radar-association",
            "tracklet-viterbi",
            "--tracklet-variant",
            "range-covariance",
            "--tracklet-replay-tracker",
            "imm",
            "--smoother",
            "fixed-lag",
            "--smoother-lag-s",
            "20",
        ),
        env={
            "RAFT_UAV_DO_NO_HARM_RADAR_UPDATE_POLICY": "posterior-error-nondegrading",
        },
        tags=("sota", "imm", "tracklet", "do-no-harm", "fixed-lag"),
    ),
    "imm_tracklet_viterbi_fixed_lag_softk_dnh": MethodSpec(
        method_id="imm_tracklet_viterbi_fixed_lag_softk_dnh",
        description=(
            "IMM fixed-lag tracklet Viterbi with soft top-k retention and "
            "do-no-harm updates."
        ),
        command=(
            "raft-uav",
            "run-baseline",
            "{dataset_root}",
            "--flight",
            "{flight}",
            "--output-dir",
            "{output_dir}",
            "--radar-association",
            "tracklet-viterbi",
            "--tracklet-variant",
            "range-covariance",
            "--tracklet-replay-tracker",
            "imm",
            "--smoother",
            "fixed-lag",
            "--smoother-lag-s",
            "20",
        ),
        env={
            "RAFT_UAV_TRACKLET_SOFT_TOP_K_PATHS": "3",
            "RAFT_UAV_TRACKLET_SOFT_PATH_TEMPERATURE": "1.5",
            "RAFT_UAV_DO_NO_HARM_RADAR_UPDATE_POLICY": "posterior-error-nondegrading",
        },
        tags=(
            "sota",
            "imm",
            "tracklet",
            "soft-top-k",
            "do-no-harm",
            "fixed-lag",
        ),
    ),
    "hetero_cv_lofo_nis_fixed_lag": MethodSpec(
        method_id="hetero_cv_lofo_nis_fixed_lag",
        description=(
            "Heteroscedastic CV fixed-lag row with LOFO NIS covariance "
            "calibration."
        ),
        command=(
            "raft-uav-heteroscedastic",
            "{dataset_root}",
            "--flight",
            "{flight}",
            "--output-dir",
            "{output_dir}",
            "--radar-association",
            "prediction-nis",
            "--fixed-lag-s",
            "20",
        ),
        tags=("heteroscedastic", "lofo", "nis", "fixed-lag"),
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-method-registry",
        description="list or resolve named RaFT-UAV experiment methods",
    )
    parser.add_argument("--method", choices=sorted(METHOD_REGISTRY), default=None)
    parser.add_argument("--dataset-root", default="{dataset_root}")
    parser.add_argument("--flight", default="{flight}")
    parser.add_argument("--output-dir", default="{output_dir}")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--shell",
        action="store_true",
        help="print only the resolved shell command",
    )
    args = parser.parse_args(argv)

    if args.method is None:
        frame = method_registry_frame()
        if args.output_json is not None:
            write_method_registry_json(args.output_json)
            print(f"registry_json={args.output_json}")
        print(frame.to_string(index=False))
        return 0

    spec = get_method_spec(args.method)
    resolved = resolve_method_spec(
        spec,
        dataset_root=args.dataset_root,
        flight=args.flight,
        output_dir=args.output_dir,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(resolved, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.shell:
        print(resolved["shell_command"])
    else:
        print(json.dumps(resolved, indent=2, sort_keys=True))
    return 0


def get_method_spec(method_id: str) -> MethodSpec:
    """Return a named method spec, raising ValueError if unknown."""

    try:
        return METHOD_REGISTRY[str(method_id)]
    except KeyError as exc:
        raise ValueError(f"unknown method_id {method_id!r}") from exc


def method_registry_frame(methods: Mapping[str, MethodSpec] | None = None) -> pd.DataFrame:
    """Return a compact dataframe of registered method metadata."""

    registry = METHOD_REGISTRY if methods is None else methods
    rows = []
    for spec in registry.values():
        rows.append(
            {
                "method_id": spec.method_id,
                "leakage_safe": bool(spec.leakage_safe),
                "tags": ",".join(spec.tags),
                "env": " ".join(f"{key}={value}" for key, value in spec.env.items()),
                "command": " ".join(spec.command),
                "description": spec.description,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("method_id").reset_index(drop=True)


def resolve_method_spec(
    spec: MethodSpec | str,
    *,
    dataset_root: Path | str,
    flight: str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Fill command placeholders for one method invocation."""

    if isinstance(spec, str):
        spec = get_method_spec(spec)
    replacements = {
        "dataset_root": str(dataset_root),
        "flight": str(flight),
        "output_dir": str(output_dir),
    }
    command = tuple(_replace_placeholders(part, replacements) for part in spec.command)
    resolved = spec.to_record()
    resolved["command"] = list(command)
    resolved["shell_command"] = _shell_command(command, spec.env)
    return resolved


def write_method_registry_json(path: Path | str) -> Path:
    """Write the registry as deterministic JSON."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {method_id: spec.to_record() for method_id, spec in sorted(METHOD_REGISTRY.items())}
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _replace_placeholders(value: str, replacements: Mapping[str, str]) -> str:
    out = str(value)
    for key, replacement in replacements.items():
        out = out.replace("{" + key + "}", replacement)
    return out


def _shell_command(command: Sequence[str], env: Mapping[str, str]) -> str:
    env_prefix = " ".join(f"{key}={_shell_quote(value)}" for key, value in env.items())
    command_text = " ".join(_shell_quote(part) for part in command)
    return f"{env_prefix} {command_text}".strip()


def _shell_quote(value: object) -> str:
    text = str(value)
    if text == "":
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:=+-{}")
    if all(char in safe for char in text):
        return text
    return "'" + text.replace("'", "'\\''") + "'"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
