"""Multi-head edge likelihood aligned with LTS HOTA, CLEAR, and identity losses."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from ._proposal_edge_likelihood import EdgeLikelihoodModel

_MODEL_SCHEMA = "raft-uav-multi-uav-lts-metric-edge-likelihood-v1"


@dataclass(frozen=True)
class MetricEdgeLikelihoodModel:
    """Three calibrated edge heads combined into one association cost."""

    schema: str
    identity: EdgeLikelihoodModel
    hota_005: EdgeLikelihoodModel
    clear_050: EdgeLikelihoodModel
    identity_weight: float = 0.75
    hota_weight: float = 1.0
    clear_weight: float = 0.25
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.schema != _MODEL_SCHEMA:
            raise ValueError(f"unsupported metric-edge schema: {self.schema}")
        self.identity.validate()
        self.hota_005.validate()
        self.clear_050.validate()
        for name, value in (
            ("identity_weight", self.identity_weight),
            ("hota_weight", self.hota_weight),
            ("clear_weight", self.clear_weight),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"metric-edge {name} must be finite and non-negative")
        if self.identity_weight + self.hota_weight + self.clear_weight <= 0.0:
            raise ValueError("at least one metric-edge head weight must be positive")
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise ValueError("metric-edge metadata must be an object")
        counts = {
            self.identity.training_example_count,
            self.hota_005.training_example_count,
            self.clear_050.training_example_count,
        }
        sequences = {
            self.identity.sequence_count,
            self.hota_005.sequence_count,
            self.clear_050.sequence_count,
        }
        if len(counts) != 1 or len(sequences) != 1:
            raise ValueError("metric-edge heads must be fitted on the same example panel")

    @property
    def training_example_count(self) -> int:
        return self.identity.training_example_count

    @property
    def sequence_count(self) -> int:
        return self.identity.sequence_count

    def head_probabilities(self, features: Sequence[float]) -> dict[str, float]:
        return {
            "identity": self.identity.probability(features),
            "hota_005": self.hota_005.probability(features),
            "clear_050": self.clear_050.probability(features),
        }

    def negative_log_probability(self, features: Sequence[float]) -> float:
        total_weight = self.identity_weight + self.hota_weight + self.clear_weight
        value = (
            self.identity_weight * self.identity.negative_log_probability(features)
            + self.hota_weight * self.hota_005.negative_log_probability(features)
            + self.clear_weight * self.clear_050.negative_log_probability(features)
        ) / total_weight
        return float(value)

    def with_weights(
        self,
        *,
        identity_weight: float | None = None,
        hota_weight: float | None = None,
        clear_weight: float | None = None,
    ) -> MetricEdgeLikelihoodModel:
        model = replace(
            self,
            identity_weight=(
                self.identity_weight if identity_weight is None else identity_weight
            ),
            hota_weight=self.hota_weight if hota_weight is None else hota_weight,
            clear_weight=self.clear_weight if clear_weight is None else clear_weight,
        )
        model.validate()
        return model

    def to_dict(self) -> dict[str, object]:
        self.validate()
        metadata = dict(self.metadata or {})
        return {
            "schema": self.schema,
            "identity": self.identity.to_dict(),
            "hota_005": self.hota_005.to_dict(),
            "clear_050": self.clear_050.to_dict(),
            "identity_weight": self.identity_weight,
            "hota_weight": self.hota_weight,
            "clear_weight": self.clear_weight,
            "training_example_count": self.training_example_count,
            "sequence_count": self.sequence_count,
            "metadata": metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> MetricEdgeLikelihoodModel:
        try:
            identity_payload = payload["identity"]
            hota_payload = payload["hota_005"]
            clear_payload = payload["clear_050"]
            if not all(
                isinstance(value, Mapping)
                for value in (identity_payload, hota_payload, clear_payload)
            ):
                raise TypeError("metric-edge heads must be objects")
            model = cls(
                schema=str(payload["schema"]),
                identity=EdgeLikelihoodModel.from_dict(identity_payload),
                hota_005=EdgeLikelihoodModel.from_dict(hota_payload),
                clear_050=EdgeLikelihoodModel.from_dict(clear_payload),
                identity_weight=float(payload.get("identity_weight", 0.75)),
                hota_weight=float(payload.get("hota_weight", 1.0)),
                clear_weight=float(payload.get("clear_weight", 0.25)),
                metadata=dict(payload.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed metric-edge likelihood model") from exc
        model.validate()
        return model


def load_metric_edge_likelihood_model(path: Path) -> MetricEdgeLikelihoodModel:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read metric-edge model: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("metric-edge model must contain a JSON object")
    return MetricEdgeLikelihoodModel.from_dict(payload)


def write_metric_edge_likelihood_model(
    model: MetricEdgeLikelihoodModel,
    path: Path,
) -> None:
    model.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
