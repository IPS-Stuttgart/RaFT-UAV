from __future__ import annotations

import re
from pathlib import Path


INSTALLER = Path("scripts/install_multi_uav_lts_public_runtime.sh")
PYRECEST_REVISION = "75b3b0b9e8b7a7c1a39fc69cdf85f0af9365f158"


def test_public_runtime_installs_immutable_pyrecest_metric_provider() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    revision_match = re.search(
        r'^pyrecest_revision="([0-9a-f]{40})"$',
        source,
        re.MULTILINE,
    )
    assert revision_match is not None
    assert revision_match.group(1) == PYRECEST_REVISION
    assert "PyRecEst.git@${pyrecest_revision}" in source
    assert '"${py}" -m pip install --no-deps' in source
    assert '"${pyrecest_requirement}"' in source
    assert "PyRecEst.git@main" not in source


def test_public_runtime_validates_metric_import_and_versions() -> None:
    source = INSTALLER.read_text(encoding="utf-8")

    runtime_id = "py311-torch222-cu118-flash273-pyrecest-v5"
    assert f'runtime_id="{runtime_id}"' in source
    assert '"numpy==1.26.4"' in source
    assert '"scipy==1.15.3"' in source
    assert '"pyshtools==4.14.1"' in source
    assert "from pyrecest.evaluation import tracking_metrics" in source
    assert "tracking_metrics.ClearCounts" in source
    assert "tracking_metrics.IdentityCounts" in source
    assert '"schema": "raft-uav-multi-uav-lts-runtime-v5"' in source
