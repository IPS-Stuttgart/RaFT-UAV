#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 VENV_PATH PROVENANCE_JSON" >&2
  exit 2
fi

venv_path="$1"
provenance_json="$2"
py="${venv_path}/bin/python"
work_root="$(cd "$(dirname "${venv_path}")/../.." && pwd)"
botsort_root="${work_root}/repos/YOLOv12-BoT-SORT-ReID/BoT-SORT"
pyrecest_commit="6e240d27dccc5d938e7432bf5eb70a281982b9d2"

if [[ ! -x "${py}" ]]; then
  echo "runtime Python is not executable: ${py}" >&2
  exit 1
fi
if [[ ! -d "${botsort_root}/tracker" ]]; then
  echo "BoT-SORT checkout is missing: ${botsort_root}" >&2
  exit 1
fi

runtime_extras_valid=false
if BOTSORT_ROOT="${botsort_root}" PYRECEST_COMMIT="${pyrecest_commit}" \
  "${py}" - <<'PY'
from __future__ import annotations

import json
import os
import sys
from importlib.metadata import distribution, version

import faiss
import numpy as np
from cython_bbox import bbox_overlaps
from pyrecest.evaluation.tracking_metrics import (
    ClearCounts,
    IdentityCounts,
    evaluate_clear,
    evaluate_identity,
)

assert version("Cython") == "3.0.11"
assert version("cython-bbox") == "0.1.5"
assert version("faiss-cpu") == "1.8.0.post1"
assert version("pyrecest") == "2.4.3"
direct_url = json.loads(distribution("pyrecest").read_text("direct_url.json") or "{}")
actual_commit = direct_url.get("vcs_info", {}).get("commit_id")
assert actual_commit == os.environ["PYRECEST_COMMIT"]
assert callable(evaluate_clear)
assert callable(evaluate_identity)
assert ClearCounts is not None
assert IdentityCounts is not None
boxes = np.ascontiguousarray([[0.0, 0.0, 9.0, 9.0]], dtype=np.float64)
overlaps = bbox_overlaps(boxes, boxes)
assert overlaps.shape == (1, 1)
assert np.isclose(overlaps[0, 0], 1.0)
vectors = np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
index = faiss.IndexFlatL2(2)
index.add(vectors)
distances, indices = index.search(vectors[:1], 1)
assert indices.tolist() == [[0]]
assert np.allclose(distances, [[0.0]])
sys.path.insert(0, os.environ["BOTSORT_ROOT"])
from tracker.mc_bot_sort import BoTSORT  # noqa: F401,E402
PY
then
  runtime_extras_valid=true
fi

if [[ "${runtime_extras_valid}" != "true" ]]; then
  "${py}" -m pip install "Cython==3.0.11"
  "${py}" -m pip install --no-build-isolation "cython_bbox==0.1.5"
  "${py}" -m pip install --no-deps "faiss-cpu==1.8.0.post1"
  "${py}" -m pip install --no-deps --force-reinstall \
    "git+https://github.com/FlorianPfaff/PyRecEst.git@${pyrecest_commit}"
fi

BOTSORT_ROOT="${botsort_root}" PYRECEST_COMMIT="${pyrecest_commit}" \
  "${py}" - <<'PY'
from __future__ import annotations

import json
import os
import sys
from importlib.metadata import distribution, version

import faiss
import numpy as np
from cython_bbox import bbox_overlaps
from pyrecest.evaluation.tracking_metrics import (
    ClearCounts,
    IdentityCounts,
    evaluate_clear,
    evaluate_identity,
)

boxes = np.ascontiguousarray(
    [
        [0.0, 0.0, 9.0, 9.0],
        [20.0, 20.0, 29.0, 29.0],
    ],
    dtype=np.float64,
)
overlaps = bbox_overlaps(boxes, boxes)
expected = np.eye(2, dtype=np.float64)
if overlaps.shape != expected.shape or not np.allclose(overlaps, expected):
    raise SystemExit(f"cython_bbox smoke test failed: {overlaps!r}")

vectors = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ],
    dtype=np.float32,
)
index = faiss.IndexFlatL2(3)
index.add(vectors)
distances, indices = index.search(vectors[:1], 2)
if indices.tolist() != [[0, 1]] or not np.allclose(distances, [[0.0, 1.0]]):
    raise SystemExit(
        f"faiss nearest-neighbour smoke test failed: {distances!r}, {indices!r}"
    )

pyrecest_direct_url = json.loads(
    distribution("pyrecest").read_text("direct_url.json") or "{}"
)
pyrecest_actual_commit = pyrecest_direct_url.get("vcs_info", {}).get("commit_id")
if pyrecest_actual_commit != os.environ["PYRECEST_COMMIT"]:
    raise SystemExit(
        "PyRecEst commit mismatch: "
        f"{pyrecest_actual_commit!r} != {os.environ['PYRECEST_COMMIT']!r}"
    )
if not callable(evaluate_clear) or not callable(evaluate_identity):
    raise SystemExit("PyRecEst tracking metric exports are unavailable")
if ClearCounts is None or IdentityCounts is None:
    raise SystemExit("PyRecEst tracking metric count types are unavailable")

sys.path.insert(0, os.environ["BOTSORT_ROOT"])
from tracker.mc_bot_sort import BoTSORT  # noqa: F401,E402

print(
    json.dumps(
        {
            "Cython": version("Cython"),
            "cython-bbox": version("cython-bbox"),
            "faiss-cpu": version("faiss-cpu"),
            "pyrecest": version("pyrecest"),
            "pyrecest_commit": pyrecest_actual_commit,
            "bbox_overlaps_smoke_test": "passed",
            "faiss_search_smoke_test": "passed",
            "pyrecest_tracking_metrics_smoke_test": "passed",
            "botsort_import_smoke_test": "passed",
        },
        indent=2,
        sort_keys=True,
    )
)
PY

"${py}" -m pip freeze --all > "$(dirname "${provenance_json}")/pip-freeze.txt"
PYRECEST_COMMIT="${pyrecest_commit}" "${py}" - "${provenance_json}" <<'PY'
from __future__ import annotations

import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["runtime_extras"] = {
    "Cython": version("Cython"),
    "cython-bbox": version("cython-bbox"),
    "faiss-cpu": version("faiss-cpu"),
    "pyrecest": version("pyrecest"),
    "pyrecest_commit": os.environ["PYRECEST_COMMIT"],
    "bbox_overlaps_smoke_test": "passed",
    "faiss_search_smoke_test": "passed",
    "pyrecest_tracking_metrics_smoke_test": "passed",
    "botsort_import_smoke_test": "passed",
}
path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
