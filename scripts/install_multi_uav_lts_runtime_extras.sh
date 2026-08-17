#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 VENV_PATH PROVENANCE_JSON" >&2
  exit 2
fi

venv_path="$1"
provenance_json="$2"
py="${venv_path}/bin/python"

if [[ ! -x "${py}" ]]; then
  echo "runtime Python is not executable: ${py}" >&2
  exit 1
fi

runtime_extras_valid=false
if "${py}" - <<'PY'
from __future__ import annotations

from importlib.metadata import version

import numpy as np
from cython_bbox import bbox_overlaps

assert version("Cython") == "3.0.11"
assert version("cython-bbox") == "0.1.5"
boxes = np.ascontiguousarray([[0.0, 0.0, 9.0, 9.0]], dtype=np.float64)
overlaps = bbox_overlaps(boxes, boxes)
assert overlaps.shape == (1, 1)
assert np.isclose(overlaps[0, 0], 1.0)
PY
then
  runtime_extras_valid=true
fi

if [[ "${runtime_extras_valid}" != "true" ]]; then
  "${py}" -m pip install "Cython==3.0.11"
  "${py}" -m pip install --no-build-isolation "cython_bbox==0.1.5"
fi

"${py}" - <<'PY'
from __future__ import annotations

import json
from importlib.metadata import version

import numpy as np
from cython_bbox import bbox_overlaps

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
print(
    json.dumps(
        {
            "Cython": version("Cython"),
            "cython-bbox": version("cython-bbox"),
            "bbox_overlaps_smoke_test": "passed",
        },
        indent=2,
        sort_keys=True,
    )
)
PY

"${py}" -m pip freeze --all > "$(dirname "${provenance_json}")/pip-freeze.txt"
"${py}" - "${provenance_json}" <<'PY'
from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["runtime_extras"] = {
    "Cython": version("Cython"),
    "cython-bbox": version("cython-bbox"),
    "bbox_overlaps_smoke_test": "passed",
}
path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
