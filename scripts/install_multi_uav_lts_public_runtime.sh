#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 VENV_PATH ASSET_DIR PROVENANCE_JSON" >&2
  exit 2
fi

venv_path="$1"
asset_dir="$2"
provenance_json="$3"
runtime_id="py311-torch222-cu118-flash273-v2"
marker="${venv_path}/.raft-uav-${runtime_id}"

mkdir -p "${asset_dir}" "$(dirname "${provenance_json}")"

flash_asset_id="219545450"
flash_asset_name="flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
flash_asset_size="193082618"
flash_metadata="${asset_dir}/flash-attention-${flash_asset_id}.json"
flash_wheel="${asset_dir}/${flash_asset_name}"

python - "${flash_metadata}" <<'PY'
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

asset_id = 219_545_450
expected_name = (
    "flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-"
    "cp311-cp311-linux_x86_64.whl"
)
expected_size = 193_082_618
url = (
    "https://api.github.com/repos/Dao-AILab/flash-attention/"
    f"releases/assets/{asset_id}"
)
request = urllib.request.Request(
    url,
    headers={"User-Agent": "RaFT-UAV-evidence-workflow/1.0"},
)
for attempt in range(1, 9):
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
        break
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        if attempt == 8:
            raise
        time.sleep(min(attempt * 5, 30))
else:
    raise AssertionError("unreachable")

actual = (int(payload.get("id", -1)), payload.get("name"), int(payload.get("size", -1)))
expected = (asset_id, expected_name, expected_size)
if actual != expected:
    raise SystemExit(f"flash-attention release asset changed: {actual!r} != {expected!r}")
if payload.get("state") != "uploaded":
    raise SystemExit(f"flash-attention asset is not uploaded: {payload.get('state')!r}")
if not payload.get("browser_download_url"):
    raise SystemExit("flash-attention asset has no download URL")

Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

flash_url="$(${PYTHON:-python} - "${flash_metadata}" <<'PY'
import json
import sys
from pathlib import Path

print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["browser_download_url"])
PY
)"

if [[ -f "${flash_wheel}" ]] && [[ "$(stat -c %s "${flash_wheel}")" != "${flash_asset_size}" ]]; then
  rm -f "${flash_wheel}"
fi
partial="${flash_wheel}.part"
if [[ -f "${partial}" ]] && [[ "$(stat -c %s "${partial}")" -ge "${flash_asset_size}" ]]; then
  rm -f "${partial}"
fi
if [[ ! -f "${flash_wheel}" ]]; then
  curl \
    --location \
    --fail \
    --show-error \
    --silent \
    --retry 8 \
    --retry-delay 5 \
    --retry-all-errors \
    --continue-at - \
    --output "${partial}" \
    "${flash_url}"
  actual_size="$(stat -c %s "${partial}")"
  if [[ "${actual_size}" != "${flash_asset_size}" ]]; then
    echo "flash-attention wheel size ${actual_size} != ${flash_asset_size}" >&2
    exit 1
  fi
  mv "${partial}" "${flash_wheel}"
fi
flash_sha256="$(sha256sum "${flash_wheel}" | awk '{print $1}')"

find_cuda11_runtime_lib() {
  local py="$1"
  "${py}" - <<'PY'
from __future__ import annotations

import site
from pathlib import Path

candidates: list[Path] = []
for root_text in site.getsitepackages():
    root = Path(root_text)
    candidates.extend(
        (
            root / "nvidia" / "cuda_runtime" / "lib",
            root / "nvidia" / "cuda_runtime" / "lib64",
        )
    )
for candidate in candidates:
    if (candidate / "libcudart.so.11.0").is_file():
        print(candidate)
        break
else:
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise SystemExit(f"libcudart.so.11.0 not found under: {rendered}")
PY
}

runtime_valid=false
cuda_runtime_lib=""
if [[ -x "${venv_path}/bin/python" ]] && [[ -f "${marker}" ]]; then
  py="${venv_path}/bin/python"
  if cuda_runtime_lib="$(find_cuda11_runtime_lib "${py}")"; then
    export LD_LIBRARY_PATH="${cuda_runtime_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    if "${py}" - <<'PY'
import torch
import cv2
import flash_attn
import lap
import motmetrics

assert torch.__version__.split("+")[0] == "2.2.2"
assert torch.version.cuda == "11.8"
assert torch.cuda.is_available()
assert cv2.__version__ == "4.9.0"
assert getattr(flash_attn, "__version__", "") == "2.7.3"
assert lap.__version__ == "0.5.12"
assert motmetrics.__version__ == "1.4.0"
PY
    then
      runtime_valid=true
    fi
  fi
fi

if [[ "${runtime_valid}" != "true" ]]; then
  rm -rf "${venv_path}"
  python -m venv "${venv_path}"
  py="${venv_path}/bin/python"
  "${py}" -m pip install --upgrade \
    "pip==25.1.1" \
    "setuptools==69.5.1" \
    "wheel==0.45.1" \
    "packaging==24.2" \
    "ninja==1.11.1.3"
  "${py}" -m pip install \
    --index-url https://download.pytorch.org/whl/cu118 \
    "torch==2.2.2" \
    "torchvision==0.17.2"
  "${py}" -m pip install \
    "nvidia-cuda-runtime-cu11==11.8.89" \
    "${flash_wheel}"
  "${py}" -m pip install \
    "numpy==1.26.4" \
    "pandas==2.2.3" \
    "PyYAML==6.0.1" \
    "scipy==1.13.0" \
    "scikit-learn==1.5.2" \
    "matplotlib==3.9.2" \
    "timm==1.0.14" \
    "albumentations==2.0.4" \
    "pycocotools==2.0.7" \
    "opencv-python-headless==4.9.0.80" \
    "psutil==5.9.8" \
    "py-cpuinfo==9.0.0" \
    "huggingface-hub==0.23.2" \
    "safetensors==0.4.3" \
    "loguru==0.7.2" \
    "scikit-image==0.24.0" \
    "tqdm==4.66.5" \
    "Pillow==10.4.0" \
    "thop==0.1.1.post2209072238" \
    "tabulate==0.9.0" \
    "tensorboard==2.17.1" \
    "lap==0.5.12" \
    "motmetrics==1.4.0" \
    "filterpy==1.4.5" \
    "h5py==3.11.0" \
    "prettytable==3.11.0" \
    "easydict==1.13" \
    "yacs==0.1.8" \
    "termcolor==2.4.0" \
    "gdown==5.2.0"
  touch "${marker}"
fi

py="${venv_path}/bin/python"
cuda_runtime_lib="$(find_cuda11_runtime_lib "${py}")"
export LD_LIBRARY_PATH="${cuda_runtime_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -n "${GITHUB_ENV:-}" ]]; then
  printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH}" >> "${GITHUB_ENV}"
fi

flash_extension="$(${py} - <<'PY'
import importlib.util

spec = importlib.util.find_spec("flash_attn_2_cuda")
if spec is None or spec.origin is None:
    raise SystemExit("flash_attn_2_cuda extension could not be located")
print(spec.origin)
PY
)"
ldd_output="$(dirname "${provenance_json}")/flash-attention-ldd.txt"
ldd "${flash_extension}" | tee "${ldd_output}"
if grep -q 'not found' "${ldd_output}"; then
  echo "FlashAttention still has unresolved shared libraries." >&2
  exit 1
fi

"${py}" -m pip install -e . --no-deps
"${py}" - <<'PY'
import json

import torch
import cv2
import flash_attn
import lap
import motmetrics

if torch.version.cuda != "11.8":
    raise SystemExit(f"torch CUDA runtime is {torch.version.cuda!r}, expected '11.8'")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the pinned evidence environment")
print(
    json.dumps(
        {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "opencv": cv2.__version__,
            "flash_attn": getattr(flash_attn, "__version__", "unknown"),
            "lap": getattr(lap, "__version__", "unknown"),
            "motmetrics": getattr(motmetrics, "__version__", "unknown"),
        },
        indent=2,
        sort_keys=True,
    )
)
PY
"${py}" -m pip freeze --all > "$(dirname "${provenance_json}")/pip-freeze.txt"

"${py}" - \
  "${flash_metadata}" \
  "${provenance_json}" \
  "${runtime_id}" \
  "${flash_wheel}" \
  "${flash_sha256}" \
  "${cuda_runtime_lib}" \
  "${ldd_output}" <<'PY'
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

(
    metadata_path,
    output_path,
    runtime_id,
    wheel_path,
    wheel_sha256,
    cuda_runtime_lib,
    ldd_output,
) = sys.argv[1:]
metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
freeze_path = Path(output_path).parent / "pip-freeze.txt"
payload = {
    "schema": "raft-uav-multi-uav-lts-runtime-v2",
    "runtime_id": runtime_id,
    "python": sys.version,
    "python_executable": sys.executable,
    "pip_freeze_path": str(freeze_path),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "cuda_runtime_library_path": cuda_runtime_lib,
    "flash_attention_ldd_path": ldd_output,
    "flash_attention": {
        "asset_id": metadata["id"],
        "asset_name": metadata["name"],
        "asset_size": metadata["size"],
        "asset_created_at": metadata["created_at"],
        "wheel_path": wheel_path,
        "wheel_sha256": wheel_sha256,
    },
}
Path(output_path).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
