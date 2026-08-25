#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 VENV_PATH ASSET_DIR PROVENANCE_JSON" >&2
  exit 2
fi

venv_path="$1"
asset_dir="$2"
provenance_json="$3"
runtime_id="py311-torch222-cu118-flash273-pyrecest-v5"
pyrecest_revision="75b3b0b9e8b7a7c1a39fc69cdf85f0af9365f158"
pyrecest_requirement="pyrecest @ git+https://github.com/FlorianPfaff/PyRecEst.git@${pyrecest_revision}"
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

find_runtime_library_path() {
  local py="$1"
  "${py}" - <<'PY'
from __future__ import annotations

import importlib.util
import site
from pathlib import Path

site_roots = [Path(root) for root in site.getsitepackages()]
cuda_candidates: list[Path] = []
for root in site_roots:
    cuda_candidates.extend(
        (
            root / "nvidia" / "cuda_runtime" / "lib",
            root / "nvidia" / "cuda_runtime" / "lib64",
        )
    )
cuda_runtime = next(
    (
        candidate
        for candidate in cuda_candidates
        if (candidate / "libcudart.so.11.0").is_file()
    ),
    None,
)
if cuda_runtime is None:
    rendered = ", ".join(str(candidate) for candidate in cuda_candidates)
    raise SystemExit(f"libcudart.so.11.0 not found under: {rendered}")

torch_spec = importlib.util.find_spec("torch")
if torch_spec is None or torch_spec.origin is None:
    raise SystemExit("torch package could not be located")
torch_lib = Path(torch_spec.origin).resolve().parent / "lib"
if not (torch_lib / "libtorch_cuda.so").is_file():
    raise SystemExit(f"PyTorch CUDA libraries are missing from {torch_lib}")

paths: list[Path] = []
for candidate in (cuda_runtime, torch_lib):
    if candidate not in paths:
        paths.append(candidate)
print(":".join(str(path) for path in paths))
PY
}

runtime_valid=false
runtime_library_path=""
if [[ -x "${venv_path}/bin/python" ]] && [[ -f "${marker}" ]]; then
  py="${venv_path}/bin/python"
  if runtime_library_path="$(find_runtime_library_path "${py}")"; then
    export LD_LIBRARY_PATH="${runtime_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    if "${py}" - <<'PY'
import importlib.metadata
import json

import cv2
import flash_attn
import lap
import motmetrics
import numpy as np
import seaborn
import scipy
import torch
from pyrecest.evaluation import tracking_metrics

assert torch.__version__.split("+")[0] == "2.2.2"
assert torch.version.cuda == "11.8"
assert torch.cuda.is_available()
assert np.__version__ == "1.26.4"
assert scipy.__version__ == "1.15.3"
assert cv2.__version__ == "4.9.0"
assert getattr(flash_attn, "__version__", "") == "2.7.3"
assert lap.__version__ == "0.5.12"
assert motmetrics.__version__ == "1.4.0"
assert seaborn.__version__ == "0.13.2"
pyrecest_distribution = importlib.metadata.distribution("pyrecest")
pyrecest_direct_url = json.loads(
    pyrecest_distribution.read_text("direct_url.json") or "{}"
)
pyrecest_vcs = pyrecest_direct_url.get("vcs_info", {})
assert pyrecest_vcs.get("commit_id") == (
    "75b3b0b9e8b7a7c1a39fc69cdf85f0af9365f158"
)
clear = tracking_metrics.finalize_clear(
    tracking_metrics.ClearCounts(
        tp=1,
        fp=0,
        fn=0,
        id_switches=0,
        motp_sum=1.0,
    )
)
identity = tracking_metrics.finalize_identity(
    tracking_metrics.IdentityCounts(tp=1, fp=0, fn=0)
)
assert clear == {"mota": 1.0, "motp": 1.0}
assert identity == {
    "idf1": 1.0,
    "id_precision": 1.0,
    "id_recall": 1.0,
}
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
    "scipy==1.15.3" \
    "scikit-learn==1.5.2" \
    "matplotlib==3.9.2" \
    "mpmath==1.3.0" \
    "pyshtools==4.14.1" \
    "beartype==0.22.9" \
    "shapely==2.1.2" \
    "seaborn==0.13.2" \
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
  # Torch 2.2 requires the NumPy 1.x ABI. Install the immutable PyRecEst
  # tracking-metric revision without resolving its broader NumPy >=2 metadata;
  # every dependency imported by pyrecest.evaluation is pinned above and the
  # metric contract is exercised both here and when reusing the environment.
  "${py}" -m pip install --no-deps \
    "${pyrecest_requirement}"
  touch "${marker}"
fi

py="${venv_path}/bin/python"
runtime_library_path="$(find_runtime_library_path "${py}")"
export LD_LIBRARY_PATH="${runtime_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -n "${GITHUB_ENV:-}" ]]; then
  printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH}" >> "${GITHUB_ENV}"
fi

work_root="$(cd "$(dirname "${venv_path}")/../.." && pwd)"
upstream_inference="${work_root}/repos/YOLOv12-BoT-SORT-ReID/BoT-SORT/tools/inference.py"
"${py}" - "${upstream_inference}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

path = Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
old = '''from huggingface_hub import hf_hub_download
import shutil

# Define your target directory
target_dir = "logs/sbs_S50"
os.makedirs(target_dir, exist_ok=True)  # Make sure the directory exists

# List of files to download
files_to_download = ["model_0016.pth", "config.yaml"]

# Download each file and move it to the target directory
for filename in files_to_download:
    downloaded_path = hf_hub_download(
        repo_id="wish44165/YOLOv12-BoT-SORT-ReID",
        filename=filename
    )
    shutil.copy(downloaded_path, os.path.join(target_dir, filename))

print(f"Downloaded files are saved to: {target_dir}")
'''
new = '''from huggingface_hub import hf_hub_download
import shutil

# RaFT-UAV evidence runs provide checksum-verified assets at these paths. Keep
# the pinned revision as a deterministic fallback rather than downloading the
# mutable default branch on every inference process.
target_dir = "logs/sbs_S50"
os.makedirs(target_dir, exist_ok=True)
files_to_download = ["model_0016.pth", "config.yaml"]
for filename in files_to_download:
    destination = os.path.join(target_dir, filename)
    if not os.path.isfile(destination):
        downloaded_path = hf_hub_download(
            repo_id="wish44165/YOLOv12-BoT-SORT-ReID",
            filename=filename,
            revision="e677d81dac9909ddeabb6bc70ded5510ff4872aa",
        )
        shutil.copy2(downloaded_path, destination)

print(f"Using ReID assets from: {target_dir}")
'''
marker = "# RaFT-UAV evidence runs provide checksum-verified assets"
if marker not in source:
    if source.count(old) != 1:
        raise SystemExit("upstream ReID download block did not match the pinned source")
    source = source.replace(old, new)
    path.write_text(source, encoding="utf-8")
if marker not in path.read_text(encoding="utf-8"):
    raise SystemExit("upstream ReID download block was not patched")
PY
upstream_inference_sha256="$(sha256sum "${upstream_inference}" | awk '{print $1}')"

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
import importlib.metadata
import json

import cv2
import flash_attn
import lap
import motmetrics
import numpy as np
import seaborn
import scipy
import torch
from pyrecest.evaluation import tracking_metrics

if torch.version.cuda != "11.8":
    raise SystemExit(f"torch CUDA runtime is {torch.version.cuda!r}, expected '11.8'")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the pinned evidence environment")
pyrecest_distribution = importlib.metadata.distribution("pyrecest")
pyrecest_direct_url = json.loads(
    pyrecest_distribution.read_text("direct_url.json") or "{}"
)
pyrecest_revision = pyrecest_direct_url.get("vcs_info", {}).get("commit_id")
expected_pyrecest_revision = "75b3b0b9e8b7a7c1a39fc69cdf85f0af9365f158"
if pyrecest_revision != expected_pyrecest_revision:
    raise SystemExit(
        f"PyRecEst revision is {pyrecest_revision!r}, "
        f"expected {expected_pyrecest_revision!r}"
    )
clear = tracking_metrics.finalize_clear(
    tracking_metrics.ClearCounts(
        tp=1,
        fp=0,
        fn=0,
        id_switches=0,
        motp_sum=1.0,
    )
)
identity = tracking_metrics.finalize_identity(
    tracking_metrics.IdentityCounts(tp=1, fp=0, fn=0)
)
if clear != {"mota": 1.0, "motp": 1.0}:
    raise SystemExit(f"PyRecEst CLEAR metric smoke failed: {clear!r}")
expected_identity = {"idf1": 1.0, "id_precision": 1.0, "id_recall": 1.0}
if identity != expected_identity:
    raise SystemExit(f"PyRecEst identity metric smoke failed: {identity!r}")
print(
    json.dumps(
        {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opencv": cv2.__version__,
            "flash_attn": getattr(flash_attn, "__version__", "unknown"),
            "lap": getattr(lap, "__version__", "unknown"),
            "motmetrics": getattr(motmetrics, "__version__", "unknown"),
            "pyrecest": pyrecest_distribution.version,
            "pyrecest_revision": pyrecest_revision,
            "seaborn": seaborn.__version__,
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
  "${pyrecest_revision}" \
  "${flash_wheel}" \
  "${flash_sha256}" \
  "${runtime_library_path}" \
  "${ldd_output}" \
  "${upstream_inference}" \
  "${upstream_inference_sha256}" <<'PY'
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

(
    metadata_path,
    output_path,
    runtime_id,
    pyrecest_revision,
    wheel_path,
    wheel_sha256,
    runtime_library_path,
    ldd_output,
    upstream_inference,
    upstream_inference_sha256,
) = sys.argv[1:]
metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
freeze_path = Path(output_path).parent / "pip-freeze.txt"
payload = {
    "schema": "raft-uav-multi-uav-lts-runtime-v5",
    "runtime_id": runtime_id,
    "pyrecest": {
        "revision": pyrecest_revision,
    },
    "python": sys.version,
    "python_executable": sys.executable,
    "pip_freeze_path": str(freeze_path),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "runtime_library_path": runtime_library_path,
    "flash_attention_ldd_path": ldd_output,
    "upstream_inference_path": upstream_inference,
    "upstream_inference_sha256": upstream_inference_sha256,
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
