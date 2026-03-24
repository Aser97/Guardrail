#!/usr/bin/env bash
set -euo pipefail

# Repo root = parent of project/ (where pyproject.toml and .venv live)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR=".venv"
HACKATHON_JSON="$REPO_ROOT/hackathon.json"
USE_SYSTEM_SITE_PACKAGES="${USE_SYSTEM_SITE_PACKAGES:-0}"

echo "== configure =="

# Goal of this step:
# - Install dependencies
# - Download / materialize model artifacts from hackathon.json (optional)

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtualenv at $VENV_DIR"
  USE_SYSTEM_SITE_PACKAGES_NORM="$(printf '%s' "$USE_SYSTEM_SITE_PACKAGES" | tr '[:upper:]' '[:lower:]')"
  case "$USE_SYSTEM_SITE_PACKAGES_NORM" in
    1|true|yes)
      python3 -m venv "$VENV_DIR" --system-site-packages
      ;;
    0|false|no|"")
      python3 -m venv "$VENV_DIR"
      ;;
    *)
      echo "ERROR: USE_SYSTEM_SITE_PACKAGES must be one of: 1,true,yes,0,false,no (got: $USE_SYSTEM_SITE_PACKAGES)" >&2
      exit 1
      ;;
  esac
fi
PY="$VENV_DIR/bin/python"
"$PY" -m pip install -U pip >/dev/null

if [[ -f requirements.txt ]]; then
  echo "Installing Python dependencies from requirements.txt"
  "$PY" -m pip install -r requirements.txt
elif [[ -f pyproject.toml ]]; then
  echo "Installing project and dependencies from pyproject.toml"
  "$PY" -m pip install -e .
else
  echo "No requirements.txt or pyproject.toml found. Skipping dependency install."
fi

echo "Ensuring ipykernel is available for notebooks"
"$PY" -m pip install ipykernel

if [[ ! -f "$HACKATHON_JSON" ]]; then
  echo "ERROR: hackathon.json not found at $HACKATHON_JSON" >&2
  exit 1
fi

# Parse and apply hackathon runtime configuration.
# - needs_gpu: if true, fail fast when CUDA is unavailable
# - artifacts: optional list of downloadable resources
export REPO_ROOT HACKATHON_JSON
"$PY" - <<'PY'
import datetime
import hashlib
import hmac
import json
import os
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def s3_download(bucket: str, key: str, dest_path: Path) -> None:
    """Download from S3-compatible storage using signature v4."""
    endpoint = os.environ.get("S3_ENDPOINT_URL", "")
    access_key = os.environ.get("S3_ACCESS_KEY", "")
    secret_key = os.environ.get("S3_SECRET_KEY", "")

    if not all([endpoint, access_key, secret_key]):
        raise RuntimeError("Missing S3 environment variables: S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY")

    host = endpoint.replace("https://", "").replace("http://", "")
    region = "us-east-1"
    service = "s3"

    now = datetime.datetime.utcnow()
    date_stamp = now.strftime("%Y%m%d")
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")

    empty_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    canonical_uri = f"/{bucket}/{key}"
    canonical_querystring = ""
    signed_headers = "host;x-amz-content-sha256;x-amz-date"

    canonical_request = f"GET\n{canonical_uri}\n{canonical_querystring}\nhost:{host}\nx-amz-content-sha256:{empty_hash}\nx-amz-date:{amz_date}\n\n{signed_headers}\n{empty_hash}"

    algorithm = "AWS4-HMAC-SHA256"
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    canonical_request_hash = hashlib.sha256(canonical_request.encode()).hexdigest()
    string_to_sign = f"{algorithm}\n{amz_date}\n{credential_scope}\n{canonical_request_hash}"

    def sign(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode(), hashlib.sha256).digest()

    k_date = sign(f"AWS4{secret_key}".encode(), date_stamp)
    k_region = sign(k_date, region)
    k_service = sign(k_region, service)
    k_signing = sign(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode(), hashlib.sha256).hexdigest()

    authorization = f"{algorithm} Credential={access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"

    url = f"{endpoint}/{bucket}/{key}"
    req = urllib.request.Request(url, method="GET")
    req.add_header("Host", host)
    req.add_header("x-amz-content-sha256", empty_hash)
    req.add_header("x-amz-date", amz_date)
    req.add_header("Authorization", authorization)

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(req, context=ctx) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        last_milestone = 0
        with dest_path.open("wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded * 100 // total_size
                    milestone = pct // 25 * 25  # 0, 25, 50, 75, 100
                    if milestone > last_milestone:
                        print(f" {milestone}%", end="", file=sys.stderr, flush=True)
                        last_milestone = milestone
            print("", file=sys.stderr)


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (destination / member.name).resolve()
            if not str(member_path).startswith(str(destination)):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(destination)


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


repo_root = Path(os.environ["REPO_ROOT"]).resolve()
hackathon_json = Path(os.environ["HACKATHON_JSON"]).resolve()
print(f"[configure] Reading config file: {hackathon_json}", file=sys.stderr)
raw_json = hackathon_json.read_text()
print(f"[configure] hackathon.json size: {len(raw_json)} bytes", file=sys.stderr)
cfg = json.loads(raw_json)
print("[configure] Decoded hackathon.json content:", file=sys.stderr)
print(json.dumps(cfg, indent=2, sort_keys=True), file=sys.stderr)
if not isinstance(cfg, dict):
    raise RuntimeError("hackathon.json must be a JSON object")
print(f"[configure] Top-level config keys: {sorted(cfg.keys())}", file=sys.stderr)

if "needs_gpu" not in cfg:
    raise RuntimeError("hackathon.json missing required field: needs_gpu")
if not isinstance(cfg["needs_gpu"], bool):
    raise RuntimeError("hackathon.json field 'needs_gpu' must be a boolean")
needs_gpu = cfg["needs_gpu"]
print(f"[configure] needs_gpu={needs_gpu}", file=sys.stderr)

if "artifacts" not in cfg:
    raise RuntimeError("hackathon.json missing required field: artifacts")
artifacts = cfg["artifacts"]
if not isinstance(artifacts, list):
    raise RuntimeError("hackathon.json field 'artifacts' must be a list")
print(f"[configure] artifacts count={len(artifacts)}", file=sys.stderr)

for idx, artifact in enumerate(artifacts):
    if not isinstance(artifact, dict):
        raise RuntimeError(f"Artifact #{idx} must be an object")

    uri = artifact.get("uri")
    destination = artifact.get("destination")
    required = bool(artifact.get("required", True))
    expected_sha256 = artifact.get("sha256")
    if "required" in artifact and not isinstance(artifact["required"], bool):
        raise RuntimeError(f"Artifact #{idx} field 'required' must be a boolean")
    if expected_sha256 is not None and not isinstance(expected_sha256, str):
        raise RuntimeError(f"Artifact #{idx} field 'sha256' must be a string when provided")

    if not isinstance(uri, str) or not uri.strip():
        raise RuntimeError(f"Artifact #{idx} missing required string field: uri")
    if not isinstance(destination, str) or not destination.strip():
        raise RuntimeError(f"Artifact #{idx} missing required string field: destination")
    print(
        (
            f"[configure] Artifact #{idx} decoded: "
            f"uri={uri}, destination={destination}, required={required}, "
            f"sha256={'present' if bool(expected_sha256) else 'missing'}"
        ),
        file=sys.stderr,
    )

    dest_path = Path(destination)
    if not dest_path.is_absolute():
        dest_path = (repo_root / dest_path).resolve()
    else:
        dest_path = dest_path.resolve()

    # Keep artifacts inside the repository for safety.
    if not str(dest_path).startswith(str(repo_root)):
        raise RuntimeError(f"Artifact destination must stay under repo root: {dest_path}")

    print(f"[configure] Fetching artifact #{idx}: {uri} -> {dest_path}", file=sys.stderr)
    with tempfile.TemporaryDirectory() as td:
        tmp_file = Path(td) / "artifact.bin"
        try:
            # Step 1: Download
            print("  [1/3] Downloading...", file=sys.stderr)
            if uri.startswith("s3://"):
                # Parse s3://bucket/key format
                s3_path = uri[5:]  # Remove "s3://"
                bucket, key = s3_path.split("/", 1)
                s3_download(bucket, key, tmp_file)
            elif uri.startswith("http://") or uri.startswith("https://"):
                with urllib.request.urlopen(uri) as r:
                    if r.status < 200 or r.status >= 300:
                        raise RuntimeError(f"HTTP download failed with status {r.status}")
                    with tmp_file.open("wb") as f:
                        shutil.copyfileobj(r, f)
            else:
                raise RuntimeError("Unsupported uri scheme; use s3:// or https://")

            # Step 2: Verify SHA256
            if expected_sha256:
                print("  [2/3] Verifying SHA256...", end="", file=sys.stderr, flush=True)
                got = sha256sum(tmp_file)
                if got.lower() != str(expected_sha256).lower():
                    print(" FAILED", file=sys.stderr)
                    raise RuntimeError(f"SHA256 mismatch for {uri}: expected {expected_sha256}, got {got}")
                print(" OK", file=sys.stderr)
            else:
                print("  [2/3] Skipping SHA256 verification (none provided)", file=sys.stderr)

            # Step 3: Extract/Copy
            lower_uri = uri.lower()
            if lower_uri.endswith(".tar.gz") or lower_uri.endswith(".tgz"):
                print("  [3/3] Extracting tar.gz...", end="", file=sys.stderr, flush=True)
                dest_path.mkdir(parents=True, exist_ok=True)
                safe_extract_tar(tmp_file, dest_path)
                print(" Done", file=sys.stderr)
            elif lower_uri.endswith(".zip"):
                print("  [3/3] Extracting zip...", end="", file=sys.stderr, flush=True)
                dest_path.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(tmp_file, "r") as zf:
                    for member in zf.namelist():
                        member_path = (dest_path / member).resolve()
                        if not str(member_path).startswith(str(dest_path)):
                            raise RuntimeError(f"Unsafe zip member path: {member}")
                    zf.extractall(dest_path)
                print(" Done", file=sys.stderr)
            else:
                print("  [3/3] Copying file...", end="", file=sys.stderr, flush=True)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tmp_file, dest_path)
                print(" Done", file=sys.stderr)

        except Exception as exc:
            if required:
                raise
            print(f"[configure] Optional artifact failed ({uri}): {exc}", file=sys.stderr)

# If a GPU is required, verify CUDA availability after dependencies are installed.
if needs_gpu:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"needs_gpu=true but torch is not available: {exc}")
    if not torch.cuda.is_available():
        raise RuntimeError("needs_gpu=true but CUDA is not available on this machine")
PY

echo "OK"
