#!/usr/bin/env bash
set -euo pipefail

# Required environment variables (set in .env or environment):
#   S3_BUCKET_NAME  - bucket name
#   S3_ENDPOINT_URL     - endpoint URL
#   S3_ACCESS_KEY   - access key
#   S3_SECRET_KEY   - secret key

if [ $# -ne 2 ]; then
  echo "Usage: scripts/publish_artifact.sh <team_id> <local_path>" >&2
  exit 1
fi

TEAM_ID="$1"
LOCAL_PATH="$2"

# Determine script and repo root directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load .env file if it exists (provides S3 credentials)
ENV_FILE="$REPO_ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
  echo "Loading credentials from $ENV_FILE"
  # Export variables from .env, ignoring comments and empty lines
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  echo "No .env file found at $ENV_FILE; using environment variables"
fi

# Validate team_id format: team_XXX where XXX is 001-200
if [[ ! "$TEAM_ID" =~ ^team_[0-9]{3}$ ]]; then
  echo "Invalid team_id format: $TEAM_ID (must be team_XXX where XXX is 001-200)" >&2
  exit 1
fi
TEAM_NUM="${TEAM_ID#team_}"
TEAM_NUM_INT=$((10#$TEAM_NUM))  # Force base-10 interpretation
if [[ $TEAM_NUM_INT -lt 1 || $TEAM_NUM_INT -gt 200 ]]; then
  echo "Invalid team_id: $TEAM_ID (number must be between 001 and 200)" >&2
  exit 1
fi

if [[ ! -e "$LOCAL_PATH" ]]; then
  echo "Local path not found: $LOCAL_PATH" >&2
  exit 1
fi

# Read from environment variables (no hardcoded fallbacks)
BUCKET_NAME="${S3_BUCKET_NAME:-}"
S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-}"
ACCESS_KEY="${S3_ACCESS_KEY:-}"
SECRET_KEY="${S3_SECRET_KEY:-}"

# Validate all required variables are set
MISSING_VARS=""
if [[ -z "$BUCKET_NAME" ]]; then
  MISSING_VARS="${MISSING_VARS} S3_BUCKET_NAME"
fi
if [[ -z "$S3_ENDPOINT_URL" ]]; then
  MISSING_VARS="${MISSING_VARS} S3_ENDPOINT_URL"
fi
if [[ -z "$ACCESS_KEY" ]]; then
  MISSING_VARS="${MISSING_VARS} S3_ACCESS_KEY"
fi
if [[ -z "$SECRET_KEY" ]]; then
  MISSING_VARS="${MISSING_VARS} S3_SECRET_KEY"
fi

if [[ -n "$MISSING_VARS" ]]; then
  echo "Missing required environment variables:$MISSING_VARS" >&2
  echo "Set these in .env file or as environment variables." >&2
  exit 1
fi

echo "Using S3 endpoint: $S3_ENDPOINT_URL"
echo "Using S3 bucket: $BUCKET_NAME"

# Remove protocol from endpoint for Host header
S3_HOST="${S3_ENDPOINT_URL#https://}"
S3_HOST="${S3_HOST#http://}"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

TO_UPLOAD=""
BASENAME="$(basename "$LOCAL_PATH")"
if [[ -d "$LOCAL_PATH" ]]; then
  ARCHIVE_NAME="${BASENAME}.tar.gz"
  TO_UPLOAD="$TMP_DIR/$ARCHIVE_NAME"
  echo "Compressing directory to $ARCHIVE_NAME..."
  tar -czf "$TO_UPLOAD" -C "$(dirname "$LOCAL_PATH")" "$BASENAME"
  echo "Compression complete."
  OBJECT_KEY="${TEAM_ID}/${ARCHIVE_NAME}"
else
  TO_UPLOAD="$LOCAL_PATH"
  OBJECT_KEY="${TEAM_ID}/${BASENAME}"
fi

S3_PATH="s3://${BUCKET_NAME}/${OBJECT_KEY}"

# Compute file hash
echo "Computing SHA256 hash..."
if command -v sha256sum >/dev/null 2>&1; then
  SHA256="$(sha256sum "$TO_UPLOAD" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  SHA256="$(shasum -a 256 "$TO_UPLOAD" | awk '{print $1}')"
else
  echo "Could not find sha256sum or shasum." >&2
  exit 1
fi

SIZE_BYTES="$(wc -c < "$TO_UPLOAD" | tr -d ' ')"
echo "SHA256: $SHA256"

# S3 Signature Version 4 signing
REGION="us-east-1"
SERVICE="s3"
DATE_STAMP="$(date -u +%Y%m%d)"
AMZ_DATE="$(date -u +%Y%m%dT%H%M%SZ)"
CONTENT_TYPE="application/octet-stream"

# Canonical request components
HTTP_METHOD="PUT"
CANONICAL_URI="/${BUCKET_NAME}/${OBJECT_KEY}"
CANONICAL_QUERYSTRING=""
CANONICAL_HEADERS="content-type:${CONTENT_TYPE}\nhost:${S3_HOST}\nx-amz-content-sha256:${SHA256}\nx-amz-date:${AMZ_DATE}"
SIGNED_HEADERS="content-type;host;x-amz-content-sha256;x-amz-date"

# Create canonical request
CANONICAL_REQUEST="${HTTP_METHOD}
${CANONICAL_URI}
${CANONICAL_QUERYSTRING}
content-type:${CONTENT_TYPE}
host:${S3_HOST}
x-amz-content-sha256:${SHA256}
x-amz-date:${AMZ_DATE}

${SIGNED_HEADERS}
${SHA256}"

# Create string to sign
ALGORITHM="AWS4-HMAC-SHA256"
CREDENTIAL_SCOPE="${DATE_STAMP}/${REGION}/${SERVICE}/aws4_request"
CANONICAL_REQUEST_HASH="$(printf '%s' "$CANONICAL_REQUEST" | openssl dgst -sha256 | awk '{print $NF}')"
STRING_TO_SIGN="${ALGORITHM}
${AMZ_DATE}
${CREDENTIAL_SCOPE}
${CANONICAL_REQUEST_HASH}"

# Calculate signature
hmac_sha256() {
  printf '%s' "$2" | openssl dgst -sha256 -mac HMAC -macopt "hexkey:$1" | awk '{print $NF}'
}

hmac_sha256_key() {
  printf '%s' "$2" | openssl dgst -sha256 -mac HMAC -macopt "key:$1" | awk '{print $NF}'
}

DATE_KEY="$(hmac_sha256_key "AWS4${SECRET_KEY}" "$DATE_STAMP")"
DATE_REGION_KEY="$(hmac_sha256 "$DATE_KEY" "$REGION")"
DATE_REGION_SERVICE_KEY="$(hmac_sha256 "$DATE_REGION_KEY" "$SERVICE")"
SIGNING_KEY="$(hmac_sha256 "$DATE_REGION_SERVICE_KEY" "aws4_request")"
SIGNATURE="$(hmac_sha256 "$SIGNING_KEY" "$STRING_TO_SIGN")"

# Create authorization header
AUTHORIZATION="${ALGORITHM} Credential=${ACCESS_KEY}/${CREDENTIAL_SCOPE}, SignedHeaders=${SIGNED_HEADERS}, Signature=${SIGNATURE}"

# Upload using curl
UPLOAD_URL="${S3_ENDPOINT_URL}/${BUCKET_NAME}/${OBJECT_KEY}"

echo "Uploading $TO_UPLOAD -> $UPLOAD_URL"
echo "File size: $SIZE_BYTES bytes"
echo "Connecting to $S3_HOST..."
curl -X PUT \
  -H "Content-Type: ${CONTENT_TYPE}" \
  -H "Host: ${S3_HOST}" \
  -H "x-amz-content-sha256: ${SHA256}" \
  -H "x-amz-date: ${AMZ_DATE}" \
  -H "Authorization: ${AUTHORIZATION}" \
  -T "${TO_UPLOAD}" \
  --insecure \
  --progress-bar \
  --connect-timeout 30 \
  "${UPLOAD_URL}"
echo
echo "Upload complete."
echo
echo "Copy this into the 'artifacts' array in hackathon.json:"
echo "This is the URI to your artifact. The destination is where it will be downloaded in the evaluation environment (e.g., project/models). The SHA256 is used to verify integrity during download."
cat <<EOF
{
  "uri": "$S3_PATH",
  "destination": "enter-download-destination-path-here (e.g., project/models)",
  "sha256": "$SHA256",
  "required": true
}
EOF
