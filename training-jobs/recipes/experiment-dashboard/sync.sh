#!/usr/bin/env bash
# Pull the experiment store down from S3 into ./runs/.
#
# S3 stays the store of record; everything local is a read-only mirror plus
# generated manifests. This script never writes to S3.
#
# S3_URI and working AWS credentials must come from your team — do not guess
# them. See README.md "Before you start". If `aws s3 sync` fails with
# AccessDenied/NoCredentialProviders, stop and fix credentials (aws configure
# or AWS_PROFILE); there is no workaround in this script.
#
# Usage:
#   S3_URI=s3://<bucket>/<prefix> ./sync.sh
#   ./sync.sh s3://<bucket>/<prefix>
set -euo pipefail
cd "$(dirname "$0")"

S3_URI="${S3_URI:-${1:-}}"
if [ -z "$S3_URI" ]; then
  echo "usage: S3_URI=s3://<bucket>/<prefix> ./sync.sh   (ask your team for the URI)" >&2
  exit 1
fi

mkdir -p runs
aws s3 sync "$S3_URI" ./runs --exact-timestamps
echo "synced $(find runs -mindepth 1 -maxdepth 1 -type d | wc -l) run(s) into $(pwd)/runs"
echo "next: S3_URI=$S3_URI python3 build_manifests.py"
