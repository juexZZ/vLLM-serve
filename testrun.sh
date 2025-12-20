#!/bin/bash
export SSL_CERT_FILE=/scratch/jz4725/cacert-2025-11-04.pem

export PATH="/ext3/uv:$PATH"

source .venv/bin/activate

uv run python test_mm.py