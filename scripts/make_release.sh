#!/usr/bin/env bash
set -euo pipefail
python -m build
python -m twine upload dist/*
