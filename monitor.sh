#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
source venv/bin/activate
python monitor_tui.py
