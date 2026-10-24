#!/usr/bin/env bash
# macOS / Linux: install pygame (first time) and start the game.
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
python3 main.py
