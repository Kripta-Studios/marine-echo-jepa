"""Serve packaged research artifacts on loopback only."""

import argparse
from pathlib import Path

import uvicorn

from marine_echo.serving.api import create_app

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8765)
args = parser.parse_args()
if not 1 <= args.port <= 65535:
    parser.error("port must be between 1 and 65535")
root = Path(__file__).resolve().parent
uvicorn.run(create_app(root / "artifacts", root / "web"), host="127.0.0.1", port=args.port)
