#!/usr/bin/env python3
"""Dependency-light HTTP JSON API for exported final models."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.final_predictor import load_bundle


class Repository:
    def __init__(self, artifact_dir: Path, data_root: Path, mapping: Path, cache_dir: Path):
        self.data_root = data_root
        self.mapping = mapping
        self.cache_dir = cache_dir
        self.models = {}
        for path in sorted(artifact_dir.glob("*.pkl")):
            model = load_bundle(path)
            self.models[(model.crop, model.trait)] = model

    def predict(self, crop: str, trait: str, records: list[dict]) -> list[dict]:
        model = self.models.get((crop, trait))
        if model is None:
            raise KeyError(f"Model not loaded: {crop}/{trait}")
        return model.predict_dataset(self.data_root, records, self.mapping, self.cache_dir)


class Handler(BaseHTTPRequestHandler):
    repository: Repository

    def _send(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send(404, {"error": "not found"})
            return
        models = [{"crop": crop, "trait": trait} for crop, trait in sorted(self.repository.models)]
        self._send(200, {"status": "ok", "models": models})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self._send(404, {"error": "not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(size))
            records = request.get("records", [])
            if not isinstance(records, list) or not records:
                raise ValueError("records must be a non-empty list")
            result = self.repository.predict(str(request["crop"]), str(request["trait"]), records)
            self._send(200, {"predictions": result})
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
            self._send(400, {"error": str(error)})
        except Exception as error:  # keep service response structured
            self._send(500, {"error": f"inference failed: {error}"})

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write("API " + format % args + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/final"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, default=Path("data/interim/environment_id_map.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/interim/inference_genotype_cache"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    Handler.repository = Repository(args.artifact_dir, args.data_root, args.mapping, args.cache_dir)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving {len(Handler.repository.models)} models on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping inference API", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
