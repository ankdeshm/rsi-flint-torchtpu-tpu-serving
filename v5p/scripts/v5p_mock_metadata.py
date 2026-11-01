#!/usr/bin/env python3
"""Dynamic TPU v5p Single-Host / Sub-Host Mock GCE Metadata Server.

Allows each host of a 4-node GKE `2x2x4` TPU v5p slice (`ct5p-hightpu-4t`) to run
independent `TP=4` (`2,2,1`), `TP=2` (`1,2,1`), or `TP=1` (`1,1,1`) SGLang replicas
by reading `/tmp/v5p_mock_meta_state.json` on each metadata query while forwarding
OAuth token and non-TPU requests to the real `169.254.169.254` GCE metadata server.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

STATE_FILE = Path("/tmp/v5p_mock_meta_state.json")


def _real_meta_get(rel_path: str, default: str = "") -> str:
    try:
        req = urllib.request.Request(
            f"http://169.254.169.254/computeMetadata/v1/{rel_path.lstrip('/')}",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=2.0) as r:
            return r.read().decode("utf-8").strip()
    except Exception:
        return default


REAL_HOSTNAME = _real_meta_get("instance/hostname", "tpu-v5p-node").split(".")[0]
REAL_ZONE = _real_meta_get("instance/zone", "projects/0/zones/us-central1-a").rsplit("/", 1)[-1]


def _load_state() -> dict[str, str]:
    default = {
        "chips_per_host_bounds": "2,2,1",
        "host_bounds": "1,1,1",
        "topology": "2x2x1",
        "accelerator_type": "v5p-8",
        "metrics_ports": "8431,8432,8433,8434",
    }
    try:
        if STATE_FILE.exists():
            default.update(json.loads(STATE_FILE.read_text()))
    except Exception:
        pass
    return default


class V5pMockMetadataHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        return

    def forward_to_real_metadata(self) -> None:
        real_url = f"http://169.254.169.254{self.path}"
        req = urllib.request.Request(real_url)
        for key, val in self.headers.items():
            if key.lower() != "host":
                req.add_header(key, val)
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                self.send_response(response.status)
                for key, val in response.headers.items():
                    self.send_header(key, val)
                self.end_headers()
                self.wfile.write(response.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for key, val in e.headers.items():
                self.send_header(key, val)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def do_GET(self) -> None:
        if self.headers.get("Metadata-Flavor") != "Google":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing Metadata-Flavor header")
            return

        path = urllib.parse.urlparse(self.path).path
        st = _load_state()
        tpu_env_keys = {
            "TPU_WORKER_ID": "0",
            "ALT": "false",
            "CHIPS_PER_HOST_BOUNDS": st["chips_per_host_bounds"],
            "CONSUMER_PROJECT_ID": "mock-tpu-project",
            "ENABLE_ICI_RESILIENCY": "false",
            "ENABLE_IMPROVED_REROUTE_ALLREDUCE_STRATEGY": "false",
            "HOST_BOUNDS": st["host_bounds"],
            "INJECT_SLICE_BUILDER_FAULT": "",
            "NODE_ID": "0",
            "TOPOLOGY": st["topology"],
            "TPU_ACCELERATOR_TYPE": st["accelerator_type"],
            "WRAP": "false,false,false",
            "ZONE": REAL_ZONE,
        }

        response_body: bytes | None = None
        attr = path.rsplit("/attributes/", 1)[-1] if "/instance/attributes/" in path else None
        if attr in tpu_env_keys:
            response_body = tpu_env_keys[attr].encode()
        elif path.endswith("/computeMetadata/v1/instance/attributes/tpu-env"):
            lines = [f"{k}: '{v}'" for k, v in tpu_env_keys.items()]
            lines.append(f"TPU_RUNTIME_METRICS_PORTS: '{st['metrics_ports']}'")
            response_body = ("\n".join(lines) + "\n").encode()
        elif path.endswith("/computeMetadata/v1/instance/attributes/worker-network-endpoints"):
            response_body = REAL_HOSTNAME.encode()
        elif path.endswith("/computeMetadata/v1/instance/attributes/accelerator-type"):
            response_body = st["accelerator_type"].encode()
        elif path.endswith("/computeMetadata/v1/instance/attributes/instance-id"):
            response_body = REAL_HOSTNAME.encode()
        elif path.endswith("/computeMetadata/v1/instance/attributes/agent-worker-number"):
            response_body = b"0"
        elif path.endswith("/computeMetadata/v1/project/project-id"):
            response_body = b"mock-tpu-project"
        elif path.endswith("/computeMetadata/v1/project/numeric-project-id"):
            response_body = b"1234567890"
        elif path.endswith("/computeMetadata/v1/universe/universe-domain") or path.endswith("/universe-domain"):
            response_body = b"googleapis.com"

        if response_body is not None:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(response_body)
        else:
            self.forward_to_real_metadata()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=80)
    args = ap.parse_args()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), V5pMockMetadataHandler)
    print(f"[v5p-mock-meta] Listening on 127.0.0.1:{args.port} (zone={REAL_ZONE}, host={REAL_HOSTNAME})", file=sys.stderr, flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
