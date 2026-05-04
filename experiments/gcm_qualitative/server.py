"""
Tiny HTTP server in a daemon thread, serving the harness output dir.

The dashboard polls manifest.json; new fragments appear without a page reload.
Default bind 127.0.0.1:8765.
"""
from __future__ import annotations

import http.server
import socketserver
import threading
from pathlib import Path


class _Handler(http.server.SimpleHTTPRequestHandler):
    serve_root: Path = Path.cwd()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.serve_root), **kwargs)

    def log_message(self, format, *args):  # silence default access log
        return

    def end_headers(self):
        # No-cache so manifest polling actually reflects updates.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


class _ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_server(serve_root: Path, port: int = 8765) -> "_ReusableTCPServer":
    """Start an HTTP server on a daemon thread. Returns the server handle."""
    handler_cls = type("Handler", (_Handler,), {"serve_root": serve_root})
    server = _ReusableTCPServer(("127.0.0.1", port), handler_cls)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server
