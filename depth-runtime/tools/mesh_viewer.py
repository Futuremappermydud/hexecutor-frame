#!/usr/bin/env python3
"""Serve a dependency-free, side-by-side viewer for local ASCII PLY scan meshes."""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "samples/depth-model/python"),
                             str(_source_root / "samples/depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import argparse
import json
import mimetypes
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


REPO = (PATHS.repo / 'runtime')
PAGE = (PATHS.repo / 'depth-runtime/tools/mesh_viewer.py').with_suffix(".html")


def handler_for(root):
    root = root.resolve()

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, data, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == "/":
                return self.send_bytes(PAGE.read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/meshes":
                meshes = []
                for file in root.rglob("*.ply"):
                    if not file.is_file():
                        continue
                    stat = file.stat()
                    meshes.append({"path": file.relative_to(root).as_posix(),
                                   "size": stat.st_size, "modified_ns": str(stat.st_mtime_ns)})
                meshes.sort(key=lambda item: (-int(item["modified_ns"]), item["path"]))
                return self.send_bytes(json.dumps(meshes).encode(), "application/json")
            if url.path == "/api/mesh":
                requested = parse_qs(url.query).get("path", [""])[0]
                file = (root / requested).resolve()
                if not file.is_relative_to(root) or file.suffix.lower() != ".ply" or not file.is_file():
                    return self.send_bytes(b"Mesh not found", "text/plain", 404)
                return self.send_bytes(file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            return self.send_bytes(b"Not found", "text/plain", 404)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=(PATHS.repo / 'depth-runtime/experiments'),
                        help="Directory to scan for PLY files (default: all experiments)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the viewer in the default browser")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"mesh root does not exist: {args.root}")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(args.root))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Mesh viewer: {url}", flush=True)
    print(f"Scanning: {args.root.resolve()}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
