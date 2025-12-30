import argparse
import json
import mimetypes
import os
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote

try:
    from qdrant_client import QdrantClient
except ImportError:
    QdrantClient = None

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "web" / "config.json"

DEFAULT_CONFIG = {
    "dataset_root": "datasets/ds2",
    "outputs_root": "outputs",
    "results_root": "results",
    "manifest_name": "retrieval_manifest.json",
    "prefer_rendered": True,
    "default_left_method": None,
    "default_right_method": None,
    "hide_method_select": False,
    "anonymize_methods": False,
    "vibe_cards_collection": "video-vibe_cards",
    "port": 8000,
}


def load_config():
    config = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            config.update(json.load(handle))

    config["dataset_root"] = resolve_root(config["dataset_root"])
    config["outputs_root"] = resolve_root(config["outputs_root"])
    config["results_root"] = resolve_root(config["results_root"])
    return config


def resolve_root(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def safe_join(base, relative):
    base = base.resolve()
    target = (base / relative).resolve()
    if target == base or base in target.parents:
        return target
    raise ValueError(f"Unsafe path traversal: {relative}")


def build_catalog(outputs_root, manifest_name):
    catalog = {"methods": {}, "songs": []}
    if not outputs_root.exists():
        return catalog

    for method_dir in sorted(p for p in outputs_root.iterdir() if p.is_dir()):
        method_name = method_dir.name
        songs = {}
        for song_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            manifest_path = song_dir / manifest_name
            if manifest_path.exists():
                songs[song_dir.name] = str(manifest_path.relative_to(outputs_root))
        if songs:
            catalog["methods"][method_name] = {
                "songs": sorted(songs.keys()),
                "manifests": songs,
            }

    song_set = set()
    for method_info in catalog["methods"].values():
        song_set.update(method_info["songs"])
    catalog["songs"] = sorted(song_set)
    return catalog


def guess_song_audio(dataset_root, song):
    candidates = [
        dataset_root / "songs" / song / f"{song}.mp3",
        dataset_root / "songs" / song / f"{song}.wav",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown"


def ensure_results_file(results_root, username, song, left_method, right_method):
    results_root.mkdir(parents=True, exist_ok=True)
    safe_song = _slugify(song)
    ordered = sorted([left_method, right_method])
    safe_methods = [_slugify(name) for name in ordered]
    safe_user = _slugify(username)
    filename = f"{safe_user}-{safe_methods[0]}-{safe_methods[1]}-{safe_song}.json"
    return results_root / filename


def _collect_missing_vibe_ids(entries):
    missing = set()
    for entry in entries:
        for key in ("video_candidates", "vibe_candidates"):
            for candidate in entry.get(key, []) or []:
                if candidate.get("segment_id") and not candidate.get("vibe_card"):
                    missing.add(candidate["segment_id"])
    return sorted(missing)


def _hydrate_vibe_cards(candidates, vibe_lookup):
    if not candidates:
        return
    for candidate in candidates:
        if not candidate.get("vibe_card"):
            segment_id = candidate.get("segment_id")
            if segment_id and segment_id in vibe_lookup:
                candidate["vibe_card"] = vibe_lookup[segment_id]


class VibeCardStore:
    def __init__(self, db_path: Path, collection: str):
        self.db_path = db_path
        self.collection = collection
        self.client = None
        self.cache: dict[str, str | None] = {}

    def _ensure_client(self) -> bool:
        if self.client is not None:
            return True
        if QdrantClient is None:
            return False
        if not self.db_path.exists():
            return False
        self.client = QdrantClient(path=str(self.db_path))
        return True

    def get_vibe_cards(self, segment_ids: list[str]) -> dict[str, str]:
        if not segment_ids:
            return {}
        if not self._ensure_client():
            return {}

        missing = [seg_id for seg_id in segment_ids if seg_id not in self.cache]
        if missing:
            points = self.client.retrieve(
                collection_name=self.collection,
                ids=missing,
                with_vectors=False,
                with_payload=True,
            )
            for point in points:
                payload = point.payload or {}
                vibe_card = payload.get("vibe_card")
                self.cache[str(point.id)] = vibe_card
            for seg_id in missing:
                self.cache.setdefault(seg_id, None)

        return {
            seg_id: vibe
            for seg_id, vibe in self.cache.items()
            if seg_id in segment_ids and vibe
        }


class MVRequestHandler(BaseHTTPRequestHandler):
    server_version = "MVHumanEval/0.1"

    def do_GET(self):
        path, _, query = self.path.partition("?")

        if path == "/":
            return self.serve_static("index.html")
        if path.startswith("/static/"):
            return self.serve_static(path[len("/static/"):])
        if path.startswith("/assets/"):
            return self.serve_asset(path[len("/assets/"):])
        if path == "/api/catalog":
            return self.handle_catalog()
        if path == "/api/config":
            return self.handle_config()
        if path == "/api/manifest":
            return self.handle_manifest(query)

        self.send_error(404, "Not found")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/results":
            return self.handle_results()
        self.send_error(404, "Not found")

    def serve_static(self, relative_path):
        static_root = ROOT / "web" / "static"
        try:
            target = safe_join(static_root, relative_path)
        except ValueError:
            self.send_error(400, "Invalid path")
            return
        return self.send_file(target)

    def serve_asset(self, relative_path):
        config = self.server.config
        if relative_path.startswith("dataset/"):
            base = config["dataset_root"]
            relative = relative_path[len("dataset/"):]
        elif relative_path.startswith("output/"):
            base = config["outputs_root"]
            relative = relative_path[len("output/"):]
        else:
            self.send_error(400, "Invalid asset path")
            return

        relative = unquote(relative)
        try:
            target = safe_join(base, relative)
        except ValueError:
            self.send_error(400, "Invalid asset path")
            return
        return self.send_file(target)

    def handle_catalog(self):
        config = self.server.config
        catalog = build_catalog(config["outputs_root"], config["manifest_name"])
        self.send_json(catalog)

    def handle_config(self):
        config = self.server.config
        payload = {
            "default_left_method": config.get("default_left_method"),
            "default_right_method": config.get("default_right_method"),
            "hide_method_select": bool(config.get("hide_method_select", False)),
            "anonymize_methods": bool(config.get("anonymize_methods", False)),
        }
        self.send_json(payload)

    def handle_manifest(self, query):
        params = parse_qs(query)
        method = params.get("method", [None])[0]
        song = params.get("song", [None])[0]
        if not method or not song:
            self.send_error(400, "Missing method or song")
            return

        config = self.server.config
        try:
            manifest_path = safe_join(
                config["outputs_root"], os.path.join(method, song, config["manifest_name"])
            )
        except ValueError:
            self.send_error(400, "Invalid method or song")
            return

        if not manifest_path.exists():
            self.send_error(404, "Manifest not found")
            return

        with manifest_path.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)

        prefer_rendered = bool(config.get("prefer_rendered", True))
        outputs_root = config["outputs_root"]
        dataset_root = config["dataset_root"]
        vibe_lookup = self.server.vibe_store.get_vibe_cards(
            _collect_missing_vibe_ids(entries)
        )

        processed = []
        for entry in entries:
            entry_copy = dict(entry)
            audio_path = entry.get("audio_path")
            if audio_path:
                entry_copy["line_audio_url"] = f"/assets/dataset/{audio_path}"

            selected_display_url = None
            selected_display_source = None
            postprocess = entry.get("postprocess") or {}
            selected = entry.get("selected") or {}
            output_path = postprocess.get("output_path")
            if prefer_rendered and output_path:
                output_path_obj = Path(output_path)
                if not output_path_obj.is_absolute():
                    output_path_obj = outputs_root / output_path_obj
                if output_path_obj.exists():
                    try:
                        relative_output = output_path_obj.resolve().relative_to(outputs_root.resolve())
                        selected_display_url = f"/assets/output/{relative_output.as_posix()}"
                        selected_display_source = "rendered"
                    except ValueError:
                        selected_display_url = None

            if not selected_display_url:
                segment_path = selected.get("segment_path")
                if segment_path:
                    selected_display_url = f"/assets/dataset/{segment_path}"
                    selected_display_source = "dataset"

            entry_copy["selected_display_url"] = selected_display_url
            entry_copy["selected_display_source"] = selected_display_source

            _hydrate_vibe_cards(entry_copy.get("video_candidates"), vibe_lookup)
            _hydrate_vibe_cards(entry_copy.get("vibe_candidates"), vibe_lookup)
            processed.append(entry_copy)

        song_audio = guess_song_audio(dataset_root, song)
        song_audio_url = None
        if song_audio:
            try:
                relative_song = song_audio.resolve().relative_to(dataset_root.resolve())
                song_audio_url = f"/assets/dataset/{relative_song.as_posix()}"
            except ValueError:
                song_audio_url = None

        payload = {
            "method": method,
            "song": song,
            "song_audio_url": song_audio_url,
            "entries": processed,
            "assets": {
                "dataset_prefix": "/assets/dataset/",
                "outputs_prefix": "/assets/output/",
            },
        }
        self.send_json(payload)

    def handle_results(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self.send_error(400, "Empty payload")
            return

        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        song = payload.get("song")
        left_method = payload.get("left_method")
        right_method = payload.get("right_method")
        username = payload.get("username")
        record = payload.get("record")
        if not song or not left_method or not right_method or not username or not record:
            self.send_error(400, "Missing required fields")
            return

        config = self.server.config
        results_path = ensure_results_file(
            config["results_root"], username, song, left_method, right_method
        )

        if results_path.exists():
            with results_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        else:
            data = {
                "metadata": {
                    "song": song,
                    "methods": sorted([left_method, right_method]),
                    "username": username,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "dataset_root": str(config["dataset_root"]),
                    "outputs_root": str(config["outputs_root"]),
                },
                "comparisons": [],
            }

        data["metadata"]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        data["comparisons"].append(record)

        with results_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

        self.send_json({"status": "ok", "path": str(results_path)})

    def send_json(self, payload):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_file(self, path):
        if not path.exists() or not path.is_file():
            self.send_error(404, "File not found")
            return

        file_size = path.stat().st_size
        content_type = mimetypes.guess_type(path.as_posix())[0] or "application/octet-stream"

        range_header = self.headers.get("Range")
        if range_header:
            match = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if match:
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else file_size - 1
                end = min(end, file_size - 1)
                if start > end:
                    self.send_error(416, "Requested Range Not Satisfiable")
                    return
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with path.open("rb") as handle:
                    handle.seek(start)
                    self.copy_file(handle, length)
                return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with path.open("rb") as handle:
            self.copy_file(handle, file_size)

    def copy_file(self, handle, remaining):
        chunk_size = 64 * 1024
        while remaining > 0:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                break
            try:
                self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                break
            remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser(description="MV Synthesis Human Eval server")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    if args.port:
        config["port"] = args.port

    server = HTTPServer(("0.0.0.0", config["port"]), MVRequestHandler)
    server.config = config
    server.vibe_store = VibeCardStore(
        config["dataset_root"] / "db", config["vibe_cards_collection"]
    )
    print(f"Serving MV human eval at http://localhost:{config['port']}")
    server.serve_forever()


if __name__ == "__main__":
    main()
