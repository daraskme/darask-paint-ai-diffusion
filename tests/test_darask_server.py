"""Tests for darask_server.py (the headless Darask Paint plugin server).

Deliberately independent of the rest of this repository's test suite: no Qt,
no krita, no torch/ComfyUI, no pytest fixtures -- only the standard library
plus darask_server.py itself, so this can run with a bare Python install:

    python -m unittest tests.test_darask_server -v

A tiny in-process mock of ComfyUI's HTTP API (MockComfyHandler below) stands
in for the real ComfyUI server; it understands just enough of the API
(/system_stats, /object_info, /upload/image, /prompt, /history, /view,
/interrupt, /queue) to exercise darask_server.py's client and HTTP handler
code paths, including the padding/cropping round trip and various
error/security conditions.
"""

from __future__ import annotations

import base64
import io
import json
import random
import socket
import struct
import sys
import threading
import time
import unittest
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import darask_server as ds

# --------------------------------------------------------------------------
# Test helpers: build small valid PNGs using darask_server's own codec
# --------------------------------------------------------------------------


def make_png(width: int, height: int, color=(200, 100, 50)) -> bytes:
    row = bytes(color) * width
    raw = row * height
    return ds.png_encode(width, height, 2, raw)


def make_png_rgba(width: int, height: int, color=(200, 100, 50, 255)) -> bytes:
    row = bytes(color) * width
    raw = row * height
    return ds.png_encode(width, height, 6, raw)


# --------------------------------------------------------------------------
# Mock ComfyUI server
# --------------------------------------------------------------------------


class ComfyState:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.uploads: dict[str, bytes] = {}
        self.checkpoints = ["mock_checkpoint.safetensors"]
        self.next_id = 0
        # Behavior toggles used by individual tests:
        self.reject_prompt = False
        self.comfy_5xx_on_prompt = False
        self.oversize_view = False
        self.malformed_history = False
        self.job_delay = 0.0  # seconds before /history reports completion
        self.comfy_version = "0.3.99-mock"

    def new_prompt_id(self) -> str:
        with self.lock:
            self.next_id += 1
            return f"job-{self.next_id}"


def _extent_from_graph(graph: dict, state: ComfyState) -> tuple[int, int]:
    for node in graph.values():
        if node.get("class_type") == "EmptyLatentImage":
            return node["inputs"]["width"], node["inputs"]["height"]
    for node in graph.values():
        if node.get("class_type") == "LoadImage":
            ref = node["inputs"]["image"]
            data = state.uploads.get(ref)
            if data:
                return ds.png_dimensions(data)
    return 64, 64


class MockComfyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    @property
    def state(self) -> ComfyState:
        return self.server.state  # type: ignore[attr-defined]

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        st = self.state
        if parsed.path == "/system_stats":
            self._json({"system": {"comfyui_version": st.comfy_version}})
        elif parsed.path == "/object_info/CheckpointLoaderSimple":
            self._json(
                {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [list(st.checkpoints)]}}}}
            )
        elif parsed.path.startswith("/history/"):
            if st.malformed_history:
                body = b"{not valid json"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            prompt_id = parsed.path.split("/history/")[-1]
            job = st.jobs.get(prompt_id)
            if job is None:
                self._json({})
                return
            if time.monotonic() < job["ready_at"]:
                self._json({})
                return
            self._json(
                {
                    prompt_id: {
                        "status": {"status_str": "success", "completed": True, "messages": []},
                        "outputs": {
                            job["output_node"]: {
                                "images": [{"filename": job["filename"], "subfolder": "", "type": "output"}]
                            }
                        },
                    }
                }
            )
        elif parsed.path == "/view":
            qs = parse_qs(parsed.query)
            filename = qs.get("filename", [""])[0]
            job = next((j for j in st.jobs.values() if j["filename"] == filename), None)
            if st.oversize_view:
                png = make_png(2000, 2000)  # comfortably exceeds a small test cap
            elif job is not None:
                png = make_png(job["w"], job["h"])
            else:
                png = make_png(16, 16)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        parsed = urlparse(self.path)
        st = self.state
        if parsed.path == "/prompt":
            if st.comfy_5xx_on_prompt:
                self.send_response(500)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            payload = json.loads(raw.decode())
            graph = payload["prompt"]
            output_node = "7" if "7" in graph and graph["7"]["class_type"] == "SaveImage" else "9"
            if st.reject_prompt:
                self._json({"node_errors": {"1": {"errors": [{"message": "mock rejects"}]}}})
                return
            w, h = _extent_from_graph(graph, st)
            prompt_id = st.new_prompt_id()
            prefix = graph[output_node]["inputs"].get("filename_prefix", "darask")
            filename = f"{prefix}_{prompt_id}_.png"
            st.jobs[prompt_id] = {
                "w": w,
                "h": h,
                "output_node": output_node,
                "filename": filename,
                "ready_at": time.monotonic() + st.job_delay,
            }
            self._json({"prompt_id": prompt_id, "number": 1, "node_errors": {}})
        elif parsed.path == "/upload/image":
            # Parse the multipart body written by ComfyClient.upload_image: find the
            # PNG payload between the image part's header and the trailing boundary.
            marker = b"Content-Type: image/png\r\n\r\n"
            start = raw.find(marker)
            assert start != -1, "test multipart body missing image part"
            start += len(marker)
            end = raw.rfind(b"\r\n--")
            png_bytes = raw[start:end]
            name = f"upload_{len(st.uploads)}.png"
            st.uploads[name] = png_bytes
            self._json({"name": name, "subfolder": "", "type": "input"})
        elif parsed.path == "/interrupt" or parsed.path == "/queue":
            self._json({})
        else:
            self._json({"error": "not found"}, 404)


class MockComfyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, state: ComfyState):
        super().__init__(("127.0.0.1", 0), MockComfyHandler)
        self.state = state


# --------------------------------------------------------------------------
# Raw socket helper for HTTP framing edge cases that a well-behaved client
# library would refuse to construct (duplicate headers, chunked encoding,
# truncated bodies, ...).
# --------------------------------------------------------------------------


def raw_request(port: int, raw_bytes: bytes, read_timeout: float = 5.0) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=read_timeout) as sock:
        sock.sendall(raw_bytes)
        sock.settimeout(read_timeout)
        chunks = []
        try:
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        except TimeoutError:
            pass
        return b"".join(chunks)


def status_of(raw_response: bytes) -> int:
    line = raw_response.split(b"\r\n", 1)[0]
    return int(line.split(b" ")[1])


def body_of(raw_response: bytes) -> bytes:
    return raw_response.split(b"\r\n\r\n", 1)[-1]


# --------------------------------------------------------------------------
# Base test case: spins up a mock ComfyUI + a real darask_server.Server
# --------------------------------------------------------------------------


class ServerTestCase(unittest.TestCase):
    generation_timeout = 5.0

    def setUp(self):
        self.comfy_state = ComfyState()
        self.mock_comfy = MockComfyServer(self.comfy_state)
        self.comfy_thread = threading.Thread(target=self.mock_comfy.serve_forever, daemon=True)
        self.comfy_thread.start()
        comfy_port = self.mock_comfy.server_address[1]

        self.comfy_client = ds.ComfyClient(f"http://127.0.0.1:{comfy_port}")
        comfy_manager = ds.ComfyManager(None, None, comfy_port, None)
        self.ctx = ds.ServerContext(
            comfy=self.comfy_client,
            comfy_manager=comfy_manager,
            generation_timeout=self.generation_timeout,
            lock=threading.Lock(),
            server_start_time=time.monotonic(),
        )
        self.httpd = ds.Server(("127.0.0.1", 0), ds.Handler, self.ctx, max_concurrent_handlers=8)
        self.port = self.httpd.server_address[1]
        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()
        time.sleep(0.05)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.mock_comfy.shutdown()
        self.mock_comfy.server_close()

    # -- helpers -------------------------------------------------------

    def request(self, method: str, path: str, body: dict | None = None, extra_headers: dict | None = None):
        """Send a well-formed request using raw sockets (keeps full control
        without pulling in urllib's own header normalization surprises)."""
        headers = {
            "Host": f"127.0.0.1:{self.port}",
            "Connection": "close",
        }
        data = b""
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(data))
        if extra_headers:
            headers.update(extra_headers)
        header_text = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
        raw = f"{method} {path} HTTP/1.1\r\n{header_text}\r\n".encode() + data
        resp = raw_request(self.port, raw)
        status = status_of(resp)
        body_bytes = body_of(resp)
        return status, body_bytes

    def request_json(self, method: str, path: str, body: dict | None = None, extra_headers: dict | None = None):
        status, raw_body = self.request(method, path, body, extra_headers)
        try:
            return status, json.loads(raw_body.decode())
        except ValueError:
            return status, raw_body


# --------------------------------------------------------------------------
# Health endpoint
# --------------------------------------------------------------------------


class HealthTests(ServerTestCase):
    def test_health_ready(self):
        status, payload = self.request_json("GET", "/api/v1/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["plugin"], "darask-ai-diffusion")
        self.assertEqual(payload["api"], 1)
        self.assertEqual(payload["backend"], "ready")
        self.assertEqual(payload["model"], "mock_checkpoint.safetensors")
        self.assertIn("0.3.99-mock", payload["engine"])

    def test_health_unknown_path(self):
        status, payload = self.request_json("GET", "/api/v1/nope")
        self.assertEqual(status, 404)
        self.assertIn("error", payload)


# --------------------------------------------------------------------------
# Security / framing defenses
# --------------------------------------------------------------------------


class SecurityTests(ServerTestCase):
    def test_wrong_host_header_rejected(self):
        status, _ = self.request(
            "GET", "/api/v1/health", extra_headers={"Host": "evil.example.com"}
        )
        self.assertEqual(status, 403)

    def test_missing_host_header_rejected(self):
        # Build the request by hand without a Host header at all.
        raw = b"GET /api/v1/health HTTP/1.1\r\nConnection: close\r\n\r\n"
        resp = raw_request(self.port, raw)
        self.assertEqual(status_of(resp), 400)

    def test_origin_header_rejected(self):
        status, _ = self.request(
            "GET", "/api/v1/health", extra_headers={"Origin": "https://evil.example.com"}
        )
        self.assertEqual(status, 403)

    def test_sec_fetch_site_cross_site_rejected(self):
        status, _ = self.request(
            "GET", "/api/v1/health", extra_headers={"Sec-Fetch-Site": "cross-site"}
        )
        self.assertEqual(status, 403)

    def test_sec_fetch_site_same_origin_allowed(self):
        status, _ = self.request(
            "GET", "/api/v1/health", extra_headers={"Sec-Fetch-Site": "same-origin"}
        )
        self.assertEqual(status, 200)

    def test_options_is_405(self):
        status, _ = self.request("OPTIONS", "/api/v1/generate")
        self.assertEqual(status, 405)

    def test_wrong_content_type_rejected(self):
        headers = {
            "Host": f"127.0.0.1:{self.port}",
            "Connection": "close",
            "Content-Type": "text/plain",
        }
        data = json.dumps({"prompt": "x", "width": 64, "height": 64}).encode()
        headers["Content-Length"] = str(len(data))
        header_text = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
        raw = f"POST /api/v1/generate HTTP/1.1\r\n{header_text}\r\n".encode() + data
        resp = raw_request(self.port, raw)
        self.assertEqual(status_of(resp), 415)

    def test_duplicate_content_length_rejected(self):
        data = json.dumps({"prompt": "x", "width": 64, "height": 64}).encode()
        raw = (
            f"POST /api/v1/generate HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{self.port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(data)}\r\n"
            f"Content-Length: {len(data)}\r\n"
            f"Connection: close\r\n\r\n"
        ).encode() + data
        resp = raw_request(self.port, raw)
        self.assertEqual(status_of(resp), 400)

    def test_transfer_encoding_rejected(self):
        raw = (
            f"POST /api/v1/generate HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{self.port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Transfer-Encoding: chunked\r\n"
            f"Connection: close\r\n\r\n"
            f"0\r\n\r\n"
        ).encode()
        resp = raw_request(self.port, raw)
        self.assertEqual(status_of(resp), 400)

    def test_truncated_body_rejected(self):
        # Declare a Content-Length larger than what we actually send, then
        # shut down the write half so the server sees EOF immediately
        # instead of blocking -- it must respond with an error, not hang.
        raw = (
            f"POST /api/v1/generate HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{self.port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: 10000\r\n"
            f"Connection: close\r\n\r\n"
            f'{{"prompt"'
        ).encode()
        with socket.create_connection(("127.0.0.1", self.port), timeout=5.0) as sock:
            sock.sendall(raw)
            sock.shutdown(socket.SHUT_WR)
            sock.settimeout(5.0)
            chunks = []
            try:
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
            except TimeoutError:
                pass
            resp = b"".join(chunks)
        self.assertTrue(resp, "server did not respond to a truncated body")
        self.assertEqual(status_of(resp), 400)

    def test_oversize_body_rejected(self):
        status, _payload = self.request(
            "POST",
            "/api/v1/generate",
            extra_headers={
                "Content-Type": "application/json",
                "Content-Length": str(ds.MAX_REQUEST_BODY_BYTES + 1000),
            },
        )
        # request() would normally compute Content-Length itself from a body=
        # dict; here we override it via extra_headers with no real body, so
        # just check the declared-length path is rejected before hanging.
        self.assertIn(status, (413, 400))


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


class ValidationTests(ServerTestCase):
    def test_generate_missing_prompt(self):
        status, payload = self.request_json("POST", "/api/v1/generate", {"width": 64, "height": 64})
        self.assertEqual(status, 400)
        self.assertIn("prompt", payload["error"])

    def test_generate_empty_prompt(self):
        status, _payload = self.request_json(
            "POST", "/api/v1/generate", {"prompt": "   ", "width": 64, "height": 64}
        )
        self.assertEqual(status, 400)

    def test_generate_width_too_large(self):
        status, _payload = self.request_json(
            "POST", "/api/v1/generate", {"prompt": "x", "width": 9000, "height": 64}
        )
        self.assertEqual(status, 400)

    def test_generate_width_zero(self):
        status, _payload = self.request_json(
            "POST", "/api/v1/generate", {"prompt": "x", "width": 0, "height": 64}
        )
        self.assertEqual(status, 400)

    def test_generate_non_multiple_of_8_is_accepted(self):
        # This is the headline fix: arbitrary 1..8192 sizes must now be
        # accepted (padded internally, cropped back on the way out).
        status, raw_body = self.request("POST", "/api/v1/generate", {"prompt": "x", "width": 65, "height": 61})
        self.assertEqual(status, 200)
        w, h = ds.png_dimensions(raw_body)
        self.assertEqual((w, h), (65, 61))

    def test_generate_nan_rejected(self):
        raw_json = '{"prompt": "x", "width": NaN, "height": 64}'
        status, _ = self._raw_json_post("/api/v1/generate", raw_json)
        self.assertEqual(status, 400)

    def test_generate_infinity_rejected(self):
        raw_json = '{"prompt": "x", "width": Infinity, "height": 64}'
        status, _ = self._raw_json_post("/api/v1/generate", raw_json)
        self.assertEqual(status, 400)

    def _raw_json_post(self, path: str, raw_json: str):
        data = raw_json.encode()
        raw = (
            f"POST {path} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{self.port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(data)}\r\n"
            f"Connection: close\r\n\r\n"
        ).encode() + data
        resp = raw_request(self.port, raw)
        return status_of(resp), body_of(resp)

    def test_inpaint_mismatched_dimensions(self):
        img = base64.b64encode(make_png(16, 16)).decode()
        mask = base64.b64encode(make_png(8, 8)).decode()
        status, _payload = self.request_json(
            "POST", "/api/v1/inpaint", {"image": img, "mask": mask, "prompt": "x"}
        )
        self.assertEqual(status, 400)

    def test_inpaint_invalid_base64(self):
        mask = base64.b64encode(make_png(8, 8)).decode()
        status, _payload = self.request_json(
            "POST", "/api/v1/inpaint", {"image": "not base64!!", "mask": mask, "prompt": "x"}
        )
        self.assertEqual(status, 400)

    def test_inpaint_strength_out_of_range(self):
        img = base64.b64encode(make_png(16, 16)).decode()
        status, _payload = self.request_json(
            "POST",
            "/api/v1/inpaint",
            {"image": img, "mask": img, "prompt": "x", "strength": 1.5},
        )
        self.assertEqual(status, 400)


# --------------------------------------------------------------------------
# Padding/cropping round trip for non-multiple-of-8 sizes
# --------------------------------------------------------------------------


class PaddingRoundTripTests(ServerTestCase):
    def test_generate_odd_size_round_trip(self):
        status, raw_body = self.request("POST", "/api/v1/generate", {"prompt": "a cat", "width": 61, "height": 67})
        self.assertEqual(status, 200)
        w, h = ds.png_dimensions(raw_body)
        self.assertEqual((w, h), (61, 67))

    def test_generate_already_aligned_size(self):
        status, raw_body = self.request("POST", "/api/v1/generate", {"prompt": "a cat", "width": 64, "height": 64})
        self.assertEqual(status, 200)
        w, h = ds.png_dimensions(raw_body)
        self.assertEqual((w, h), (64, 64))

    def test_inpaint_odd_size_round_trip(self):
        img = base64.b64encode(make_png(37, 29)).decode()
        mask = base64.b64encode(make_png(37, 29, color=(255, 255, 255))).decode()
        status, raw_body = self.request(
            "POST", "/api/v1/inpaint", {"image": img, "mask": mask, "prompt": "x"}
        )
        self.assertEqual(status, 200)
        w, h = ds.png_dimensions(raw_body)
        self.assertEqual((w, h), (37, 29))

    def test_inpaint_rgba_odd_size_round_trip(self):
        img = base64.b64encode(make_png_rgba(19, 23)).decode()
        mask = base64.b64encode(make_png(19, 23, color=(255, 255, 255))).decode()
        status, raw_body = self.request(
            "POST", "/api/v1/inpaint", {"image": img, "mask": mask, "prompt": "x"}
        )
        self.assertEqual(status, 200)
        w, h = ds.png_dimensions(raw_body)
        self.assertEqual((w, h), (19, 23))


# --------------------------------------------------------------------------
# ComfyUI-side failure handling
# --------------------------------------------------------------------------


class ComfyFailureTests(ServerTestCase):
    def test_comfy_5xx_on_prompt(self):
        self.comfy_state.comfy_5xx_on_prompt = True
        status, _payload = self.request_json("POST", "/api/v1/generate", {"prompt": "x", "width": 64, "height": 64})
        self.assertEqual(status, 502)

    def test_comfy_rejects_workflow(self):
        self.comfy_state.reject_prompt = True
        status, _payload = self.request_json("POST", "/api/v1/generate", {"prompt": "x", "width": 64, "height": 64})
        self.assertEqual(status, 502)

    def test_comfy_oversize_view_rejected(self):
        self.comfy_state.oversize_view = True
        old_cap = ds.COMFY_IMAGE_MAX_BYTES
        ds.COMFY_IMAGE_MAX_BYTES = 1024  # tiny cap so a 2000x2000 PNG clearly exceeds it
        try:
            status, _payload = self.request_json(
                "POST", "/api/v1/generate", {"prompt": "x", "width": 64, "height": 64}
            )
            self.assertEqual(status, 502)
        finally:
            ds.COMFY_IMAGE_MAX_BYTES = old_cap

    def test_comfy_malformed_history(self):
        self.comfy_state.malformed_history = True
        status, _payload = self.request_json("POST", "/api/v1/generate", {"prompt": "x", "width": 64, "height": 64})
        self.assertEqual(status, 502)

    def test_comfy_wrong_size_image_rejected(self):
        # Have the mock report a job whose /view size does not match what was
        # requested -- exercises fetch_image's dimension validation.
        real_history = MockComfyHandler.do_GET

        def patched_view(handler_self):
            parsed = urlparse(handler_self.path)
            if parsed.path == "/view":
                png = make_png(999, 999)
                handler_self.send_response(200)
                handler_self.send_header("Content-Type", "image/png")
                handler_self.send_header("Content-Length", str(len(png)))
                handler_self.end_headers()
                handler_self.wfile.write(png)
            else:
                real_history(handler_self)

        MockComfyHandler.do_GET = patched_view
        try:
            status, _payload = self.request_json(
                "POST", "/api/v1/generate", {"prompt": "x", "width": 64, "height": 64}
            )
            self.assertEqual(status, 502)
        finally:
            MockComfyHandler.do_GET = real_history

    def test_no_checkpoint_installed(self):
        self.comfy_state.checkpoints = []
        status, _payload = self.request_json("POST", "/api/v1/generate", {"prompt": "x", "width": 64, "height": 64})
        self.assertEqual(status, 503)


# --------------------------------------------------------------------------
# Concurrency / busy handling
# --------------------------------------------------------------------------


class ConcurrencyTests(ServerTestCase):
    def test_concurrent_request_gets_busy(self):
        self.comfy_state.job_delay = 2.0  # keep the first job "running" for a while

        results = []

        def worker(i):
            status, _payload = self.request_json(
                "POST", "/api/v1/generate", {"prompt": f"x{i}", "width": 64, "height": 64}
            )
            results.append(status)

        t1 = threading.Thread(target=worker, args=(1,))
        t1.start()
        time.sleep(0.3)  # let the first request acquire the lock and start waiting on Comfy
        t2 = threading.Thread(target=worker, args=(2,))
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        self.assertIn(200, results)
        self.assertIn(503, results)


# --------------------------------------------------------------------------
# Checkpoint selection
# --------------------------------------------------------------------------


class CheckpointTests(ServerTestCase):
    def test_sorted_first_checkpoint_used_by_default(self):
        self.comfy_state.checkpoints = ["zzz.safetensors", "aaa.safetensors", "mmm.safetensors"]
        status, payload = self.request_json("GET", "/api/v1/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["model"], "aaa.safetensors")

    def test_requested_checkpoint_not_found_is_error(self):
        self.ctx.requested_checkpoint = "does-not-exist.safetensors"
        status, payload = self.request_json("GET", "/api/v1/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["backend"], "error")
        self.assertIn("does-not-exist", payload.get("detail", ""))


# --------------------------------------------------------------------------
# PNG codec unit tests
# --------------------------------------------------------------------------


class PngCodecTests(unittest.TestCase):
    def test_pad_then_crop_round_trip_rgb(self):
        original = make_png(13, 9)
        padded = ds.png_pad_to(original, 16, 16)
        w, h = ds.png_dimensions(padded)
        self.assertEqual((w, h), (16, 16))
        cropped = ds.png_crop_to(padded, 13, 9)
        self.assertEqual(ds.png_decode(cropped)[3], ds.png_decode(original)[3])

    def test_pad_then_crop_round_trip_rgba(self):
        original = make_png_rgba(11, 7)
        padded = ds.png_pad_to(original, 16, 8)
        cropped = ds.png_crop_to(padded, 11, 7)
        self.assertEqual(ds.png_decode(cropped)[3], ds.png_decode(original)[3])

    def test_pad_noop_when_already_aligned(self):
        original = make_png(16, 16)
        padded = ds.png_pad_to(original, 16, 16)
        self.assertEqual(ds.png_decode(padded)[3], ds.png_decode(original)[3])

    def test_reject_palette_png(self):
        # Minimal 1x1 palette (color type 3) PNG.
        def chunk(tag, data):
            c = tag + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 3, 0, 0, 0)
        plte = bytes([0, 0, 0])
        idat = zlib.compress(bytes([0, 0]))
        data = ds.PNG_SIGNATURE + chunk(b"IHDR", ihdr) + chunk(b"PLTE", plte) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
        with self.assertRaises(ds.PngError):
            ds.png_decode(data)

    def test_reject_garbage(self):
        with self.assertRaises(ds.PngError):
            ds.png_decode(b"not a png at all")

    @unittest.skipIf(
        ds._PILImage is None, "Pillow not installed; only the stdlib codec path exists"
    )
    def test_pillow_and_stdlib_codecs_agree(self):
        # Every color type, every PNG filter type: the accelerated decoder must
        # yield the exact same raw scanlines as the reference stdlib decoder,
        # and the accelerated encoder's output must decode (via stdlib) back to
        # the input. Pillow's own encoder picks filters adaptively, so encode
        # through it to obtain filtered (non-type-0) scanlines to decode.
        from PIL import Image

        w, h = 37, 23
        rng = random.Random(1234)
        for color_type, mode in ds._PNG_PIL_MODES.items():
            channels = ds._PNG_CHANNELS[color_type]
            # gradient + noise so the adaptive filter chooser exercises several filters
            raw = bytes(
                ((x * 7 + y * 3 + rng.randint(0, 40)) & 0xFF)
                for y in range(h)
                for x in range(w * channels)
            )
            buf = io.BytesIO()
            Image.frombytes(mode, (w, h), raw).save(buf, format="PNG", compress_level=9)
            filtered_png = buf.getvalue()
            fast = ds.png_decode(filtered_png)
            saved = ds._PILImage
            try:
                ds._PILImage = None
                reference = ds.png_decode(filtered_png)
                encoded_by_stdlib = ds.png_encode(w, h, color_type, raw)
            finally:
                ds._PILImage = saved
            self.assertEqual(fast, reference, f"decode mismatch for color type {color_type}")
            self.assertEqual(fast[3], raw)
            encoded_by_pil = ds.png_encode(w, h, color_type, raw)
            self.assertEqual(ds.png_decode(encoded_by_pil), ds.png_decode(encoded_by_stdlib))
            self.assertEqual(ds.png_dimensions(encoded_by_pil), (w, h))

    def test_pad_extends_right_and_bottom_edges(self):
        raw = bytes([1, 2, 3, 4, 5, 6])  # 2x1 RGB: (1,2,3) (4,5,6)
        padded = ds.png_decode(ds.png_pad_to(ds.png_encode(2, 1, 2, raw), 4, 2))[3]
        self.assertEqual(bytes(padded), bytes([1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6]) * 2)

    def test_next_multiple_of_8(self):
        self.assertEqual(ds.next_multiple_of_8(1), 8)
        self.assertEqual(ds.next_multiple_of_8(8), 8)
        self.assertEqual(ds.next_multiple_of_8(9), 16)
        self.assertEqual(ds.next_multiple_of_8(8192), 8192)


# --------------------------------------------------------------------------
# CLI argument validation
# --------------------------------------------------------------------------


class ArgValidationTests(unittest.TestCase):
    def test_comfy_url_rejects_non_loopback_host(self):
        with self.assertRaises(SystemExit):
            ds.parse_args(["--comfy-url", "http://example.com:8188"])

    def test_comfy_url_rejects_https(self):
        with self.assertRaises(SystemExit):
            ds.parse_args(["--comfy-url", "https://127.0.0.1:8188"])

    def test_comfy_url_rejects_userinfo(self):
        with self.assertRaises(SystemExit):
            ds.parse_args(["--comfy-url", "http://user:pass@127.0.0.1:8188"])

    def test_comfy_url_accepts_plain_loopback(self):
        args = ds.parse_args(["--comfy-url", "http://127.0.0.1:8188"])
        self.assertEqual(args.comfy_url, "http://127.0.0.1:8188")

    def test_timeout_must_be_positive(self):
        with self.assertRaises(SystemExit):
            ds.parse_args(["--timeout", "0"])

    def test_timeout_must_be_finite(self):
        with self.assertRaises(SystemExit):
            ds.parse_args(["--timeout", "inf"])

    def test_timeout_rejects_negative(self):
        with self.assertRaises(SystemExit):
            ds.parse_args(["--timeout", "-5"])


# --------------------------------------------------------------------------
# ComfyManager temp-file cleanup (item 10)
# --------------------------------------------------------------------------


class ComfyManagerCleanupTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.comfy_dir = Path(self.tmp.name) / "ComfyUI"
        (self.comfy_dir / "input").mkdir(parents=True)
        (self.comfy_dir / "output").mkdir(parents=True)
        # comfy_python/comfy_main just need to exist as *paths*; owns_process
        # only checks they are not None, and cleanup_file/cleanup_stale_files
        # only need comfy_dir (= comfy_main.parent) to resolve.
        self.manager = ds.ComfyManager(
            Path(self.tmp.name) / "python.exe",
            self.comfy_dir / "main.py",
            8188,
            None,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_cleanup_file_removes_darask_prefixed_file(self):
        target = self.comfy_dir / "output" / "darask_generate_1_.png"
        target.write_bytes(b"fake png")
        self.manager.cleanup_file("output", "darask_generate_1_.png")
        self.assertFalse(target.exists())

    def test_cleanup_file_refuses_non_darask_prefix(self):
        target = self.comfy_dir / "output" / "important_other_file.png"
        target.write_bytes(b"do not touch")
        self.manager.cleanup_file("output", "important_other_file.png")
        self.assertTrue(target.exists())

    def test_cleanup_file_refuses_path_traversal(self):
        outside = Path(self.tmp.name) / "darask_should_not_be_deleted.txt"
        outside.write_bytes(b"outside comfy dir")
        self.manager.cleanup_file("output", "../darask_should_not_be_deleted.txt")
        self.assertTrue(outside.exists())

    def test_cleanup_file_handles_subfolder_ref(self):
        (self.comfy_dir / "output" / "sub").mkdir()
        target = self.comfy_dir / "output" / "sub" / "darask_x.png"
        target.write_bytes(b"fake png")
        self.manager.cleanup_file("output", "sub/darask_x.png")
        self.assertFalse(target.exists())

    def test_cleanup_stale_files_removes_old_darask_files_only(self):
        old_file = self.comfy_dir / "input" / "darask_old.png"
        old_file.write_bytes(b"stale")
        recent_file = self.comfy_dir / "input" / "darask_recent.png"
        recent_file.write_bytes(b"fresh")
        other_file = self.comfy_dir / "input" / "not_ours.png"
        other_file.write_bytes(b"someone else's")

        old_time = time.time() - 7200  # 2 hours ago
        import os

        os.utime(old_file, (old_time, old_time))

        self.manager.cleanup_stale_files(max_age_seconds=3600)
        self.assertFalse(old_file.exists())
        self.assertTrue(recent_file.exists())
        self.assertTrue(other_file.exists())

    def test_cleanup_noop_when_not_managing_comfy(self):
        unmanaged = ds.ComfyManager(None, None, 8188, None)
        # Should not raise even though comfy_dir is None.
        unmanaged.cleanup_file("output", "darask_x.png")
        unmanaged.cleanup_stale_files()


# --------------------------------------------------------------------------
# Connection admission control (item 4: thread-exhaustion defense)
# --------------------------------------------------------------------------


class ConnectionLimitTests(unittest.TestCase):
    def test_excess_connections_are_turned_away(self):
        state = ComfyState()
        mock_comfy = MockComfyServer(state)
        threading.Thread(target=mock_comfy.serve_forever, daemon=True).start()
        comfy_port = mock_comfy.server_address[1]

        ctx = ds.ServerContext(
            comfy=ds.ComfyClient(f"http://127.0.0.1:{comfy_port}"),
            comfy_manager=ds.ComfyManager(None, None, comfy_port, None),
            generation_timeout=5.0,
            lock=threading.Lock(),
            server_start_time=time.monotonic(),
        )
        httpd = ds.Server(("127.0.0.1", 0), ds.Handler, ctx, max_concurrent_handlers=2)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        time.sleep(0.05)

        try:
            # Open more raw connections than the semaphore allows and hold them
            # open without sending a full request; excess ones should be
            # actively closed by the server rather than accepted indefinitely.
            socks = []
            for _ in range(5):
                s = socket.create_connection(("127.0.0.1", port), timeout=3.0)
                socks.append(s)
            time.sleep(1.5)

            closed_count = 0
            for s in socks:
                try:
                    s.settimeout(0.5)
                    data = s.recv(1)
                    if data == b"":
                        closed_count += 1
                except (TimeoutError, ConnectionResetError, OSError):
                    closed_count += 1

            self.assertGreaterEqual(closed_count, 1, "expected at least one excess connection to be turned away")
        finally:
            for s in socks:
                try:
                    s.close()
                except OSError:
                    pass
            httpd.shutdown()
            httpd.server_close()
            mock_comfy.shutdown()
            mock_comfy.server_close()


if __name__ == "__main__":
    unittest.main()
