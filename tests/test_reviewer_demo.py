"""Exercise the local UI and model adapter over HTTP with synthetic input."""

from contextlib import contextmanager
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from radcopilot.main import AppConfig, ThreadingHTTPServer, resolve_handler


class PageAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and "src" in attrs:
            self.urls.add(attrs["src"])
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.urls.add(attrs["href"])


class FixtureModel(BaseHTTPRequestHandler):
    def reply(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/api/tags":
            self.send_error(404)
            return
        self.reply({"models": [{"name": "fixture-model"}]})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/api/chat" or payload["messages"][0]["content"] != "Synthetic test input.":
            self.send_error(400)
            return
        self.reply({"message": {"role": "assistant", "content": "Synthetic fixture response."}, "done": True})

    def log_message(self, *_args):
        pass


@contextmanager
def serve(handler):
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/"
        finally:
            server.shutdown()
            thread.join()


class ReviewerDemo(unittest.TestCase):
    def test_ui_assets_and_synthetic_chat(self):
        with TemporaryDirectory() as tmp, serve(FixtureModel) as model:
            config = AppConfig(ollama_url=model.rstrip("/"), data_dir=Path(tmp),
                               log_file=Path(tmp) / "demo.jsonl")
            with serve(resolve_handler(config)) as app:
                with urlopen(app, timeout=5) as response:
                    page = PageAssets()
                    page.feed(response.read().decode())

                self.assertEqual(page.urls, {"./style.css", "./templates.js", "./app.js"})
                for path in sorted(page.urls):
                    with urlopen(urljoin(app, path), timeout=5) as response:
                        self.assertTrue(response.read(), path)

                with urlopen(urljoin(app, "api/tags"), timeout=5) as response:
                    self.assertEqual(json.load(response)["models"][0]["name"], "fixture-model")

                request = Request(
                    urljoin(app, "api/chat"),
                    data=json.dumps({"model": "fixture-model", "stream": False,
                                     "messages": [{"role": "user", "content": "Synthetic test input."}]}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(request, timeout=5) as response:
                    self.assertEqual(json.load(response)["message"]["content"], "Synthetic fixture response.")
