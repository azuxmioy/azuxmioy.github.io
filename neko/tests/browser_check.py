#!/usr/bin/env python3
"""Dependency-free Chrome DevTools browser checks and screenshots.

Launch Chrome with --headless=new --remote-debugging-port=9222 first, then:
  python3 tests/browser_check.py --url http://127.0.0.1:4173 \
      --screenshot /tmp/neko2-desktop.png \
      --eval '({title: document.title, width: innerWidth})'

The CDP class can also be imported for richer interaction checks. Browser console
errors and uncaught runtime exceptions are recorded and cause a nonzero exit.
"""

import argparse
import base64
import hashlib
import json
import os
import socket
import struct
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


class WebSocket:
    """Small RFC 6455 client for the local Chrome debugging endpoint."""

    def __init__(self, url, timeout=15):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "ws":
            raise ValueError("Only local ws:// endpoints are supported")
        self.socket = socket.create_connection((parsed.hostname, parsed.port or 80), timeout)
        self.buffer = bytearray()
        key = base64.b64encode(os.urandom(16)).decode()
        path = parsed.path + (("?" + parsed.query) if parsed.query else "")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {parsed.netloc}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        self.socket.sendall(request.encode())
        while b"\r\n\r\n" not in self.buffer:
            self.buffer.extend(self.socket.recv(4096))
        headers, remaining = bytes(self.buffer).split(b"\r\n\r\n", 1)
        self.buffer = bytearray(remaining)
        expected = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest())
        if b" 101 " not in headers or expected.lower() not in headers.lower():
            raise RuntimeError(f"WebSocket upgrade rejected: {headers.decode(errors='replace')}")

    def _read(self, size):
        while len(self.buffer) < size:
            part = self.socket.recv(max(4096, size - len(self.buffer)))
            if not part:
                raise EOFError("Chrome closed its debugging socket")
            self.buffer.extend(part)
        result = bytes(self.buffer[:size])
        del self.buffer[:size]
        return result

    def send(self, data, opcode=1):
        payload = data.encode() if isinstance(data, str) else data
        header = bytearray([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.extend(struct.pack("!BH", 0x80 | 126, length))
        else:
            header.extend(struct.pack("!BQ", 0x80 | 127, length))
        mask = os.urandom(4)
        header.extend(mask)
        self.socket.sendall(header + bytes(value ^ mask[index % 4] for index, value in enumerate(payload)))

    def receive(self):
        result = bytearray()
        while True:
            first, second = self._read(2)
            opcode = first & 0x0F
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._read(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read(8))[0]
            mask = self._read(4) if second & 0x80 else None
            payload = self._read(length)
            if mask:
                payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
            if opcode == 8:
                raise EOFError("Chrome closed the WebSocket")
            if opcode == 9:
                self.send(payload, opcode=10)
                continue
            if opcode == 10:
                continue
            result.extend(payload)
            if first & 0x80:
                return result.decode()

    def close(self):
        try:
            self.send(b"", opcode=8)
        except (OSError, EOFError):
            pass
        self.socket.close()


class CDP:
    def __init__(self, endpoint="http://127.0.0.1:9222"):
        self.endpoint = endpoint.rstrip("/")
        request = urllib.request.Request(self.endpoint + "/json/new?about:blank", method="PUT")
        with urllib.request.urlopen(request, timeout=15) as response:
            target = json.load(response)
        self.target_id = target["id"]
        self.ws = WebSocket(target["webSocketDebuggerUrl"])
        self.sequence = 0
        self.events = []
        self.errors = []
        self.call("Page.enable")
        self.call("Network.enable")
        self.call("Network.setCacheDisabled", {"cacheDisabled": True})
        self.call("Runtime.enable")
        self.call("Log.enable")

    def call(self, method, params=None):
        self.sequence += 1
        command_id = self.sequence
        self.ws.send(json.dumps({"id": command_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.ws.receive())
            if "method" in message:
                self.events.append(message)
                event = message["method"]
                details = message.get("params", {})
                if event == "Runtime.exceptionThrown":
                    self.errors.append({"type": "exception", "details": details})
                elif event == "Runtime.consoleAPICalled" and details.get("type") == "error":
                    self.errors.append({"type": "console.error", "details": details})
                elif event == "Log.entryAdded" and details.get("entry", {}).get("level") == "error":
                    self.errors.append({"type": "browser.error", "details": details})
            if message.get("id") == command_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})

    def evaluate(self, expression):
        response = self.call("Runtime.evaluate", {
            "expression": expression, "awaitPromise": True, "returnByValue": True,
            "userGesture": True,
        })
        if "exceptionDetails" in response:
            raise RuntimeError(json.dumps(response["exceptionDetails"], ensure_ascii=False))
        return response.get("result", {}).get("value")

    def wait_for(self, expression, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.evaluate(expression):
                return
            time.sleep(0.1)
        raise TimeoutError(f"Timed out waiting for {expression}")

    def navigate(self, url, width=1440, height=1000, mobile=False):
        self.call("Emulation.setDeviceMetricsOverride", {
            "width": width, "height": height, "deviceScaleFactor": 1,
            "mobile": mobile,
        })
        result = self.call("Page.navigate", {"url": url})
        if result.get("errorText"):
            raise RuntimeError(result["errorText"])
        self.wait_for("document.readyState === 'complete'")
        self.evaluate("document.fonts.ready.then(() => true)")

    def click(self, selector):
        box = self.evaluate(f"(() => {{ const el = document.querySelector({json.dumps(selector)}); if (!el) throw new Error('Element missing'); el.scrollIntoView({{block: 'center', behavior: 'instant'}}); const r = el.getBoundingClientRect(); return {{x: r.x + r.width / 2, y: r.y + r.height / 2}}; }})()")
        self.call("Input.dispatchMouseEvent", {"type": "mousePressed", "button": "left", "clickCount": 1, **box})
        self.call("Input.dispatchMouseEvent", {"type": "mouseReleased", "button": "left", "clickCount": 1, **box})

    def screenshot(self, path, full_page=False):
        params = {"format": "png", "captureBeyondViewport": full_page}
        if full_page:
            size = self.call("Page.getLayoutMetrics")["cssContentSize"]
            params["clip"] = {"x": 0, "y": 0, "width": size["width"], "height": size["height"], "scale": 1}
        result = self.call("Page.captureScreenshot", params)
        Path(path).write_bytes(base64.b64decode(result["data"]))

    def close(self):
        self.ws.close()
        try:
            urllib.request.urlopen(self.endpoint + "/json/close/" + self.target_id, timeout=5).close()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:4173")
    parser.add_argument("--endpoint", default="http://127.0.0.1:9222")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--mobile", action="store_true")
    parser.add_argument("--wait", type=float, default=0.3)
    parser.add_argument("--wait-expression")
    parser.add_argument("--eval", action="append", default=[])
    parser.add_argument("--click", action="append", default=[])
    parser.add_argument("--screenshot")
    parser.add_argument("--full-page", action="store_true")
    args = parser.parse_args()
    browser = CDP(args.endpoint)
    try:
        browser.navigate(args.url, args.width, args.height, args.mobile)
        if args.wait_expression:
            browser.wait_for(args.wait_expression)
        time.sleep(args.wait)
        for selector in args.click:
            browser.click(selector)
        results = [browser.evaluate(expression) for expression in args.eval]
        if args.screenshot:
            browser.screenshot(args.screenshot, args.full_page)
        browser.evaluate("true")
        print(json.dumps({"results": results, "errors": browser.errors, "screenshot": args.screenshot}, ensure_ascii=False, indent=2))
        return 1 if browser.errors else 0
    finally:
        browser.close()


if __name__ == "__main__":
    sys.exit(main())
