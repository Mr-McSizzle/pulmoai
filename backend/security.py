"""Local deployment protections. Rate limits are per process; use a proxy for clustered deployments."""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from uuid import uuid4

from fastapi import HTTPException
from starlette.responses import JSONResponse

MAX_REQUEST_BYTES = 65 * 1024 * 1024
DEFAULT_ORIGINS = {"http://127.0.0.1:8000", "http://localhost:8000", "http://localhost:5173", "http://127.0.0.1:5173"}


class ApiSecurityMiddleware:
    def __init__(self, app):
        self.app = app
        self.attempts = defaultdict(deque)
        self.origins = DEFAULT_ORIGINS | {s.strip() for s in os.getenv("PULMOAI_ALLOWED_ORIGINS", "").split(",") if s.strip()}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/api/"):
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        request_id = uuid4().hex

        async def secured_send(message):
            if message["type"] == "http.response.start":
                message["headers"] = list(message.get("headers", [])) + [
                    (b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"), (b"x-request-id", request_id.encode())]
            await send(message)

        async def reject(status, detail):
            await JSONResponse({"detail": detail, "request_id": request_id}, status_code=status)(scope, receive, secured_send)

        origin = headers.get(b"origin", b"").decode("latin1")
        if scope["method"] in {"POST", "PATCH", "DELETE", "PUT"} and origin and origin not in self.origins:
            return await reject(403, "This request origin is not allowed.")
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await reject(400, "Invalid Content-Length.")
        if length > MAX_REQUEST_BYTES:
            return await reject(413, "Request exceeds the 65 MB limit.")
        if scope["path"] in {"/api/auth/login", "/api/auth/register"} and scope["method"] == "POST":
            now = time.monotonic()
            client = (scope.get("client") or ("local", 0))[0]
            key = (client, scope["path"])
            queue = self.attempts[key]
            while queue and queue[0] < now - 60:
                queue.popleft()
            if len(queue) >= 10:
                return await reject(429, "Too many attempts. Try again in one minute.")
            queue.append(now)
            if len(self.attempts) > 2048:
                self.attempts = defaultdict(deque, {k: v for k, v in self.attempts.items() if v and v[-1] >= now - 60})
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_REQUEST_BYTES:
                    raise HTTPException(413, "Request exceeds the 65 MB limit.")
            return message

        await self.app(scope, bounded_receive, secured_send)
