"""The embedder's failure behaviour, which is the part that fails silently.

Every entry point in embeddings.py fails open by design: a missing or broken
endpoint degrades retrieval to lexical-only rather than raising into a turn. That
is the right choice, and it is also why a bug here is invisible — nothing errors,
answers just get quietly worse. So the failure path needs tests more than the
happy path does.

The bug these were written for: the "endpoint is down" latch was a bool that was
only ever set to True, and the reset() that cleared it had no callers anywhere in
the tree. One connection refused — an ollama restart, or a cold model load
exceeding the 8s timeout — therefore disabled dense retrieval for the entire life
of the agent process, silently.

No network is touched here: the endpoint is a stub HTTP server on a loopback port,
or a closed port when a failure is what is being tested.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("numpy")

from plugins.memory.holographic.embeddings import Embedder


class _Stub(BaseHTTPRequestHandler):
    """Answers /api/embed with unit vectors, and records what it was sent."""

    requests: list[dict] = []
    dim = 768

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).requests.append(body)
        count = len(body.get("input") or [])
        vectors = [[1.0] + [0.0] * (self.dim - 1) for _ in range(count)]
        payload = json.dumps({"embeddings": vectors}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args) -> None:  # keep pytest output clean
        return


@pytest.fixture
def stub():
    _Stub.requests = []
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/api/embed", _Stub
    finally:
        server.shutdown()
        server.server_close()


def _closed_port() -> int:
    """A port nothing is listening on, obtained by closing one we just held."""
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_a_transient_outage_does_not_disable_dense_retrieval_forever(monkeypatch):
    """The regression this file exists for.

    With the old permanent latch, `available` never returned True again after a
    single failure, no matter how long the endpoint had been healthy — and since
    nothing called reset(), that meant "until the agent restarts".
    """
    embedder = Embedder(endpoint=f"http://127.0.0.1:{_closed_port()}/api/embed", timeout=0.05)
    embedder.retry_after = 30.0

    clock = {"now": 1000.0}
    monkeypatch.setattr(
        "plugins.memory.holographic.embeddings.time.monotonic", lambda: clock["now"]
    )

    assert embedder.embed(["anything"]) is None, "a closed port must fail, not hang"
    assert embedder.available is False, "the failure must suppress further attempts"

    # Still inside the window: no retry, so one outage costs one timeout rather
    # than one per query. This is the property the latch was added for and it
    # must survive the fix.
    clock["now"] += 29.0
    assert embedder.available is False

    # Past the window: the endpoint is allowed to be healthy again.
    clock["now"] += 2.0
    assert embedder.available is True, "the suppression window must expire"


def test_a_suppressed_embedder_does_not_touch_the_network(stub, monkeypatch):
    """The reason the latch exists at all.

    Without this, a dead endpoint costs one timeout PER QUERY, and since prefetch
    runs before every turn, an ollama outage becomes a per-turn stall. The latch
    trades a window of degraded recall for never paying that.

    (An earlier version of this file asserted that reading `available` does not
    consume the retry. That test could not fail: a self-clearing check and a pure
    one are indistinguishable through this API, so the assertion was theatre. The
    pure check is still the better shape — a getter should not mutate — but it is
    not a behaviour, so it is not pinned as one.)
    """
    endpoint, cls = stub
    embedder = Embedder(endpoint=endpoint, timeout=2.0)
    embedder.retry_after = 60.0
    clock = {"now": 0.0}
    monkeypatch.setattr(
        "plugins.memory.holographic.embeddings.time.monotonic", lambda: clock["now"]
    )

    # Force the latch, then point back at the live stub: if suppression is not
    # honoured, the calls below will reach it.
    embedder.endpoint = f"http://127.0.0.1:{_closed_port()}/api/embed"
    embedder.timeout = 0.05
    assert embedder.embed(["down"]) is None
    embedder.endpoint, embedder.timeout = endpoint, 2.0
    calls = len(cls.requests)

    for i in range(5):
        assert embedder.embed([f"query {i}"]) is None, "suppressed means no answer"
    assert len(cls.requests) == calls, "a suppressed embedder must make no requests"

    # ...and once the window passes, it starts working again on its own.
    clock["now"] += 61.0
    assert embedder.embed(["after the window"]) is not None
    assert len(cls.requests) == calls + 1


def test_a_still_dead_endpoint_relatches_instead_of_retrying_every_query(monkeypatch):
    """Recovery must not become a per-query stall when the endpoint stays down."""
    embedder = Embedder(endpoint=f"http://127.0.0.1:{_closed_port()}/api/embed", timeout=0.05)
    embedder.retry_after = 20.0
    clock = {"now": 0.0}
    monkeypatch.setattr(
        "plugins.memory.holographic.embeddings.time.monotonic", lambda: clock["now"]
    )

    embedder.embed(["first"])
    clock["now"] += 21.0
    assert embedder.available is True

    # The retry fails, so a fresh window opens rather than leaving the endpoint
    # marked healthy and paying a timeout on every subsequent query.
    assert embedder.embed(["second"]) is None
    assert embedder.available is False
    clock["now"] += 19.0
    assert embedder.available is False


def test_a_malformed_response_is_latched_too(stub):
    """A 200 with the wrong shape is a broken endpoint, not a usable answer."""
    endpoint, cls = stub
    cls.dim = 4  # below the 8-dim sanity floor: not a real embedding
    embedder = Embedder(endpoint=endpoint, timeout=2.0)
    assert embedder.embed(["x"]) is None
    assert embedder.available is False, "garbage must suppress, not be trusted"
    cls.dim = 768


def test_the_model_is_asked_to_stay_resident(stub):
    """keep_alive is what stops every-turn prefetch paying a 3s cold load.

    ollama unloads an idle model after ~5 minutes and reloading nomic-embed-text
    costs ~3s, so without this an operator who pauses pays it on their next
    message — on the prefetch that runs before every turn.
    """
    endpoint, cls = stub
    embedder = Embedder(endpoint=endpoint, timeout=2.0)
    assert embedder.embed_one("hello") is not None
    assert cls.requests, "the stub should have been called"
    assert cls.requests[0].get("keep_alive"), "the request must ask ollama to hold the model"


def test_a_successful_call_leaves_no_suppression_behind(stub):
    """A healthy endpoint must not be marked down by a previous instance's luck."""
    endpoint, _cls = stub
    embedder = Embedder(endpoint=endpoint, timeout=2.0)
    assert embedder.available is True
    assert embedder.embed_one("hello") is not None
    assert embedder.available is True


def test_the_content_cache_survives_an_outage(stub, monkeypatch):
    """Text already embedded must keep working while the endpoint is down.

    Facts embedded before an outage should still be searchable: the vectors are in
    the cache, and re-embedding the same text is exactly what the cache exists to
    avoid.
    """
    endpoint, cls = stub
    embedder = Embedder(endpoint=endpoint, timeout=2.0)
    assert embedder.embed_one("a stable fact") is not None
    calls = len(cls.requests)

    embedder.endpoint = f"http://127.0.0.1:{_closed_port()}/api/embed"
    embedder.timeout = 0.05
    assert embedder.embed_one("a stable fact") is not None, "cached text must not need the network"
    assert len(cls.requests) == calls, "a cache hit must not call out at all"
