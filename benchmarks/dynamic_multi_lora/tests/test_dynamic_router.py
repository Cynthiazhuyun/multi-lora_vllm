# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for ``DynamicLoRARouter``.

We test the routing decisions and switch wiring in isolation by
faking out the :class:`ProfileSwitchManager` (we never spawn a real
vLLM) and the HTTP session (we never send real requests). This keeps
the tests sub-second and runnable on a laptop.

Properties under test:

* The hot adapter is routed to the *base model id* (fast path), and
  cold adapters are routed to their delta adapter name (cold path).
* Each ``send`` call records the adapter into the popularity tracker.
* When the tracker decides a switch is due, the router triggers the
  switch manager exactly once and emits the corresponding events.
* Failed HTTP responses don't crash the router (they are surfaced as
  ``RoutedResponse.error``).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))

from dynamic_router import (DynamicLoRARouter, RoutedResponse,  # noqa: E402
                            ServerHandle)
from lora_profile_builder import LoraProfile  # noqa: E402
from popularity_tracker import PopularityTrackerConfig  # noqa: E402


# ----------------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------------


class _FakeProcess:
    returncode = 0

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0


class FakeSwitchManager:
    """Stand-in for :class:`ProfileSwitchManager` that doesn't spawn vLLMs."""

    def __init__(self, profiles: dict[str, LoraProfile],
                 base_model_id: str) -> None:
        self._profiles = profiles
        self._base_model_id = base_model_id
        self._handles: dict[str, ServerHandle] = {}
        self._active: str | None = None
        self.switch_calls: list[str] = []

    def _make_handle(self, profile_name: str) -> ServerHandle:
        profile = self._profiles[profile_name]
        return ServerHandle(
            profile_name=profile_name,
            fused_model_dir=profile.fused_model_dir,
            delta_adapter_dirs=profile.delta_adapter_dirs,
            port=18000 + abs(hash(profile_name)) % 1000,
            process=_FakeProcess(),
            log_file_path=Path("/tmp/_fake_log"),
            base_model_id=self._base_model_id,
        )

    def start_server(self, profile_name: str) -> ServerHandle:
        if profile_name not in self._handles:
            self._handles[profile_name] = self._make_handle(profile_name)
        if self._active is None:
            self._active = profile_name
        return self._handles[profile_name]

    def stop_server(self, profile_name: str) -> None:
        self._handles.pop(profile_name, None)
        if self._active == profile_name:
            self._active = None

    def shutdown(self) -> None:
        for name in list(self._handles):
            self.stop_server(name)

    def get_active(self) -> tuple[str | None, ServerHandle | None]:
        if self._active is None:
            return None, None
        return self._active, self._handles[self._active]

    def get_profile(self, profile_name: str) -> LoraProfile:
        return self._profiles[profile_name]

    def switch_to(self, profile_name: str, *, on_event=None) -> ServerHandle:
        self.switch_calls.append(profile_name)
        if profile_name not in self._handles:
            self._handles[profile_name] = self._make_handle(profile_name)
        old = self._active
        if on_event is not None:
            on_event("switch_begin", {"from": old, "to": profile_name})
        self._active = profile_name
        if on_event is not None:
            on_event("switch_promoted", {"from": old, "to": profile_name})
            on_event("switch_complete", {"profile": profile_name})
        return self._handles[profile_name]


class _FakeResponse:
    def __init__(self, status_code: int = 200, text_body: str = "hello"):
        self.status_code = status_code
        self.text = "" if status_code == 200 else "error body"
        self._json = {
            "choices": [
                {
                    "text": text_body,
                    "finish_reason": "stop",
                }
            ]
        }

    def json(self):
        return self._json


class FakeHTTPSession:
    """Records POSTs and returns canned responses."""

    def __init__(self, status_code: int = 200, latency_s: float = 0.001):
        self.status_code = status_code
        self.latency_s = latency_s
        self.posts: list[dict[str, Any]] = []

    def post(self, url, json=None, timeout=None):
        time.sleep(self.latency_s)
        self.posts.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse(self.status_code, text_body="ok")


def _profile(name: str, deltas: list[str]) -> LoraProfile:
    return LoraProfile(
        hot_name=name,
        fused_model_dir=Path(f"/tmp/profile_{name}/fused_model"),
        delta_adapter_dirs={d: Path(f"/tmp/profile_{name}/deltas/{d}")
                            for d in deltas},
    )


@pytest.fixture
def setup():
    profiles = {
        "A": _profile("A", deltas=["B", "C"]),
        "B": _profile("B", deltas=["A", "C"]),
        "C": _profile("C", deltas=["A", "B"]),
    }
    sm = FakeSwitchManager(profiles, base_model_id="BASE")
    sm.start_server("A")
    return profiles, sm


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


def test_hot_request_routes_to_base_model(setup):
    profiles, sm = setup
    session = FakeHTTPSession()
    router = DynamicLoRARouter(profiles=profiles,
                               switch_manager=sm,
                               initial_hot="A",
                               request_session=session)
    resp = router.send("A", "hello world", max_tokens=4)
    assert resp.status_code == 200
    assert resp.routed_to_hot is True
    assert resp.served_model == "BASE"
    assert session.posts[0]["json"]["model"] == "BASE"


def test_cold_request_routes_to_delta_adapter(setup):
    profiles, sm = setup
    session = FakeHTTPSession()
    router = DynamicLoRARouter(profiles=profiles,
                               switch_manager=sm,
                               initial_hot="A",
                               request_session=session)
    resp = router.send("B", "hello cold", max_tokens=4)
    assert resp.status_code == 200
    assert resp.routed_to_hot is False
    assert resp.served_model == "B"
    assert session.posts[0]["json"]["model"] == "B"


def test_unknown_cold_adapter_surfaces_error(setup):
    profiles, sm = setup
    session = FakeHTTPSession()
    router = DynamicLoRARouter(profiles=profiles,
                               switch_manager=sm,
                               initial_hot="A",
                               request_session=session)
    resp = router.send("not_a_real_adapter", "x")
    assert resp.status_code == 0
    assert resp.error is not None and "No delta adapter" in resp.error
    # Check that the error surfaces as a route_resolve_failed event.
    kinds = {e.kind for e in router.events()}
    assert "route_resolve_failed" in kinds


def test_failed_http_does_not_raise(setup):
    profiles, sm = setup
    session = FakeHTTPSession(status_code=500)
    router = DynamicLoRARouter(profiles=profiles,
                               switch_manager=sm,
                               initial_hot="A",
                               request_session=session)
    resp = router.send("B", "x")
    assert resp.status_code == 500
    assert "http_500" in (resp.error or "")


def test_switch_decision_triggers_manager_once(setup):
    profiles, sm = setup
    session = FakeHTTPSession()
    # Aggressive thresholds + zero cooldown so a uniform B workload
    # triggers a switch quickly.
    cfg = PopularityTrackerConfig(
        window_size=20,
        min_window_size=10,
        merge_threshold=0.5,
        switch_margin=0.0,
        cooldown_sec=0.0,
    )
    router = DynamicLoRARouter(profiles=profiles,
                               switch_manager=sm,
                               initial_hot="A",
                               tracker_config=cfg,
                               request_session=session)
    for _ in range(20):
        router.send("B", "x")
    router.wait_for_pending_switch(timeout=10.0)
    # Switch manager called with target B, exactly once.
    assert sm.switch_calls == ["B"]
    assert router.tracker.current_hot == "B"
    # Even more B traffic should not provoke another switch.
    for _ in range(20):
        router.send("B", "x")
    router.wait_for_pending_switch(timeout=10.0)
    assert sm.switch_calls == ["B"]


def test_switch_under_cooldown_is_blocked(setup):
    profiles, sm = setup
    session = FakeHTTPSession()
    cfg = PopularityTrackerConfig(
        window_size=20,
        min_window_size=10,
        merge_threshold=0.5,
        switch_margin=0.0,
        cooldown_sec=10_000.0,
    )
    router = DynamicLoRARouter(profiles=profiles,
                               switch_manager=sm,
                               initial_hot="A",
                               tracker_config=cfg,
                               request_session=session)
    for _ in range(40):
        router.send("B", "x")
    router.wait_for_pending_switch(timeout=10.0)
    # Cooldown >> elapsed -> no switch.
    assert sm.switch_calls == []
    assert router.tracker.current_hot == "A"


def test_records_into_tracker(setup):
    profiles, sm = setup
    session = FakeHTTPSession()
    router = DynamicLoRARouter(profiles=profiles,
                               switch_manager=sm,
                               initial_hot="A",
                               request_session=session)
    for adapter in ["A", "A", "B", "C", "B"]:
        router.send(adapter, "x")
    stats = router.tracker.stats()
    assert stats.window_size == 5
    assert stats.counts["A"] == 2
    assert stats.counts["B"] == 2
    assert stats.counts["C"] == 1
