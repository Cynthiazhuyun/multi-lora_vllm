# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dynamic multi-LoRA serving primitives.

This module ties together three pieces that, together, implement the
"online popularity-aware rebase" architecture described in the project
proposal:

* :class:`ProfileSwitchManager` -- knows about a set of pre-built
  serving profiles (see ``lora_profile_builder.py``). Each profile has
  a fused base model (= base + hot LoRA) and a set of delta adapters
  for the other (cold) LoRAs. The manager spawns / kills vLLM
  subprocesses, one per active profile, and exposes blue-green style
  switching.

* :class:`DynamicLoRARouter` -- the user-facing entrypoint. For each
  request it (a) records the requested adapter into a sliding-window
  popularity tracker, (b) chooses the right model name to send to vLLM
  based on the current hot, and (c) schedules a background switch
  whenever the tracker says the dominant adapter has changed.

* The actual switch policy lives in
  :class:`popularity_tracker.PopularityTracker` and is fully decoupled
  from IO so it can be unit-tested.

Why blue-green at the *process* level instead of in-place GPU weight
mutation? It is the safest, simplest implementation: vLLM never has to
support hot-swapping its base model, and the router only has to know
about HTTP endpoints. The cost is briefly running two vLLM processes
during the switch (acceptable for TinyLlama-1.1B on an A100). For
larger models, a single-process implementation could replace
``ProfileSwitchManager`` without touching the router or the tracker.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import queue
import socket
import subprocess
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from lora_profile_builder import LoraProfile
from popularity_tracker import PopularityTracker, PopularityTrackerConfig

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _pick_free_port(preferred: Optional[int] = None) -> int:
    """Pick a TCP port that's currently free."""
    if preferred is not None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", preferred))
                return preferred
            except OSError:
                pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ----------------------------------------------------------------------------
# vLLM subprocess handle
# ----------------------------------------------------------------------------


@dataclass
class ServerHandle:
    """A running vLLM OpenAI-API server backed by one profile."""

    profile_name: str
    fused_model_dir: Path
    delta_adapter_dirs: dict[str, Path]
    port: int
    process: subprocess.Popen
    log_file_path: Path
    base_model_id: str

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def models_url(self) -> str:
        return f"{self.base_url}/v1/models"

    @property
    def completions_url(self) -> str:
        return f"{self.base_url}/v1/completions"

    def is_alive(self) -> bool:
        return self.process.poll() is None


# ----------------------------------------------------------------------------
# Profile switch manager
# ----------------------------------------------------------------------------


class ProfileSwitchManager:
    """Spawn and switch vLLM subprocesses, one per profile.

    Public API (all thread-safe):

    * :meth:`start_server` -- launch a new vLLM for a profile and wait
      for ``/v1/models`` to respond. Returns the handle.
    * :meth:`stop_server` -- terminate a server gracefully.
    * :meth:`get_active` -- ``(profile_name, handle)`` for the active
      server, or ``(None, None)`` if nothing is active.
    * :meth:`switch_to` -- ensure the named profile is running, mark it
      active, then drain & stop the previous active server in the
      background. Idempotent.
    * :meth:`shutdown` -- stop everything.
    """

    def __init__(
        self,
        profiles: dict[str, LoraProfile],
        *,
        log_dir: str | Path,
        base_model_id: str,
        gpu_memory_utilization: float = 0.4,
        max_loras: int = 4,
        max_lora_rank: int = 256,
        dtype: str = "float16",
        startup_timeout_s: float = 600.0,
        idle_grace_s: float = 60.0,
        extra_server_args: Optional[list[str]] = None,
    ) -> None:
        self._profiles = profiles
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._base_model_id = base_model_id
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_loras = max_loras
        self._max_lora_rank = max_lora_rank
        self._dtype = dtype
        self._startup_timeout_s = startup_timeout_s
        self._idle_grace_s = idle_grace_s
        self._extra_server_args = list(extra_server_args or [])

        self._servers: dict[str, ServerHandle] = {}
        self._active_profile: Optional[str] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def known_profiles(self) -> list[str]:
        return list(self._profiles)

    def get_profile(self, profile_name: str) -> LoraProfile:
        if profile_name not in self._profiles:
            raise KeyError(f"Unknown profile {profile_name!r}; "
                           f"known: {sorted(self._profiles)}")
        return self._profiles[profile_name]

    def get_active(self) -> tuple[Optional[str], Optional[ServerHandle]]:
        with self._lock:
            if self._active_profile is None:
                return None, None
            return self._active_profile, self._servers[self._active_profile]

    def get_running(self, profile_name: str) -> Optional[ServerHandle]:
        with self._lock:
            return self._servers.get(profile_name)

    # ------------------------------------------------------------------

    def _build_server_command(
            self, profile: LoraProfile,
            port: int) -> tuple[list[str], list[str]]:
        """Return ``(server_argv, lora_module_strings_for_logging)``."""
        # vLLM accepts either ``name=path`` or ``name=path,base_model=path``.
        # Use the simple form here.
        lora_modules = [
            f"{cold_name}={delta_path}"
            for cold_name, delta_path in profile.delta_adapter_dirs.items()
        ]

        argv = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(profile.fused_model_dir),
            "--served-model-name",
            self._base_model_id,
            "--port",
            str(port),
            "--dtype",
            self._dtype,
            "--gpu-memory-utilization",
            str(self._gpu_memory_utilization),
        ]
        if lora_modules:
            argv += [
                "--enable-lora",
                "--max-loras",
                str(self._max_loras),
                "--max-lora-rank",
                str(self._max_lora_rank),
                "--lora-modules",
            ]
            argv += lora_modules
        argv += self._extra_server_args
        return argv, lora_modules

    def _wait_for_health(self, handle: ServerHandle) -> None:
        """Block until the vLLM server reports ``/v1/models`` 200 OK."""
        log_q: queue.Queue[tuple[str, str]] = queue.Queue()

        def _drain(stream, name):
            try:
                for raw_line in iter(stream.readline, b""):
                    if not raw_line:
                        break
                    log_q.put((name,
                               raw_line.decode("utf-8",
                                               errors="replace").rstrip()))
            finally:
                stream.close()

        threading.Thread(target=_drain,
                         args=(handle.process.stdout, "STDOUT"),
                         daemon=True).start()
        threading.Thread(target=_drain,
                         args=(handle.process.stderr, "STDERR"),
                         daemon=True).start()

        log_path = handle.log_file_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w")

        deadline = time.time() + self._startup_timeout_s
        last_log_time = time.time()
        last_status = 0.0

        try:
            while True:
                # Drain log queue.
                while True:
                    try:
                        name, line = log_q.get_nowait()
                    except queue.Empty:
                        break
                    log_fh.write(f"[{name}] {line}\n")
                    log_fh.flush()
                    last_log_time = time.time()

                if not handle.is_alive():
                    raise RuntimeError(
                        f"vLLM for profile {handle.profile_name!r} exited "
                        f"with code {handle.process.returncode}; logs: "
                        f"{log_path}")

                try:
                    r = requests.get(handle.models_url, timeout=2)
                    if r.status_code == 200:
                        ids = [m["id"] for m in r.json().get("data", [])]
                        # Validate: base + each delta should be visible.
                        expected = {handle.base_model_id, *
                                    handle.delta_adapter_dirs}
                        if expected.issubset(set(ids)):
                            logger.info(
                                "vLLM server ready for profile=%s on port=%d "
                                "(models=%s)", handle.profile_name,
                                handle.port, ids)
                            return
                        logger.debug(
                            "Server up but missing expected models: have=%s "
                            "want=%s", ids, expected)
                except requests.RequestException:
                    pass

                now = time.time()
                if now > deadline:
                    raise TimeoutError(
                        f"vLLM for profile {handle.profile_name!r} did not "
                        f"become ready within {self._startup_timeout_s:.0f}s; "
                        f"logs: {log_path}")
                if (now - last_log_time) > self._idle_grace_s:
                    raise TimeoutError(
                        f"vLLM for profile {handle.profile_name!r} idle "
                        f"({self._idle_grace_s:.0f}s with no logs) and not "
                        f"ready; logs: {log_path}")
                if now - last_status >= 15:
                    elapsed = int(now - (deadline - self._startup_timeout_s))
                    logger.info(
                        "Waiting for vLLM (profile=%s, port=%d, elapsed=%ds)",
                        handle.profile_name, handle.port, elapsed)
                    last_status = now
                time.sleep(1.5)
        finally:
            # Keep tail of logs streaming in background even after ready.
            threading.Thread(
                target=self._stream_logs_background,
                args=(handle, log_q, log_fh),
                daemon=True,
            ).start()

    def _stream_logs_background(
        self,
        handle: ServerHandle,
        log_q: queue.Queue,
        log_fh,
    ) -> None:
        try:
            while handle.is_alive():
                try:
                    name, line = log_q.get(timeout=1.0)
                except queue.Empty:
                    continue
                try:
                    log_fh.write(f"[{name}] {line}\n")
                    log_fh.flush()
                except ValueError:
                    return
        finally:
            try:
                log_fh.close()
            except Exception:
                pass

    # ------------------------------------------------------------------

    def start_server(self,
                     profile_name: str,
                     *,
                     preferred_port: Optional[int] = None) -> ServerHandle:
        with self._lock:
            existing = self._servers.get(profile_name)
            if existing is not None and existing.is_alive():
                return existing
        profile = self.get_profile(profile_name)
        port = _pick_free_port(preferred_port)
        argv, lora_modules = self._build_server_command(profile, port)
        log_path = self._log_dir / f"vllm_{profile_name}_{port}.log"
        logger.info("Spawning vLLM for profile=%s on port=%d (cmd=%s)",
                    profile_name, port, " ".join(argv))
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "VLLM_LOGGING_LEVEL": "INFO"},
        )
        handle = ServerHandle(
            profile_name=profile_name,
            fused_model_dir=profile.fused_model_dir,
            delta_adapter_dirs=profile.delta_adapter_dirs,
            port=port,
            process=proc,
            log_file_path=log_path,
            base_model_id=self._base_model_id,
        )
        try:
            self._wait_for_health(handle)
        except Exception:
            self._terminate(proc)
            raise

        with self._lock:
            self._servers[profile_name] = handle
            if self._active_profile is None:
                self._active_profile = profile_name
        return handle

    def _terminate(self, process: subprocess.Popen, timeout: float = 15.0):
        try:
            process.terminate()
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass

    def stop_server(self, profile_name: str) -> None:
        with self._lock:
            handle = self._servers.pop(profile_name, None)
            if self._active_profile == profile_name:
                self._active_profile = None
        if handle is not None and handle.is_alive():
            logger.info("Stopping vLLM for profile=%s", profile_name)
            self._terminate(handle.process)

    def shutdown(self) -> None:
        names: list[str]
        with self._lock:
            names = list(self._servers)
        for name in names:
            self.stop_server(name)

    # ------------------------------------------------------------------

    def switch_to(
        self,
        profile_name: str,
        *,
        on_event: Optional[Callable[[str, dict], None]] = None,
    ) -> ServerHandle:
        """Activate ``profile_name`` (start it if needed) and stop the old.

        The previous active server is stopped synchronously *after* the
        new one is healthy, so there is no traffic gap.
        """
        if on_event is None:
            on_event = lambda *_: None  # noqa: E731

        with self._lock:
            old_active = self._active_profile

        if old_active == profile_name:
            on_event("noop_switch", {"profile": profile_name})
            return self._servers[profile_name]

        on_event("switch_begin", {"from": old_active, "to": profile_name})
        new_handle = self.start_server(profile_name)
        with self._lock:
            self._active_profile = profile_name
        on_event("switch_promoted", {
            "from": old_active,
            "to": profile_name,
            "new_port": new_handle.port,
        })
        if old_active is not None:
            self.stop_server(old_active)
            on_event("switch_drained_old", {"old": old_active})
        on_event("switch_complete", {"profile": profile_name})
        return new_handle


# ----------------------------------------------------------------------------
# Dynamic LoRA router
# ----------------------------------------------------------------------------


@dataclass
class RouterEvent:
    """Recorded event for post-hoc analysis."""
    timestamp: float
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutedResponse:
    """Result of one routed request."""
    adapter_id: str
    profile_used: str  # which profile's vLLM served the request
    served_model: str  # what we set the OpenAI ``model`` field to
    routed_to_hot: bool  # True iff served via fused base (no LoRA)
    latency_s: float
    status_code: int
    text: str = ""
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    raw_payload: Optional[dict[str, Any]] = None


class DynamicLoRARouter:
    """Online popularity-aware router for multi-LoRA inference.

    The router is intended to be driven from a single benchmark thread.
    All popularity bookkeeping happens inline; switch decisions are
    *kicked off* synchronously here but the actual blue-green switch
    runs in a background thread so we never block the request path
    waiting for a new vLLM to start.
    """

    def __init__(
        self,
        profiles: dict[str, LoraProfile],
        *,
        switch_manager: ProfileSwitchManager,
        tracker_config: Optional[PopularityTrackerConfig] = None,
        initial_hot: str,
        request_session: Optional[requests.Session] = None,
        request_timeout_s: float = 60.0,
    ) -> None:
        self._profiles = profiles
        self._switch_manager = switch_manager
        self._tracker = PopularityTracker(
            tracker_config or PopularityTrackerConfig(),
            initial_hot=initial_hot,
            initial_time=time.time(),
        )
        self._session = request_session or requests.Session()
        self._timeout_s = request_timeout_s
        self._events: list[RouterEvent] = []
        self._events_lock = threading.Lock()
        self._switch_thread: Optional[threading.Thread] = None
        self._switch_lock = threading.Lock()

        if initial_hot not in profiles:
            raise KeyError(f"initial_hot {initial_hot!r} has no profile")

    @property
    def tracker(self) -> PopularityTracker:
        return self._tracker

    # ------------------------------------------------------------------

    def _record_event(self, kind: str, payload: dict[str, Any]) -> None:
        with self._events_lock:
            self._events.append(
                RouterEvent(timestamp=time.time(), kind=kind, payload=payload))

    def events(self) -> list[RouterEvent]:
        with self._events_lock:
            return [dataclasses.replace(e) for e in self._events]

    # ------------------------------------------------------------------

    def _resolve_routing(
            self, adapter_id: str) -> tuple[ServerHandle, str, bool]:
        """Return ``(server, served_model, routed_to_hot)`` for adapter."""
        active_name, active_handle = self._switch_manager.get_active()
        assert active_handle is not None, "No active server"
        if adapter_id == active_name:
            # Hot path: fused base served as the base model.
            return active_handle, active_handle.base_model_id, True
        # Cold path: ask for the delta adapter that lives in the active
        # profile.
        active_profile = self._switch_manager.get_profile(active_name)
        if adapter_id not in active_profile.delta_adapter_dirs:
            raise KeyError(
                f"No delta adapter for {adapter_id!r} in profile "
                f"{active_name!r}; known cold deltas: "
                f"{sorted(active_profile.delta_adapter_dirs)}")
        return active_handle, adapter_id, False

    def _maybe_trigger_switch(self) -> None:
        target = self._tracker.decide_switch()
        if target is None:
            return
        if target not in self._profiles:
            self._record_event("switch_skipped_no_profile",
                               {"target": target})
            return
        with self._switch_lock:
            if (self._switch_thread is not None
                    and self._switch_thread.is_alive()):
                return
            self._tracker.begin_switch(target)
            self._record_event(
                "switch_decided", {
                    "target": target,
                    "tracker_stats": dataclasses.asdict(self._tracker.stats()),
                })
            self._switch_thread = threading.Thread(
                target=self._run_switch,
                args=(target, ),
                daemon=True,
            )
            self._switch_thread.start()

    def _run_switch(self, target: str) -> None:
        try:
            self._switch_manager.switch_to(
                target,
                on_event=lambda kind, payload:
                    self._record_event(f"profile_{kind}", payload),
            )
            self._tracker.commit_switch(target)
            self._record_event("switch_committed", {"target": target})
        except Exception as exc:
            self._tracker.cancel_switch()
            self._record_event("switch_failed", {
                "target": target,
                "error": repr(exc)
            })
            logger.exception("Switch to profile %s failed.", target)

    # ------------------------------------------------------------------

    def send(
        self,
        adapter_id: str,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.7,
        extra_payload: Optional[dict[str, Any]] = None,
    ) -> RoutedResponse:
        """Send one OpenAI-compatible completion through the router."""
        self._tracker.record(adapter_id)

        try:
            handle, served_model, routed_to_hot = self._resolve_routing(
                adapter_id)
            profile_used = handle.profile_name
        except Exception as exc:
            self._record_event("route_resolve_failed", {
                "adapter_id": adapter_id,
                "error": repr(exc),
            })
            return RoutedResponse(
                adapter_id=adapter_id,
                profile_used="<none>",
                served_model="<none>",
                routed_to_hot=False,
                latency_s=0.0,
                status_code=0,
                error=repr(exc),
            )

        payload: dict[str, Any] = {
            "model": served_model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra_payload:
            payload.update(extra_payload)

        t0 = time.perf_counter()
        try:
            r = self._session.post(handle.completions_url,
                                   json=payload,
                                   timeout=self._timeout_s)
        except Exception as exc:
            latency = time.perf_counter() - t0
            self._maybe_trigger_switch()
            return RoutedResponse(
                adapter_id=adapter_id,
                profile_used=profile_used,
                served_model=served_model,
                routed_to_hot=routed_to_hot,
                latency_s=latency,
                status_code=0,
                error=repr(exc),
            )

        latency = time.perf_counter() - t0
        text = ""
        finish_reason = None
        raw = None
        error = None
        if r.status_code == 200:
            try:
                raw = r.json()
                choice = (raw.get("choices") or [{}])[0]
                text = choice.get("text", "") or ""
                finish_reason = choice.get("finish_reason")
            except Exception as exc:
                error = f"bad_response_body: {exc!r}"
        else:
            error = f"http_{r.status_code}: {r.text[:120]}"

        result = RoutedResponse(
            adapter_id=adapter_id,
            profile_used=profile_used,
            served_model=served_model,
            routed_to_hot=routed_to_hot,
            latency_s=latency,
            status_code=r.status_code,
            text=text,
            finish_reason=finish_reason,
            error=error,
            raw_payload=raw,
        )

        self._maybe_trigger_switch()
        return result

    # ------------------------------------------------------------------

    def wait_for_pending_switch(self, timeout: float = 600.0) -> None:
        thread = self._switch_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)


# ----------------------------------------------------------------------------
# Convenience constructor for benchmarks
# ----------------------------------------------------------------------------


def build_router(
    *,
    profiles: dict[str, LoraProfile],
    base_model_id: str,
    log_dir: str | Path,
    initial_hot: str,
    tracker_config: Optional[PopularityTrackerConfig] = None,
    gpu_memory_utilization: float = 0.4,
    max_loras: int = 4,
    max_lora_rank: int = 256,
    extra_server_args: Optional[Iterable[str]] = None,
) -> DynamicLoRARouter:
    """Build :class:`DynamicLoRARouter` and start the initial profile."""
    sm = ProfileSwitchManager(
        profiles=profiles,
        log_dir=log_dir,
        base_model_id=base_model_id,
        gpu_memory_utilization=gpu_memory_utilization,
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
        extra_server_args=list(extra_server_args or []),
    )
    sm.start_server(initial_hot)
    return DynamicLoRARouter(
        profiles=profiles,
        switch_manager=sm,
        tracker_config=tracker_config,
        initial_hot=initial_hot,
    )
