"""
F9 (FIX-2): execution-safety helpers for the official tool environment.

When the organisers' real tools arrive, every ToolSpec.fn should forward through
`timed()` so a hanging call degrades into a readable error payload instead of
freezing a rollout batch. `SessionShim` adapts their session object to what this
repo's verifiers read back (`world.call_log`, `world.sent_messages`).

Threads, not processes: tool calls are I/O-shaped (HTTP/RPC to an executor), and
a thread pool costs nothing per call. A timed-out thread cannot be killed in
Python -- but its result is simply abandoned, and the episode continues with a
clean timeout error the model can react to.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .registry import ToolError

# One shared pool, never shut down: a context-managed executor would JOIN the
# hung thread on exit, blocking exactly as long as the hang we are guarding
# against. Threads that outlive their budget are simply abandoned.
_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def timed(fn: Callable[..., Any], *args, timeout_s: float = 30.0,
          tool_name: str = "tool", **kw) -> Any:
    """Run fn(*args, **kw); raise ToolError(kind='timeout') if it exceeds budget."""
    try:
        return _EXECUTOR.submit(fn, *args, **kw).result(timeout=timeout_s)
    except TimeoutError:
        raise ToolError(
            f"{tool_name} did not return within {timeout_s:.0f}s",
            kind="timeout",
            hint="the tool environment was slow or stuck; try again once, "
                 "then continue with a different approach") from None
    except ToolError:
        raise                                   # readable failures pass through
    except Exception as e:                      # environment crash -> payload
        raise ToolError(f"{type(e).__name__}: {e}", kind="internal_error") from None


class SessionShim:
    """
    Wrap an organisers' session so it exposes what verifiers need.

    Required surface after wrapping (matches atr World):
      .call_log       -> list of {name, args, ok, result, ...}
      .sent_messages  -> list of side-effect records ({to, subject, body}, ...)
    If their session already has those attributes it is passed straight through.
    """

    def __init__(self, session: Any | None = None,
                 call_log: list | None = None,
                 sent_messages: list | None = None):
        if session is not None and hasattr(session, "call_log") \
                and hasattr(session, "sent_messages"):
            self._session = session
            self.call_log = session.call_log
            self.sent_messages = session.sent_messages
        else:
            self._session = session
            self.call_log = call_log if call_log is not None else []
            self.sent_messages = sent_messages if sent_messages is not None else []

    def record(self, name: str, args: dict, result: dict, ok: bool,
               latency_ms: float = 0.0) -> None:
        """Append one registry-style entry; call from your forwarding fn."""
        self.call_log.append({"name": name, "args": args, "ok": ok,
                              "result": result, "latency_ms": latency_ms})
