"""
Budget guard for shared free-tier API keys: daily quota + requests-per-minute
throttle + retries on 429/503. Ollama (local) is never limited.

State lives in usage.json at the project root (gitignored) so the count
survives app restarts within the same day.
"""
import json
import os
import time
from datetime import date

from config import DAILY_REQUEST_LIMITS, RPM_LIMITS, USAGE_FILE


def _load():
    if os.path.exists(USAGE_FILE):
        try:
            with open(USAGE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save(state):
    with open(USAGE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def usage_today(provider: str) -> int:
    return _load().get("daily", {}).get(str(date.today()), {}).get(provider, 0)


def check(provider: str):
    """Return (allowed, wait_seconds, reason). allowed=False means daily quota exhausted."""
    if provider == "ollama":
        return True, 0.0, ""
    limit = DAILY_REQUEST_LIMITS.get(provider)
    if limit is not None and usage_today(provider) >= limit:
        return False, 0.0, (
            f"Daily quota for {provider} reached ({limit} requests). "
            f"Switch to a local Ollama model, or raise the limit in config.py."
        )
    rpm = RPM_LIMITS.get(provider)
    if rpm:
        now = time.time()
        recent = [t for t in _load().get("recent", {}).get(provider, []) if now - t < 60]
        if len(recent) >= rpm:
            return True, 60 - (now - recent[0]) + 0.1, ""
    return True, 0.0, ""


def record(provider: str):
    if provider == "ollama":
        return
    state = _load()
    today = str(date.today())
    state.setdefault("daily", {}).setdefault(today, {})
    state["daily"][today][provider] = state["daily"][today].get(provider, 0) + 1
    now = time.time()
    recent = [t for t in state.setdefault("recent", {}).get(provider, []) if now - t < 60]
    recent.append(now)
    state["recent"][provider] = recent
    _save(state)


def guarded_call(provider: str, fn):
    allowed, wait, reason = check(provider)
    if not allowed:
        raise RuntimeError(reason)
    if wait > 0:
        time.sleep(wait)
    record(provider)
    last_error = None
    for attempt in range(3):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            # 429 = rate limited; 503/UNAVAILABLE = transient demand spike.
            if not any(s in msg for s in ("429", "503", "UNAVAILABLE")) and "rate" not in msg.lower():
                raise
            last_error = e
            time.sleep(20 * (attempt + 1))
            record(provider)
    raise last_error
