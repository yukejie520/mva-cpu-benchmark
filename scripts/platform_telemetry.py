"""Best-effort, local platform telemetry for benchmark provenance.

All readers return ``None`` when the operating system does not expose the
requested counter.  Missing telemetry is recorded as missing; it is never
replaced by a made-up value.
"""
from __future__ import annotations

import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _number(text: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def cpu_frequency_mhz() -> float | None:
    """Read a current CPU frequency in MHz on Linux or Windows when possible."""
    # Raspberry Pi / Linux sysfs values are normally expressed in kHz.
    for pattern in (
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq",
    ):
        path = Path(pattern)
        try:
            value = _number(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            value = None
        if value is not None:
            return value / 1000.0

    # /proc/cpuinfo reports MHz directly on common ARM and x86 Linux builds.
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        text = ""
    value = _number(next((line for line in text.splitlines()
                          if "cpu MHz" in line or "Cpu MHz" in line), ""))
    if value is not None:
        return value

    # Windows fallback is intentionally a single read, not a continuous log.
    try:
        command = (
            "(Get-Counter '\\Processor Information(_Total)\\% Processor Performance')"
            ".CounterSamples.CookedValue | Measure-Object -Average | "
            "Select-Object -ExpandProperty Average"
        )
        out = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                             capture_output=True, text=True, timeout=10, check=False)
        pct = _number(out.stdout)
        max_out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).MaxClockSpeed"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        maximum = _number(max_out.stdout)
        if pct is not None and maximum is not None:
            return maximum * pct / 100.0
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def cpu_temperature_c() -> float | None:
    """Read the first plausible thermal-zone temperature in degrees Celsius."""
    for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            value = _number(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            value = None
        if value is not None:
            # Linux thermal zones conventionally use millidegrees Celsius.
            if value > 200:
                value /= 1000.0
            if -20.0 <= value <= 120.0:
                return value

    try:
        out = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True,
                             text=True, timeout=10, check=False)
        value = _number(out.stdout)
        if value is not None and -20.0 <= value <= 120.0:
            return value
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def system_info() -> dict[str, str]:
    """Return non-sensitive runtime/platform labels for a run metadata file."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
