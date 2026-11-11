"""Summarise a tester's GroundFog.log: runtime, renderer, self-test, GPU time, coarse grid, warnings and errors.

Usage: python -I tools/read_tester_log.py <GroundFog.log> [more logs ...]
Python 3.10+. Exit code: 0 = no self-test failure or error line, 1 = something to look at, 2 = a file could not be read.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

MAX_LISTED = 8   # warnings/errors shown per log; the rest are counted

RUNTIME = re.compile(r"GroundFog (\S+) .*loading on runtime (\S+)")
RENDERER = re.compile(r"weather fog: renderer (.+)$")
SELFTEST_RESULT = re.compile(r"\[SELFTEST\] (PASS|FAIL) (.+)$")
SELFTEST_SUM = re.compile(r"\[SELFTEST\] ---- (\d+) PASS, (\d+) FAIL ----")
SELFTEST_START = re.compile(r"\[SELFTEST\] ---- stage A report")
GPU = re.compile(r"\[SELFTEST\] PASS GPU time : (.+)$|\[SELFTEST\] FAIL GPU time : (.+)$")
WIND_GPU = re.compile(r"\[SELFTEST\] INFO wind GPU time : (.+)$")
VIEWPORT = re.compile(r"\[SELFTEST\] INFO viewport : (.+)$")
COARSE = re.compile(r"coarse terrain: (\d+) of (\d+) cells known, (\d+) failed; coarse pools (on|off)")
RESOURCES = re.compile(r"fog resources created in (.+)$")
LEVEL = re.compile(r"\] \[(warning|warn|error|critical)\] (.+)$")


@dataclass
class Summary:
    version: str = "unknown"
    runtime: str = "unknown"
    renderer: str = "unknown (no weather fog line)"
    passes: int = 0
    fails: int = 0
    selftest_done: bool = False
    failed: list[str] = field(default_factory=list)
    gpu: str = "no sample"
    wind_gpu: str = "no sample"
    viewport: str = "unknown"
    resources: str = "not created (never outdoors?)"
    coarse: str = "no line"
    problems: list[str] = field(default_factory=list)


def summarise(lines: list[str]) -> Summary:
    """Reads one log's lines. The last self-test report and coarse line win (a log can hold several)."""
    s = Summary()
    for line in lines:
        line = line.rstrip("\r\n")
        if SELFTEST_START.search(line):
            s.failed.clear()   # only the last report's failures count
        if m := RUNTIME.search(line):
            s.version, s.runtime = m.group(1), m.group(2)
        elif m := RENDERER.search(line):
            s.renderer = m.group(1)
        elif m := SELFTEST_SUM.search(line):
            s.passes, s.fails, s.selftest_done = int(m.group(1)), int(m.group(2)), True
        elif m := GPU.search(line):
            s.gpu = m.group(1) or m.group(2)
        elif m := WIND_GPU.search(line):
            s.wind_gpu = m.group(1)
        elif m := VIEWPORT.search(line):
            s.viewport = m.group(1)
        elif m := COARSE.search(line):
            s.coarse = f"{m.group(1)} of {m.group(2)} cells known, {m.group(3)} failed, coarse pools {m.group(4)}"
        elif m := RESOURCES.search(line):
            s.resources = m.group(1)
        if (m := SELFTEST_RESULT.search(line)) and m.group(1) == "FAIL":
            s.failed.append(m.group(2))
        if m := LEVEL.search(line):
            s.problems.append(f"{m.group(1)}: {m.group(2)}")
    return s


def report(path: Path, s: Summary) -> list[str]:
    out = [f"== {path}",
           f"version {s.version}, runtime {s.runtime}",
           f"renderer: {s.renderer}",
           f"viewport: {s.viewport}"]
    if s.selftest_done:
        out.append(f"self-test: {s.passes} PASS, {s.fails} FAIL")
    else:
        out.append("self-test: no report (needs about 600 frames outdoors)")
    out += [f"  FAIL {f}" for f in s.failed]
    out += [f"fog GPU: {s.gpu}", f"wind GPU: {s.wind_gpu}", f"resources: {s.resources}", f"coarse grid: {s.coarse}"]
    out.append(f"warnings/errors: {len(s.problems)}")
    out += [f"  {p}" for p in s.problems[:MAX_LISTED]]
    if len(s.problems) > MAX_LISTED:
        out.append(f"  ... and {len(s.problems) - MAX_LISTED} more")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", type=Path, help="GroundFog.log files")
    args = parser.parse_args(argv)
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    code = 0
    for path in args.logs:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            log.error("cannot read %s: %s. Check the path; the log is in Documents\\My Games\\Skyrim Special Edition\\SKSE.", path, e)
            code = 2
            continue
        s = summarise(text.splitlines())
        for line in report(path, s):
            log.info("%s", line)
        if code == 0 and (s.fails or s.failed or any(p.startswith(("error", "critical")) for p in s.problems)):
            code = 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
