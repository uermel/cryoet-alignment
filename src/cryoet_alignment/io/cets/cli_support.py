"""Helpers shared by the CETS converter CLIs (cets-aretomo3, cets-warpm, cets-cryoet-data-portal):
the machine-readable report, source expansion and the common click options.
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import click

from cryoet_alignment.io.cets.config import ConfigError, ConfigFile, Resolver
from cryoet_alignment.io.cets.profile import PROFILE_VERSION


@dataclass
class Gate:
    name: str
    passed: bool
    value: Any = None
    expected: Any = None
    note: str = ""


@dataclass
class SeriesReport:
    name: str
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    gates: List[Gate] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    outputs: Dict[str, str] = field(default_factory=dict)
    hints: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and all(g.passed for g in self.gates)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "provenance": self.provenance,
            "gates": [g.__dict__ for g in self.gates],
            "warnings": self.warnings,
            "dropped": self.dropped,
            "outputs": self.outputs,
            "hints": self.hints,
            "error": self.error,
        }


@dataclass
class Report:
    tool: str
    command: str
    series: List[SeriesReport] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.series)

    def write(self, path: Path) -> Path:
        data = {
            "tool": self.tool,
            "command": self.command,
            "profile": PROFILE_VERSION,
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ok": self.ok,
            "series": [s.as_dict() for s in self.series],
            **self.extra,
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str) + "\n")
        return Path(path)


def echo(msg: str) -> None:
    click.echo(msg, err=False)


def warn(msg: str) -> None:
    click.echo(msg, err=True)


def print_series(sr: SeriesReport) -> None:
    echo(f"== {sr.name}")
    for p in sr.provenance:
        note = f"  ({p['note']})" if p.get("note") else ""
        echo(f"   {p['option']} = {p['value']!r}  [{p['source']}]{note}")
    for g in sr.gates:
        status = "ok " if g.passed else "FAIL"
        detail = "" if g.value is None else f" value={g.value!r}"
        detail += "" if g.expected is None else f" expected={g.expected!r}"
        echo(f"   [{status}] {g.name}{detail}{'  ' + g.note if g.note else ''}")
    for w in sr.warnings:
        warn(f"   {w}" if w.startswith("WARNING:") else f"   WARNING: {w}")
    for d in sr.dropped:
        warn(f"   dropped: {d}")
    for k, v in sr.outputs.items():
        echo(f"   {k}: {v}")
    for h in sr.hints:
        echo(f"   > {h}")
    if sr.error:
        warn(f"   ERROR: {sr.error}")


def expand_sources(tokens: Sequence[str], patterns: Sequence[str]) -> List[Path]:
    """Files as given; directories are globbed with ``patterns`` (sorted, de-duplicated by resolved path)."""
    out: List[Path] = []
    seen = set()
    for tok in tokens:
        p = Path(tok)
        if p.is_dir():
            hits: List[Path] = []
            for pat in patterns:
                hits.extend(sorted(p.glob(pat)))
            if not hits:
                raise click.ClickException(f"{p}: no {', '.join(patterns)} files")
            cands = hits
        elif p.exists():
            cands = [p]
        else:
            raise click.ClickException(f"{tok}: no such file or directory")
        for c in cands:
            key = c.resolve()
            if key not in seen:
                seen.add(key)
                out.append(c)
    return out


def common_options(func):
    """``--config``, ``--overwrite``, ``--fail-fast`` on every command."""
    func = click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), default=None,
                        help="YAML config with overrides (global 'cets', per-command and 'series' sections).")(func)
    func = click.option("--overwrite", is_flag=True, help="Replace existing outputs.")(func)
    func = click.option("--fail-fast", is_flag=True, help="Stop at the first failing series.")(func)
    return func


def selection_options(func):
    func = click.option("--region", "regions", multiple=True, help="Region id(s) to convert (default: all).")(func)
    func = click.option("--alignment", "alignment", default=None,
                        help="Alignment to export when a region has several: instance name or 0-based index.")(func)
    func = click.option("--tomogram", "tomogram", default=None, help="Reference tomogram id when a region has several.")(func)
    return func


def load_config(config_path: Optional[str], known_options: Iterable[str]) -> Optional[ConfigFile]:
    if config_path is None:
        return None
    try:
        return ConfigFile.load(config_path, known_options=known_options)
    except ConfigError as e:
        raise click.ClickException(str(e)) from e


def make_resolver(package: str, command: str, cli: Dict[str, Any], config: Optional[ConfigFile], series: Optional[str], sr: SeriesReport) -> Resolver:
    return Resolver(package, command, cli=cli, config=config, series=series, warn=sr.warnings.append)


def parse_alignment_selector(value: Optional[str]):
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def parse_size(text: Optional[str], ndim: int, name: str):
    """``XxYxZ`` (or a lone ``Z`` for ndim 3 when allowed by the caller)."""
    if text is None:
        return None
    parts = [int(v) for v in str(text).lower().split("x")]
    if len(parts) != ndim:
        raise click.ClickException(f"--{name}: expected {ndim} values like {'x'.join('N' * ndim)}, got {text!r}")
    return tuple(parts)


def finish(report: Report, report_path: Path, fail_fast_hit: bool = False) -> None:
    report.write(report_path)
    echo(f"report: {report_path}")
    if not report.ok:
        sys.exit(1)
