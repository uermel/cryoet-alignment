"""Value resolution for the CETS converter CLIs: no silent defaults, one config file.

Every value a command needs resolves through one chain::

    CLI flag  >  --config cets.yaml  >  companion manifest  >  discovered  >  package default

Each resolution is recorded with its source; falling through to a package default emits a WARNING that
names the option and the config key that would set it. Values with no sane default are hard errors
(``Resolver.require``).

Config file (YAML)::

    cets:                      # every package / command
      voltage: 300
    cets-aretomo3:
      to-cets: {mdoc_dir: mdoc}
    series:                    # per tilt series / run
      TS_01: {pix: 1.54}

Keys are option names with underscores; unknown keys are errors; ``series.<id>`` wins over the command
section for that series.
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

SOURCES = ("cli", "config", "companion", "discovered", "default", "absent")

_MISSING = object()


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Resolved:
    name: str
    value: Any
    source: str
    note: str = ""

    def __str__(self) -> str:
        note = f"  ({self.note})" if self.note else ""
        return f"{self.name} = {self.value!r}  [{self.source}]{note}"


@dataclass
class ConfigFile:
    """Parsed ``--config`` file with the section lookup rules."""

    data: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None

    @classmethod
    def load(cls, path: Union[str, Path], known_options: Optional[Iterable[str]] = None) -> "ConfigFile":
        import yaml

        p = Path(path)
        data = yaml.safe_load(p.read_text()) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"{p}: top level must be a mapping")
        cf = cls(data=data, path=p)
        if known_options is not None:
            cf.check_keys(set(known_options))
        return cf

    def check_keys(self, known: set) -> None:
        def _check(section: Dict[str, Any], where: str) -> None:
            for k, v in section.items():
                if isinstance(v, dict):
                    _check(v, f"{where}.{k}")
                elif k not in known:
                    raise ConfigError(f"{self.path}: unknown option {where}.{k!r} (known: {sorted(known)})")

        for top, section in self.data.items():
            if not isinstance(section, dict):
                raise ConfigError(f"{self.path}: section {top!r} must be a mapping")
            _check(section, top)

    def lookup(self, option: str, package: str, command: str, series: Optional[str] = None):
        """``series.<id>`` > ``<package>.<command>`` > ``<package>`` > ``cets``; ``_MISSING`` when absent."""
        if series is not None:
            hit = self.data.get("series", {}).get(series, {})
            if option in hit:
                return hit[option]
        pkg = self.data.get(package, {})
        cmd = pkg.get(command, {}) if isinstance(pkg, dict) else {}
        if isinstance(cmd, dict) and option in cmd:
            return cmd[option]
        if isinstance(pkg, dict) and option in pkg and not isinstance(pkg[option], dict):
            return pkg[option]
        glob = self.data.get("cets", {})
        if option in glob:
            return glob[option]
        return _MISSING


class Resolver:
    """Resolve one command's values and keep the provenance report."""

    def __init__(
        self,
        package: str,
        command: str,
        cli: Optional[Dict[str, Any]] = None,
        config: Optional[ConfigFile] = None,
        series: Optional[str] = None,
        warn: Callable[[str], None] = None,
    ):
        self.package = package
        self.command = command
        self.cli = {k: v for k, v in (cli or {}).items() if v is not None}
        self.config = config
        self.series = series
        self.report: List[Resolved] = []
        self._warn = warn or (lambda msg: warnings.warn(msg, stacklevel=3))

    def config_key(self, option: str) -> str:
        if self.series:
            return f"series.{self.series}.{option}"
        return f"{self.package}.{self.command}.{option}"

    def resolve(
        self,
        option: str,
        *,
        companion: Any = _MISSING,
        discovered: Any = _MISSING,
        default: Any = _MISSING,
        note: str = "",
        convert: Optional[Callable[[Any], Any]] = None,
    ) -> Resolved:
        """Resolve ``option`` through the chain. A package ``default`` is used only as the last resort and
        WARNS; without one, absence is a ``ConfigError`` naming the flag and the config key."""
        if option in self.cli:
            res = Resolved(option, self.cli[option], "cli", note)
        else:
            cfg = self.config.lookup(option, self.package, self.command, self.series) if self.config else _MISSING
            if cfg is not _MISSING:
                res = Resolved(option, cfg, "config", note)
            elif companion is not _MISSING and companion is not None:
                res = Resolved(option, companion, "companion", note)
            elif discovered is not _MISSING and discovered is not None:
                res = Resolved(option, discovered, "discovered", note)
            elif default is not _MISSING:
                res = Resolved(option, default, "default", note)
                self._warn(
                    f"WARNING: {option} defaulted to {default!r}; set it with --{option.replace('_', '-')} "
                    f"or config key {self.config_key(option)}",
                )
            else:
                raise ConfigError(
                    f"{option} is required and has no source: pass --{option.replace('_', '-')} or set config key "
                    f"{self.config_key(option)}",
                )
        if convert is not None and res.value is not None:
            res = Resolved(res.name, convert(res.value), res.source, res.note)
        self.report.append(res)
        return res

    def optional(self, option: str, *, companion: Any = _MISSING, discovered: Any = _MISSING, note: str = "",
                 absent: Any = None, convert: Optional[Callable[[Any], Any]] = None) -> Any:
        """Resolve a value that may legitimately be absent (companion-only metadata, boolean flags): no
        warning, source ``absent`` and value ``absent`` when nothing supplies it."""
        if option in self.cli or (self.config and self.config.lookup(option, self.package, self.command, self.series) is not _MISSING) \
                or (companion is not _MISSING and companion is not None) or (discovered is not _MISSING and discovered is not None):
            return self.resolve(option, companion=companion, discovered=discovered, note=note, convert=convert).value
        self.report.append(Resolved(option, absent, "absent", note))
        return absent

    def require(self, option: str, **kwargs) -> Any:
        kwargs.pop("default", None)
        return self.resolve(option, **kwargs).value

    def value(self, option: str, **kwargs) -> Any:
        return self.resolve(option, **kwargs).value

    def provenance(self) -> List[Dict[str, Any]]:
        return [{"option": r.name, "value": r.value, "source": r.source, "note": r.note} for r in self.report]

    def lines(self) -> List[str]:
        return [str(r) for r in self.report]
