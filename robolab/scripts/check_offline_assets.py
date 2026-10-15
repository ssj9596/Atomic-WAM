#!/usr/bin/env python3
"""Check RoboLab USD dependencies without starting Isaac Sim.

The simulator can resolve asset-valued paths embedded in a binary USD layer
after the layer is opened, so a simple grep of the scene source files misses
remote MDL and texture references.  This utility uses Isaac's bundled USD
Python bindings (``pxr.Sdf``) to inspect the task inventory and its complete
local composition graph before a renderer is started.

The checker is intentionally read-only. It takes the evaluation code root from
``--robolab`` and the separately installed benchmark inputs from ``--assets``.
It reports remote URLs and missing local references; ``--strict`` turns either
condition into a non-zero exit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


_BOOTSTRAP_ENV = "ROBOLAB_USD_PREFLIGHT_BOOTSTRAPPED"
_USD_GLOB = "isaacsim/extscache/omni.usd.libs-*"
_URL_RE = re.compile(r"(?:https?|s3)://[^\s@\"'<>),]+")
_MDL_RESOURCE_RE = re.compile(
    r"(?:texture_[23]d|light_profile|bsdf_measurement)\s*\(\s*[\"']([^\"'\r\n]+)[\"']",
    re.IGNORECASE,
)
_ASSET_SUFFIXES = (
    ".usd",
    ".usda",
    ".usdc",
    ".usdz",
    ".mdl",
    ".png",
    ".jpg",
    ".jpeg",
    ".exr",
    ".hdr",
    ".tga",
)
_USD_SUFFIXES = (".usd", ".usda", ".usdc", ".usdz")
_TEXT_SUFFIXES = (".usd", ".usda", ".mdl", ".material", ".mtlx", ".json")

# Kit's built-in MDL modules are resolved from its local installation rather
# than relative to each RoboLab asset.  They are deliberately not treated as
# missing repository files by the preflight checker.  Explicit file paths are
# still checked normally, so this does not hide an absent user asset.
_BUILTIN_MDL_MODULES = {
    "gltf/pbr.mdl",
    "OmniPBR.mdl",
}


def _candidate_usd_libs(root: Path) -> list[Path]:
    """Return likely Isaac USD library directories, in preference order."""

    candidates: list[Path] = []
    for value in (
        os.environ.get("ROBOLAB_USD_LIB"),
        os.environ.get("ISAACSIM_USD_LIB"),
    ):
        if value:
            candidates.append(Path(value))

    # The normal RoboLab venv is the first choice.  Also inspect the active
    # interpreter's site-packages so this works from a copied RoboLab mirror.
    candidates.extend(
        [
            *root.glob(f".venv/lib/python*/site-packages/{_USD_GLOB}"),
            *Path(sys.prefix).glob(f"lib/python*/site-packages/{_USD_GLOB}"),
            *Path("/isaac-sim/").glob(_USD_GLOB),
            *Path("/opt/isaac-sim/").glob(_USD_GLOB),
        ]
    )
    result: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            path = path.resolve()
        except OSError:
            continue
        key = str(path)
        if key not in seen and path.is_dir():
            seen.add(key)
            result.append(path)
    return result


def _load_sdf(root: Path):
    """Import ``pxr.Sdf``, re-executing once with USD library paths if needed."""

    try:
        from pxr import Sdf  # type: ignore

        return Sdf
    except ImportError as first_error:
        if os.environ.get(_BOOTSTRAP_ENV) == "1":
            raise RuntimeError(f"unable to import pxr.Sdf after USD bootstrap: {first_error}") from first_error

        libs = _candidate_usd_libs(root)
        if not libs:
            raise RuntimeError(
                "pxr.Sdf is unavailable and no Isaac USD library directory was found; "
                "set ROBOLAB_USD_LIB to omni.usd.libs-*"
            ) from first_error

        # Dynamic linker search paths must be present before the pxr extension
        # modules are loaded.  Re-exec preserves the exact RoboLab interpreter
        # selected by the caller and avoids mutating the parent launch process.
        lib = libs[0]
        env = os.environ.copy()
        env[_BOOTSTRAP_ENV] = "1"
        env["PYTHONPATH"] = f"{lib}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
        ld_parts = [str(lib / "bin"), str(lib / "lib")]
        if env.get("LD_LIBRARY_PATH"):
            ld_parts.append(env["LD_LIBRARY_PATH"])
        env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_parts)
        script = Path(__file__).resolve()
        try:
            os.execve(sys.executable, [sys.executable, str(script), *sys.argv[1:]], env)
        except OSError as exc:
            raise RuntimeError(f"failed to restart with USD libraries from {lib}: {exc}") from exc
        raise AssertionError("os.execve returned unexpectedly")


def _is_url(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value))


def _looks_like_asset(value: str) -> bool:
    value = value.strip()
    if not value or value in {".", "..", "None", "NULL"}:
        return False
    if _is_url(value):
        return True
    lower = value.lower()
    normalized = value.replace("\\", "/").lstrip("./")
    if normalized in _BUILTIN_MDL_MODULES:
        return False
    return (
        value.startswith((".", "/", "\\"))
        or lower.endswith(_ASSET_SUFFIXES)
        or "/" in value
        or "\\" in value
    )


def _resolve_reference(source: Path, reference: str) -> Path | None:
    """Resolve a local USD/MDL reference relative to its containing layer."""

    reference = reference.strip()
    if not reference or _is_url(reference):
        return None
    if reference.startswith("file:"):
        reference = reference[5:]
    # MDL module identifiers and anonymous layers are not filesystem paths.
    if reference.startswith(("mdl:", "omniverse:", "anon:", "<")):
        return None
    if reference.replace("\\", "/").lstrip("./") in _BUILTIN_MDL_MODULES:
        return None
    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    try:
        return path.resolve()
    except OSError:
        return path


class OfflineAssetChecker:
    def __init__(self, root: Path, sdf: Any):
        self.root = root.resolve()
        self.sdf = sdf
        self.remote: dict[str, set[str]] = defaultdict(set)
        self.missing: dict[str, set[str]] = defaultdict(set)
        self.errors: dict[str, str] = {}
        self.scanned: set[str] = set()
        self._layers: dict[str, Any] = {}

    def _record_remote(self, source: Path, value: str) -> None:
        for url in _URL_RE.findall(value):
            self.remote[url].add(str(source))

    def _record_local(
        self,
        source: Path,
        value: str,
        *,
        recurse: bool = True,
        composition: bool = False,
    ) -> None:
        value = value.strip()
        if (not composition and not _looks_like_asset(value)) or _is_url(value):
            self._record_remote(source, value)
            return
        resolved = _resolve_reference(source, value)
        if resolved is None:
            return
        if not resolved.exists():
            self.missing[str(resolved)].add(f"{source}: {value}")
            return
        if recurse and resolved.suffix.lower() in _TEXT_SUFFIXES:
            self.scan(resolved)

    def _asset_paths(self, layer: Any) -> list[str]:
        """Collect authored asset attribute values from an Sdf layer."""

        values: list[str] = []

        def append_value(value: Any) -> None:
            if isinstance(value, self.sdf.AssetPath):
                if value.path:
                    values.append(value.path)
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    append_value(item)

        def visit(path: Any) -> None:
            spec = layer.GetObjectAtPath(path)
            if not isinstance(spec, self.sdf.AttributeSpec):
                return
            if str(spec.typeName) not in {"asset", "asset[]"}:
                return
            try:
                append_value(spec.default)
            except Exception:
                pass
            try:
                for sample_time in layer.ListTimeSamplesForPath(path):
                    append_value(layer.QueryTimeSample(path, sample_time))
            except Exception:
                pass

        layer.Traverse(self.sdf.Path.absoluteRootPath, visit)
        return values

    def scan(self, path: Path) -> None:
        """Scan one USD/MDL layer and recursively visit local asset refs."""

        try:
            path = path.resolve()
        except OSError:
            pass
        key = str(path)
        if key in self.scanned:
            return
        self.scanned.add(key)
        if not path.exists():
            self.missing[key].add("root input")
            return

        suffix = path.suffix.lower()
        if suffix in _USD_SUFFIXES:
            try:
                layer = self.sdf.Layer.FindOrOpen(str(path))
            except Exception as exc:  # USD can throw on malformed/unsupported layers.
                self.errors[key] = f"open failed: {exc}"
                return
            if not layer:
                self.errors[key] = "open failed (Sdf.Layer.FindOrOpen returned null)"
                return
            self._layers[key] = layer
            try:
                refs = tuple(layer.GetExternalReferences())
            except Exception:
                refs = ()
            for ref in refs:
                self._record_remote(path, ref)
                self._record_local(path, ref, composition=True)
            try:
                asset_paths = self._asset_paths(layer)
            except Exception as exc:
                self.errors[key] = f"asset traversal failed: {exc}"
                asset_paths = []
            for value in asset_paths:
                self._record_remote(path, value)
                if not _is_url(value):
                    self._record_local(path, value)
            return
        elif suffix in _TEXT_SUFFIXES:
            try:
                text = path.read_text(errors="replace")
            except OSError as exc:
                self.errors[key] = f"read failed: {exc}"
                return
        else:
            # Binary image/mesh assets are existence-checked by their parent;
            # parsing them would produce noise and is unnecessary.
            return

        # URLs can be embedded in binary USD asset values or ordinary MDL text.
        self._record_remote(path, text)
        if suffix == ".mdl":
            for match in _MDL_RESOURCE_RE.finditer(text):
                resource = match.group(1).strip()
                self._record_remote(path, resource)
                if not _is_url(resource):
                    self._record_local(path, resource, recurse=False)

    def report(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "scanned": sorted(self.scanned),
            "remote": [
                {"url": url, "sources": sorted(sources)}
                for url, sources in sorted(self.remote.items())
            ],
            "missing": [
                {"path": path, "sources": sorted(sources)}
                for path, sources in sorted(self.missing.items())
            ],
            "errors": [
                {"path": path, "error": error}
                for path, error in sorted(self.errors.items())
            ],
        }


def _load_metadata(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        rows = value.get("tasks") or value.get("rows") or []
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _task_names_from_source(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    names: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            name = row.get("env_name") or row.get("task_name")
            if name:
                names.append(str(name))
    return names


def _find_task_scene(root: Path, task_name: str) -> str | None:
    """Best-effort fallback for a task absent from generated metadata."""

    task_root = root / "robolab" / "tasks"
    candidates = [task_root / task_name, task_root / f"{task_name}.py"]
    stem = Path(task_name).stem
    candidates.extend(task_root.rglob(f"{stem}.py"))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            text = candidate.read_text(errors="replace")
        except OSError:
            continue
        match = re.search(r"import_scene\s*\(\s*[\"']([^\"']+)", text, re.DOTALL)
        if match:
            return match.group(1)
    return None


def _resolve_scene(asset_root: Path, scene: str) -> Path | None:
    path = Path(scene).expanduser()
    if path.is_absolute():
        return path
    scene_root = asset_root / "scenes"
    direct = scene_root / path
    if direct.is_file():
        return direct
    # Match RoboLab's recursive scene resolver while avoiding generated _* dirs.
    matches = [p for p in scene_root.rglob(path.name) if p.is_file() and not any(part.startswith("_") for part in p.relative_to(scene_root).parts)]
    return matches[0] if matches else None


def _selected_rows(
    metadata: list[dict[str, Any]],
    task_set: str,
    names: list[str],
    source_names: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    by_name: dict[str, dict[str, Any]] = {}
    for row in metadata:
        for key in ("task_name", "env_name", "filename"):
            value = row.get(key)
            if value:
                by_name[str(value)] = row
                by_name[Path(str(value)).name] = row
                by_name[Path(str(value)).stem] = row

    if names:
        selected_names = names
    elif task_set in {"all", "simple"}:
        selected_names = [
            str(row.get("task_name"))
            for row in metadata
            if row.get("task_name") and (task_set == "all" or row.get("difficulty_label") == "simple")
        ]
    else:
        selected_names = source_names
    selected_rows = [by_name[name] for name in selected_names if name in by_name]
    unknown = [name for name in selected_names if name not in by_name]
    return selected_rows, unknown


def run_check(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    root = Path(args.robolab).expanduser().resolve()
    if not args.assets:
        raise RuntimeError("ROBOLAB_ASSETS must point to the external RoboLab assets directory")
    asset_root = Path(args.assets).expanduser().resolve()
    if not asset_root.is_dir():
        raise RuntimeError(f"RoboLab assets directory does not exist: {asset_root}")
    sdf = _load_sdf(root)
    metadata_path = Path(args.task_metadata).expanduser() if args.task_metadata else root / "robolab/tasks/_metadata/task_metadata.json"
    tasks_file = Path(args.tasks_file).expanduser() if args.tasks_file else None
    tasks_source = Path(args.tasks_source).expanduser() if args.tasks_source else None
    metadata = _load_metadata(metadata_path)

    names: list[str] = list(args.task)
    if not names and tasks_file and tasks_file.is_file():
        names = [line.strip() for line in tasks_file.read_text(errors="replace").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    source_names = _task_names_from_source(tasks_source)
    rows, unknown = _selected_rows(metadata, args.task_set, names, source_names)

    # A task file can contain a filename/path rather than metadata's task_name.
    scene_names: list[tuple[str, str]] = []
    for row in rows:
        scene = row.get("scene")
        if scene:
            scene_names.append((str(row.get("task_name") or row.get("filename") or "<task>"), str(scene)))
    for name in unknown:
        scene = _find_task_scene(root, name)
        if scene:
            scene_names.append((name, scene))

    checker = OfflineAssetChecker(root, sdf)
    missing_inputs: dict[str, set[str]] = defaultdict(set)
    if not metadata:
        missing_inputs[str(metadata_path)].add("task metadata is missing or invalid")
    if tasks_file and not tasks_file.is_file():
        missing_inputs[str(tasks_file)].add("tasks file does not exist")
    if args.task_set not in {"all", "simple"} and not names and not source_names:
        source_label = str(tasks_source) if tasks_source else "<tasks source>"
        missing_inputs[source_label].add(f"no tasks found for task set {args.task_set}")
    for task_name, scene_name in scene_names:
        scene_path = _resolve_scene(asset_root, scene_name)
        if scene_path is None:
            missing_inputs[scene_name].add(f"task {task_name}")
        else:
            checker.scan(scene_path)

    robot_path = asset_root / "robots" / "franka_robotiq_2f_85_flattened.usd"
    checker.scan(robot_path)
    for name in unknown:
        if not _find_task_scene(root, name):
            missing_inputs[f"task:{name}"].add("not found in metadata or task source")
    for path, sources in missing_inputs.items():
        checker.missing[path].update(sources)

    result = checker.report()
    result.update(
        {
            "task_set": args.task_set,
            "metadata": str(metadata_path),
            "assets": str(asset_root),
            "tasks_file": str(tasks_file) if tasks_file else None,
            "tasks_source": str(tasks_source) if tasks_source else None,
            "task_count": len(scene_names),
            "scene_count": len({scene for _, scene in scene_names}),
            "robot": str(robot_path),
        }
    )
    failed = bool(result["remote"] or result["missing"] or result["errors"])
    return result, 1 if args.strict and failed else 0


def _build_parser() -> argparse.ArgumentParser:
    env_root = os.environ.get("ROBOLAB")
    default_root = env_root or str(Path(__file__).resolve().parents[1])
    default_assets = os.environ.get("ROBOLAB_ASSETS")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robolab", "--root", dest="robolab", default=default_root, help="RoboLab repository root")
    parser.add_argument("--assets", default=default_assets, help="external RoboLab assets directory")
    parser.add_argument("--task-metadata", default=os.environ.get("TASK_METADATA"), help="task_metadata.json path")
    parser.add_argument("--tasks-file", default=os.environ.get("TASKS_FILE") or None, help="one task/environment name per line")
    parser.add_argument("--tasks-source", default=os.environ.get("TASKS_SOURCE") or None, help="episode_results.jsonl task source")
    parser.add_argument("--task-set", default=os.environ.get("TASK_SET", "all"), help="all, simple, or a source-backed set")
    parser.add_argument("--task", nargs="*", default=[], help="explicit task names (overrides task file/set)")
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    parser.add_argument("--strict", action="store_true", help="exit 1 when remote/missing/error findings exist")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        result, status = run_check(args)
    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        else:
            print(f"[robolab-offline-preflight] ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            f"[robolab-offline-preflight] tasks={result['task_count']} scenes={result['scene_count']} "
            f"layers={len(result['scanned'])} remote={len(result['remote'])} "
            f"missing={len(result['missing'])} errors={len(result['errors'])}"
        )
        for item in result["remote"]:
            print(f"[robolab-offline-preflight] REMOTE {item['url']} (source: {', '.join(item['sources'])})")
        for item in result["missing"]:
            print(f"[robolab-offline-preflight] MISSING {item['path']} (source: {', '.join(item['sources'])})")
        for item in result["errors"]:
            print(f"[robolab-offline-preflight] ERROR {item['path']}: {item['error']}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
