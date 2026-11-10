"""Build and load the object-allocation counter used by intake benches."""

import importlib.util
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

_MODULE = None


def _load():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    src = Path(__file__).with_name("alloccount.c")
    cache = Path(tempfile.gettempdir()) / "lhotse-osspr-alloccount"
    cache.mkdir(exist_ok=True)
    so = cache / f"alloccount.{sys.implementation.cache_tag}.so"
    if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
        include = sysconfig.get_path("include")
        subprocess.check_call(
            [
                "gcc",
                "-shared",
                "-fPIC",
                "-O2",
                f"-I{include}",
                str(src),
                "-o",
                str(so),
            ]
        )
    spec = importlib.util.spec_from_file_location("alloccount", so)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load allocation counter from {so}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE = module
    return module


def count_call(fn):
    """Call ``fn()`` and return ``(result, object_allocations)``."""
    return _load().count_call(fn)
