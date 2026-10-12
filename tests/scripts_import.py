"""Import scripts/<name>.py as a module in tests (scripts/ is not a package)."""
import importlib.util
import sys

from graphrag import config


def load_script(name: str):
    key = f"_script_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, config.path(f"scripts/{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(key, None)                  # no half-initialised module for the next test
            raise
    return sys.modules[key]
