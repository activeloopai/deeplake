from typing import Optional, Callable, Dict, Tuple
from types import ModuleType
import importlib
import sys


class ImportMetaHook:
    def __init__(self) -> None:
        self.modules: Dict[str, ModuleType] = dict()
        self.on_import: Dict[str, list] = dict()

    def add(self, fullname: str, on_import: Callable) -> None:
        self.on_import.setdefault(fullname, []).append(on_import)

    def install(self) -> None:
        sys.meta_path.insert(0, self)  # type: ignore

    def uninstall(self) -> None:
        sys.meta_path.remove(self)  # type: ignore

    def find_module(
        self, fullname: str, path: Optional[str] = None
    ) -> Optional["ImportMetaHook"]:
        if fullname in self.on_import:
            return self
        return None

    def load_module(self, fullname: str) -> ModuleType:
        self.uninstall()
        mod = importlib.import_module(fullname)
        self.install()
        self.modules[fullname] = mod
        on_imports = self.on_import.get(fullname)
        if on_imports:
            for f in on_imports:
                f()
        return mod

    def get_modules(self) -> Tuple[str, ...]:
        return tuple(self.modules)

    def get_module(self, module: str) -> ModuleType:
        return self.modules[module]


_IMPORT_HOOK: Optional[ImportMetaHook] = None


def add_import_hook(fullname: str, on_import: Callable) -> None:
    global _IMPORT_HOOK
    if _IMPORT_HOOK is None:
        _IMPORT_HOOK = ImportMetaHook()
        _IMPORT_HOOK.install()
    _IMPORT_HOOK.add(fullname, on_import)
