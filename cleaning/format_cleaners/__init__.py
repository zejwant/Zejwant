"""
cleaning/format_cleaners/__init__.py

Format-specific cleaners subpackage initializer.

Features:
- Dynamically imports all format-specific cleaner modules
- Exposes a factory function to retrieve cleaner by file type
"""

import importlib
import pkgutil
from typing import Callable, Optional

# Dynamically import all modules in this package
__all__ = []

for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f".{module_name}", package=__name__)
    __all__.append(module_name)

# Map file types to cleaner functions or classes
_CLEANER_MAP = {
    "csv": "clean_csv",
    "excel": "clean_excel",
    "json": "clean_json",
    "xml": "clean_xml",
    "pdf": "clean_pdf",
    "log": "clean_log",
}

def get_cleaner(file_type: str) -> Optional[Callable]:
    """
    Factory function to get the cleaner function/class for a given file type.

    Args:
        file_type (str): File type (e.g., 'csv', 'excel', 'json', 'xml', 'pdf', 'log')

    Returns:
        Callable: The corresponding cleaner function/class, or None if not found.
    """
    file_type = file_type.lower()
    cleaner_name = _CLEANER_MAP.get(file_type)
    if not cleaner_name:
        return None

    for module_name in __all__:
        module = importlib.import_module(f".{module_name}", package=__name__)
        if hasattr(module, cleaner_name):
            return getattr(module, cleaner_name)
    return None
  
