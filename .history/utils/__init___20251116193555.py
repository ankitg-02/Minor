# Prevent direct execution of package initializer
if __name__ == "__main__":
    raise SystemExit(
        "Do not run this file directly. Run modules from the project root, e.g.:\n"
        "  python -m data_ingestion.fetch_comments"
    )

# Package exports (only executed when imported as a package)
from .config import *
from .logger import *
from .exception import *
from .file_helper import *
from .timer import *

__all__ = ["config", "logger", "exception", "file_helper", "timer"]