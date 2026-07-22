"""
models/__init__.py
──────────────────
ORM model package.  Import all models here so Base.metadata knows about them.
"""

from backend.models.user import User  # noqa: F401
from backend.models.dataset import Dataset  # noqa: F401
from backend.models.analysis_run import AnalysisRun  # noqa: F401
