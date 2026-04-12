"""Pipeline execution - session management and main extraction pipeline"""

from .session import initialize_pipeline_session
from .pipeline import run_main_pipeline, run_baselines

__all__ = [
    "initialize_pipeline_session",
    "run_main_pipeline",
    "run_baselines",
]
