"""Local research dashboard: FastAPI backend plus a static Three.js client."""

from .server import create_app, run_dashboard

__all__ = ["create_app", "run_dashboard"]
