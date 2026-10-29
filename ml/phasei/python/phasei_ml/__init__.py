"""Utilities for Phase I machine-learning studies.

The package deliberately keeps optional numerical and ROOT dependencies out of
module import time.  Use :mod:`phasei_ml.source` to create deterministic source
tuples.
"""

__all__ = ["generate_source"]


def generate_source(*args, **kwargs):
    """Lazily call :func:`phasei_ml.source.generate_source`."""
    from .source import generate_source as _generate_source

    return _generate_source(*args, **kwargs)
