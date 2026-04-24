try:
    from ._version import __commit_id__, __version__
except ImportError:
    __version__ = "0+unknown"
    __commit_id__ = None

from .pipeline.make_pipeline import make_pipeline

__all__ = [
    "__version__",
    "__commit_id__",
    "make_pipeline",
]
