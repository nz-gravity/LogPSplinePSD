try:
    from ._version import __commit_id__, __version__
except ImportError:
    __version__ = "0+unknown"
    __commit_id__ = None

__all__ = [
    "__version__",
    "__commit_id__",
]
