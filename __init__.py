from .py_uci import UCIEngine, SearchFnType, UCIOption
"""
pyuci package — simple UCI wrapper for python-chess.

Exports:
- UCIEngine: main UCI engine class
- SearchFnType: typing alias for the expected search function signature
- run_engine: convenience helper to create and run an engine
"""


__all__ = ["UCIEngine", "SearchFnType", "__version__", "UCIOption"]

__version__ = "0.1.0"
