"""
Clippy AdaGrad implementation for multitask learning in recommender systems.
"""

try:
    from .clippyadagrad import ClippyAdagrad, clippy_adagrad
except ImportError:
    # Fallback for direct import
    from clippyadagrad import ClippyAdagrad, clippy_adagrad

__version__ = "1.0.0"
__all__ = ["ClippyAdagrad", "clippy_adagrad"] 