__all__ = ["Optimizer"]
__version__ = "0.1.0"


# delay importing
def Optimizer(*args, **kwargs):
    from .optimize import Optimizer as _Optimizer

    return _Optimizer(*args, **kwargs)
