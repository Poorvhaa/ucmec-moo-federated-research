try:
    from . import algorithms, envs, runner, utils, config
except ImportError:
    try:
        import algorithms, envs, runner, utils, config
    except ImportError:
        pass

__version__ = "0.1.0"

__all__ = [
    "algorithms",
    "envs",
    "runner",
    "utils",
    "config",
]