"""Support for renaming function parameters without breaking existing code."""
import functools
import warnings


def renamed_kwargs(**old_to_new):
    """Decorator that accepts the old keyword names of renamed parameters.

    ``@renamed_kwargs(s='step', l='length')`` lets callers go on passing ``s=`` and ``l=`` by
    keyword.  Each use gives a ``DeprecationWarning`` and the value is passed on under the new
    name.  Passing both the old and the new name is an error.  Positional arguments are not
    affected, because a rename does not change the order of the parameters.

    The decorated function keeps the signature (and docstring) of the original, so help() and
    the documentation show only the new names.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for old, new in old_to_new.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(f"{func.__qualname__.replace('.__init__', '')}() got values for both "
                                        f"'{old}' (the deprecated name) and '{new}'. Use '{new}' only.")
                    warnings.warn(f"{func.__qualname__.replace('.__init__', '')}(): the '{old}' parameter has "
                                  f"been renamed '{new}'. The name '{old}' still works for now, but is "
                                  f"deprecated and will be removed in a future release.",
                                  DeprecationWarning, stacklevel=2)
                    kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)
        return wrapper
    return decorator
