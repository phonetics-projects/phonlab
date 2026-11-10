"""Checks of function arguments that are shared by many phonlab functions."""
import functools
import inspect
import numbers


def check_step(step, func_name=None):
    """Raise a `ValueError` unless `step` is a positive number of seconds.

    A step of zero or less (or NaN, or something that is not a number) used to be quietly replaced by
    a default, truncated to a zero-sample hop that makes a library function fail with an obscure error,
    or (in a few functions) accepted and ignored.  Returns `step` so the check can be used in an
    expression.
    """
    if isinstance(step, bool) or not isinstance(step, numbers.Real) or not step > 0:
        where = f"{func_name}(): the" if func_name else "The"
        raise ValueError(f"{where} `step` between analysis frames must be a positive number of seconds, not {step!r}.")
    return step


def step_samples(step, fs):
    """The number of samples in a `step` (seconds) at the sampling rate `fs`: the nearest whole number.

    Every function that moves an analysis window along the audio by `step` seconds uses this, so they
    all use the same number of samples.  (Truncating the product, as in ``int(step*fs)``, gives one sample
    too few whenever floating point arithmetic makes it a little less than a whole number, and truncating
    a difference of times, as ``compute_sgram()`` did, can give one sample too many.)  Raises a
    `ValueError` if the step is shorter than half a sample.
    """
    hop = int(round(step * fs))
    if hop < 1:
        raise ValueError(f"The `step` of {step} s is shorter than one sample at a sampling rate of {fs} Hz.")
    return hop


def positive_step(func):
    """Decorator that checks the `step` argument of `func`, if the caller supplies one.

    ``@positive_step`` raises a `ValueError` if ``step`` is zero, negative, NaN or not a number.  The
    default value of ``step`` in the signature is not checked, only a value passed by the caller.  Put it
    *below* ``@renamed_kwargs`` so that a deprecated name such as ``s=0`` is checked too.
    """
    sig = inspect.signature(func)
    name = func.__qualname__.replace('.__init__', '')

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs).arguments
        if 'step' in bound:
            check_step(bound['step'], name)
        return func(*args, **kwargs)
    return wrapper
