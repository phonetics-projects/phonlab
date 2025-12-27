__name__ = "phonlab"
__version__ = "0.0.45"
import lazy_loader as lazy

# Attach the lazy loader to the current module
(__getattr__, __dir__, __all__) = lazy.attach_stub(__name__, __file__)
