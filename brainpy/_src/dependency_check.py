import importlib.util
import os
import sys

__all__ = [
    'import_taichi',
    'import_braintaichi',
    'raise_braintaichi_not_found',
]

taichi = None
braintaichi = None
braintaichi_install_info = ('We need braintaichi. Please install braintaichi by pip . \n'
                            '> pip install braintaichi -U')

os.environ["TI_LOG_LEVEL"] = "error"


def import_taichi(error_if_not_found=True):
    """Internal API to import taichi.

    If taichi is not found, it will raise a ModuleNotFoundError if error_if_not_found is True,
    otherwise it will return None.
    """
    global taichi
    if taichi is None:
        if importlib.util.find_spec('taichi') is not None:
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    import taichi as taichi  # noqa
                except ModuleNotFoundError as e:
                    if error_if_not_found:
                        raise e
                finally:
                    sys.stdout = old_stdout
        else:
            taichi = None

    return taichi


def import_braintaichi(error_if_not_found=True):
    """Internal API to import braintaichi.

    If braintaichi is not found, it will raise a ModuleNotFoundError if error_if_not_found is True,
    otherwise it will return None.
    """
    global braintaichi
    if braintaichi is None:
        if importlib.util.find_spec('braintaichi') is not None:
            try:
                import braintaichi as braintaichi
            except ModuleNotFoundError as e:
                if error_if_not_found:
                    raise e
                else:
                    braintaichi = None
        else:
            braintaichi = None
    return braintaichi


def raise_braintaichi_not_found():
    raise ModuleNotFoundError(braintaichi_install_info)
