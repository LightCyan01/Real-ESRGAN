"""Compatibility shims for the 2022-era Real-ESRGAN dependency stack on
modern PyTorch/torchvision.

IMPORTANT: this must happen BEFORE any ``import basicsr`` (or anything that
transitively imports basicsr, e.g. ``realesrgan.archs``). It is imported first
in ``realesrgan/__init__.py`` and in the inference scripts for exactly this
reason.
"""
import sys


def _install_functional_tensor_shim() -> None:
    # torchvision removed ``torchvision.transforms.functional_tensor`` in 0.17.
    # basicsr/facexlib/gfpgan still do:
    #     from torchvision.transforms.functional_tensor import rgb_to_grayscale
    # Alias the function's new home so those imports keep working.
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return  # torchvision < 0.17: the real module exists, nothing to do.
    except ModuleNotFoundError:
        pass

    import types
    import torchvision.transforms.functional as _F

    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    # basicsr only imports rgb_to_grayscale from this path; expose it.
    shim.rgb_to_grayscale = _F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = shim


_install_functional_tensor_shim()
