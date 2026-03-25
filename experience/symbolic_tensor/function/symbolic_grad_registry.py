"""
Thread-local registry for passing symbolic gradient metadata between
autograd Function backward calls.

PyTorch autograd strips custom tensor attributes (st_relative_to, st_tensor_uid)
when propagating gradients between Function nodes. This registry allows a
backward function to attach symbolic metadata to a gradient tensor, so the
next backward in the chain or the optimizer can retrieve it.

Keys can be:
  - st_tensor_uid strings (for inter-backward registry, keyed by forward tensor uid)
  - Python id() of parameter tensors (for backward-to-optimizer registry)

Usage:
    # In producer backward:
    symbolic_grad_registry.register("some_uid", symbolic_grad)

    # In consumer:
    symbolic_grad = symbolic_grad_registry.pop("some_uid")
"""
import threading

_local = threading.local()


def _get_store():
    if not hasattr(_local, "store"):
        _local.store = {}
    return _local.store


def register(key, symbolic_tensor):
    """Associate a symbolic gradient with a key (string uid or int id)."""
    _get_store()[key] = symbolic_tensor


def pop(key):
    """Retrieve and remove the symbolic gradient for a given key.
    Returns None if not found."""
    return _get_store().pop(key, None)


def peek(key):
    """Retrieve the symbolic gradient without removing it.
    Returns None if not found."""
    return _get_store().get(key, None)
