import torch

def st_pack(tensor):
    from experience.symbolic_tensor.tensor_util.pack_tensor import pack_tensor
    return pack_tensor(tensor)

torch.Tensor.st_pack = st_pack


def st_assign(lvalue, rvalue):
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor
    return assign_tensor(lvalue, rvalue)

torch.Tensor.st_assign = st_assign

def st_assign_view(lvalue, rvalue):
    from experience.symbolic_tensor.tensor_util.assign_view import assign_view
    return assign_view(lvalue, rvalue)

torch.Tensor.st_assign_view = st_assign_view

def st_get_diff(lvalue, rvalue):
    from experience.symbolic_tensor.tensor_util.get_diff_tensor import get_diff_tensor
    return get_diff_tensor(lvalue, rvalue)

torch.Tensor.st_get_diff = st_get_diff

def st_patch(lvalue, rvalue):
    from experience.symbolic_tensor.tensor_util.patch_tensor import patch_tensor
    return patch_tensor(lvalue, rvalue)

torch.Tensor.st_patch = st_patch

def st_file_paths(tensor):
    from experience.fs_util.get_nested_list_file_pathes import get_nested_list_file_pathes
    return get_nested_list_file_pathes(tensor)

torch.Tensor.st_file_paths = st_file_paths

def st_fork(tensor, num_outputs=2):
    from experience.symbolic_tensor.function.fork_tensor import fork_tensor
    return fork_tensor(tensor, num_outputs=num_outputs)

torch.Tensor.st_fork = st_fork


class _StSlicer:
    """Helper that turns tensor.st_*_slicer[...] into slice_view / slice_tensor calls.

    Converts standard Python indexing syntax into the slice_tensors list
    expected by slice_view / slice_tensor.
    """

    def __init__(self, tensor, slice_fn):
        self._tensor = tensor
        self._slice_fn = slice_fn

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        ndim = self._tensor.dim()
        # Expand Ellipsis
        slice_list = []
        for k in key:
            if k is Ellipsis:
                n_expand = ndim - (len(key) - 1)
                slice_list.extend([slice(None)] * n_expand)
            else:
                slice_list.append(k)
        # Pad with slice(None) if fewer indices than dims
        while len(slice_list) < ndim:
            slice_list.append(slice(None))
        return self._slice_fn(self._tensor, slice_list)


@property
def _st_view_slicer(self):
    from experience.symbolic_tensor.function.slice_view import slice_view
    return _StSlicer(self, slice_view)

torch.Tensor.st_view_slicer = _st_view_slicer


@property
def _st_value_slicer(self):
    from experience.symbolic_tensor.function.slice_tensor import slice_tensor
    return _StSlicer(self, slice_tensor)

torch.Tensor.st_value_slicer = _st_value_slicer