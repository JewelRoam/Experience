import torch

def st_pack(tensor):
    from experience.symbolic_tensor.tensor_util.pack_tensor import pack_tensor
    return pack_tensor(tensor)

torch.Tensor.st_pack = st_pack


def st_assign(lvalue, rvalue):
    from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor
    return assign_tensor(lvalue, rvalue)

torch.Tensor.st_assign = st_assign

def st_get_diff(lvalue, rvalue):
    from experience.symbolic_tensor.tensor_util.get_diff_tensor import get_diff_tensor
    return get_diff_tensor(lvalue, rvalue)

torch.Tensor.st_get_diff = st_get_diff

def st_patch(lvalue, rvalue):
    from experience.symbolic_tensor.tensor_util.patch_tensor import patch_tensor
    return patch_tensor(lvalue, rvalue)

torch.Tensor.st_patch = st_patch

def st_file_paths(tensor):
    from experience.symbolic_tensor.fs_util.get_nested_list_file_pathes import get_nested_list_file_pathes
    return get_nested_list_file_pathes(tensor)

torch.Tensor.st_file_paths = st_file_paths