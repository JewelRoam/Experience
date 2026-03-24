import torch
from typing import Any, List, Tuple
from collections import defaultdict

Pair = Tuple[List[torch.Tensor], List[torch.Tensor]]


def transpose_pairs_coordinates(
    pairs: List[Pair],
) -> List[Pair]:
    """
    Transpose coordinate pairs: swap keys and values.

    Each input pair is (key, value) where key is a list of scalar index tensors
    (coordinates) and value is a list of multi-element index tensors. This function
    expands each pair into element-level (source_coord, target_coord) pairs, groups
    by target coordinate, and returns (target_coord, collected_source_indices) pairs.

    Args:
        pairs: List of (key, value) pairs.
            - key: list of scalar torch.Tensor (source coordinates)
            - value: list of torch.Tensor (target index tensors, possibly multi-element)

    Returns:
        List of (output_key, output_value) pairs where:
        - output_key: list of scalar torch.Tensor — target coordinate
        - output_value: list of torch.Tensor — collected source indices
    """
    # 1. Expand each (key, value) pair into (key_tuple, value_tuple) element pairs
    all_pairs = []

    for key_tensors, value_tensors in pairs:
        # Convert key scalar tensors to tuple of ints
        key_tuple = tuple(t.item() for t in key_tensors)

        # Convert value index tensors to list of coordinate tuples
        # e.g. [tensor([0, 1]), tensor([2, 3])] → [(0, 2), (1, 3)]
        if not value_tensors:
            continue
        int_lists = []
        for t in value_tensors:
            if t.dim() == 0:
                int_lists.append([t.item()])
            else:
                int_lists.append(t.tolist())
        value_coords = list(zip(*int_lists))

        # Replicate key for each value coordinate
        for vc in value_coords:
            all_pairs.append((key_tuple, vc))

    # 2. Group by second element (target coordinate)
    groups = defaultdict(list)
    for key_tuple, value_tuple in all_pairs:
        groups[value_tuple].append(key_tuple)

    # 3. Convert to output format: list of (target_coord_tensors, source_index_tensors)
    result = []

    for target_coord, source_coords in groups.items():
        # Target coordinate → list of scalar tensors
        out_key = [torch.tensor(c, dtype=torch.long) for c in target_coord]

        # Source coordinates → per-dimension index tensors
        # e.g. [(0, 1), (2, 3)] → [tensor([0, 2]), tensor([1, 3])]
        if source_coords and source_coords[0]:
            n_dims = len(source_coords[0])
            out_value = [
                torch.tensor([sc[d] for sc in source_coords], dtype=torch.long)
                for d in range(n_dims)
            ]
        else:
            out_value = []

        result.append((out_key, out_value))

    return result


if __name__ == "__main__":
    print("Running transpose_pairs_coordinates tests...\n")

    def run_test(name, condition, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # Test 1: Simple 1:1 mapping
    print("Test 1: Simple 1:1 mapping")
    pairs = [
        ([torch.tensor(0)], [torch.tensor([3])]),
        ([torch.tensor(1)], [torch.tensor([4])]),
    ]
    result = transpose_pairs_coordinates(pairs)
    run_test("2 output pairs", len(result) == 2, 2, len(result))
    target_coords = [tuple(t.item() for t in k) for k, v in result]
    run_test("target (3,) present", (3,) in target_coords)
    run_test("target (4,) present", (4,) in target_coords)
    idx3 = target_coords.index((3,))
    idx4 = target_coords.index((4,))
    run_test("target 3 ← source [0]", result[idx3][1][0].tolist() == [0], [0], result[idx3][1][0].tolist())
    run_test("target 4 ← source [1]", result[idx4][1][0].tolist() == [1], [1], result[idx4][1][0].tolist())

    # Test 2: Many-to-one mapping
    print("Test 2: Many-to-one (two sources point to same target)")
    pairs = [
        ([torch.tensor(0)], [torch.tensor([5])]),
        ([torch.tensor(1)], [torch.tensor([5])]),
        ([torch.tensor(2)], [torch.tensor([6])]),
    ]
    result = transpose_pairs_coordinates(pairs)
    run_test("2 output pairs", len(result) == 2, 2, len(result))
    target_coords = [tuple(t.item() for t in k) for k, v in result]
    idx5 = target_coords.index((5,))
    idx6 = target_coords.index((6,))
    run_test("target 5 ← sources [0, 1]", sorted(result[idx5][1][0].tolist()) == [0, 1],
             [0, 1], sorted(result[idx5][1][0].tolist()))
    run_test("target 6 ← source [2]", result[idx6][1][0].tolist() == [2], [2], result[idx6][1][0].tolist())

    # Test 3: Multi-element value tensors (one source → multiple targets)
    print("Test 3: Multi-element value tensors")
    pairs = [
        ([torch.tensor(0)], [torch.tensor([1, 2]), torch.tensor([3, 4])]),
    ]
    # Source 0 points to targets (1,3) and (2,4)
    result = transpose_pairs_coordinates(pairs)
    run_test("2 output pairs", len(result) == 2, 2, len(result))
    target_coords = [tuple(t.item() for t in k) for k, v in result]
    run_test("target (1,3) present", (1, 3) in target_coords)
    run_test("target (2,4) present", (2, 4) in target_coords)
    idx13 = target_coords.index((1, 3))
    run_test("target (1,3) ← source [0]", result[idx13][1][0].tolist() == [0])

    # Test 4: Multi-dimensional keys
    print("Test 4: Multi-dimensional keys")
    pairs = [
        ([torch.tensor(0), torch.tensor(1)], [torch.tensor([5])]),
        ([torch.tensor(0), torch.tensor(2)], [torch.tensor([5])]),
    ]
    # Sources (0,1) and (0,2) both point to target 5
    result = transpose_pairs_coordinates(pairs)
    run_test("1 output pair", len(result) == 1, 1, len(result))
    run_test("target is (5,)", result[0][0][0].item() == 5)
    run_test("source dim0 = [0, 0]", result[0][1][0].tolist() == [0, 0],
             [0, 0], result[0][1][0].tolist())
    run_test("source dim1 = [1, 2]", result[0][1][1].tolist() == [1, 2],
             [1, 2], result[0][1][1].tolist())

    # Test 5: Empty values (no targets)
    print("Test 5: Empty value tensors skipped")
    pairs = [
        ([torch.tensor(0)], []),
        ([torch.tensor(1)], [torch.tensor([3])]),
    ]
    result = transpose_pairs_coordinates(pairs)
    run_test("1 output pair", len(result) == 1, 1, len(result))
    run_test("target is (3,)", result[0][0][0].item() == 3)

    # Test 6: Scalar value tensors (0-dim)
    print("Test 6: Scalar (0-dim) value tensors")
    pairs = [
        ([torch.tensor(0)], [torch.tensor(7)]),
    ]
    result = transpose_pairs_coordinates(pairs)
    run_test("1 output pair", len(result) == 1, 1, len(result))
    run_test("target is (7,)", result[0][0][0].item() == 7)
    run_test("source is [0]", result[0][1][0].tolist() == [0])

    # Test 7: Round-trip with convert_nested_list_coordinates_to_pairs_coordinates
    print("Test 7: Integration with convert_nested_list_coordinates_to_pairs_coordinates")
    from symbolic_tensor.sparse_util.convert_nested_list_coordinates_to_pairs_coordinates import (
        convert_nested_list_coordinates_to_pairs_coordinates,
    )
    # Simulate selected_experience_indexes: 2 inputs, each selecting from experience
    # Input 0 selected experience entries [0, 2], Input 1 selected [1]
    nested = [
        [torch.tensor([0, 2], dtype=torch.long)],
        [torch.tensor([1], dtype=torch.long)],
    ]
    input_pairs = convert_nested_list_coordinates_to_pairs_coordinates(nested)
    run_test("convert: 2 pairs", len(input_pairs) == 2, 2, len(input_pairs))
    result = transpose_pairs_coordinates(input_pairs)
    # Targets: exp entry 0 ← input 0, exp entry 2 ← input 0, exp entry 1 ← input 1
    run_test("transpose: 3 output pairs", len(result) == 3, 3, len(result))
    target_coords = [tuple(t.item() for t in k) for k, v in result]
    run_test("exp 0 present", (0,) in target_coords)
    run_test("exp 1 present", (1,) in target_coords)
    run_test("exp 2 present", (2,) in target_coords)

    # Test 8: Empty input
    print("Test 8: Empty input")
    result = transpose_pairs_coordinates([])
    run_test("0 output pairs", len(result) == 0, 0, len(result))

    print("\nAll tests completed.")
