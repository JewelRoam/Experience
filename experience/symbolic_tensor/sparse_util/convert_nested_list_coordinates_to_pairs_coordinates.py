import torch
from typing import Any, List, Tuple


def _is_leaf(obj: Any) -> bool:
    """Check if obj is a leaf (list[torch.Tensor]) rather than a nested list of lists."""
    if not isinstance(obj, list):
        return True
    if len(obj) == 0:
        return True
    return isinstance(obj[0], torch.Tensor)


Pair = Tuple[List[torch.Tensor], Any]


def _collect(
    obj: Any,
    coords: List[int],
    pairs: List[Pair],
) -> None:
    """Recursively collect (coordinate_key, leaf_value) pairs."""
    if _is_leaf(obj):
        key = [torch.tensor(c, dtype=torch.long) for c in coords]
        pairs.append((key, obj))
    else:
        for i, item in enumerate(obj):
            _collect(item, coords + [i], pairs)


def convert_nested_list_coordinates_to_pairs_coordinates(
    nested: Any,
) -> List[Pair]:
    """
    Flatten a nested list into a list of (key, value) pairs.

    Keys are the multi-dimensional coordinates (as scalar index tensors)
    to reach each leaf value in the nested structure.

    Args:
        nested: A nested list where leaves are list[torch.Tensor] (index tensor lists).

    Returns:
        List of (key, value) pairs where:
        - key: list of scalar torch.Tensor coordinates
        - value: the leaf value (list[torch.Tensor])
    """
    pairs: List[Pair] = []
    _collect(nested, [], pairs)
    return pairs


if __name__ == "__main__":
    print("Running convert_nested_list_coordinates_to_pairs_coordinates tests...\n")

    def run_test(name, condition, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # Test 1: Flat list of leaves (1D nesting)
    print("Test 1: Flat list of leaves")
    nested = [
        [torch.tensor([0]), torch.tensor([1])],
        [torch.tensor([2]), torch.tensor([3])],
    ]
    pairs = convert_nested_list_coordinates_to_pairs_coordinates(nested)
    run_test("2 pairs", len(pairs) == 2, 2, len(pairs))
    run_test("key[0] = [0]", len(pairs[0][0]) == 1 and pairs[0][0][0].item() == 0)
    run_test("key[1] = [1]", len(pairs[1][0]) == 1 and pairs[1][0][0].item() == 1)
    run_test("value[0] correct", len(pairs[0][1]) == 2 and pairs[0][1][0].item() == 0)
    run_test("value[1] correct", len(pairs[1][1]) == 2 and pairs[1][1][0].item() == 2)

    # Test 2: 2D nesting
    print("Test 2: 2D nesting")
    nested = [
        [
            [torch.tensor([10])],
            [torch.tensor([20])],
        ],
        [
            [torch.tensor([30])],
        ],
    ]
    pairs = convert_nested_list_coordinates_to_pairs_coordinates(nested)
    run_test("3 pairs", len(pairs) == 3, 3, len(pairs))
    k0 = [t.item() for t in pairs[0][0]]
    k1 = [t.item() for t in pairs[1][0]]
    k2 = [t.item() for t in pairs[2][0]]
    run_test("key[0] = [0, 0]", k0 == [0, 0], [0, 0], k0)
    run_test("key[1] = [0, 1]", k1 == [0, 1], [0, 1], k1)
    run_test("key[2] = [1, 0]", k2 == [1, 0], [1, 0], k2)
    run_test("value[0] = [10]", pairs[0][1][0].item() == 10)
    run_test("value[2] = [30]", pairs[2][1][0].item() == 30)

    # Test 3: Single leaf (no nesting)
    print("Test 3: Single leaf (list[Tensor] directly)")
    nested = [torch.tensor([5]), torch.tensor([6])]
    pairs = convert_nested_list_coordinates_to_pairs_coordinates(nested)
    run_test("1 pair", len(pairs) == 1, 1, len(pairs))
    run_test("key is empty (root)", len(pairs[0][0]) == 0, 0, len(pairs[0][0]))
    run_test("value preserved", len(pairs[0][1]) == 2 and pairs[0][1][1].item() == 6)

    # Test 4: Empty nested list
    print("Test 4: Empty list")
    nested = []
    pairs = convert_nested_list_coordinates_to_pairs_coordinates(nested)
    run_test("1 pair (empty list is leaf)", len(pairs) == 1, 1, len(pairs))
    run_test("value is empty list", pairs[0][1] == [], [], pairs[0][1])

    # Test 5: 3D nesting
    print("Test 5: 3D nesting")
    nested = [
        [
            [
                [torch.tensor([0, 1])],
                [torch.tensor([2, 3])],
            ]
        ]
    ]
    pairs = convert_nested_list_coordinates_to_pairs_coordinates(nested)
    run_test("2 pairs", len(pairs) == 2, 2, len(pairs))
    k0 = [t.item() for t in pairs[0][0]]
    k1 = [t.item() for t in pairs[1][0]]
    run_test("key[0] = [0, 0, 0]", k0 == [0, 0, 0], [0, 0, 0], k0)
    run_test("key[1] = [0, 0, 1]", k1 == [0, 0, 1], [0, 0, 1], k1)

    # Test 6: Matches _flatten_nested_indexes output order
    print("Test 6: Order matches depth-first traversal")
    nested = [
        [
            [torch.tensor([0]), torch.tensor([0])],
            [torch.tensor([1]), torch.tensor([1])],
        ],
        [
            [torch.tensor([2]), torch.tensor([2])],
        ],
    ]
    pairs = convert_nested_list_coordinates_to_pairs_coordinates(nested)
    run_test("3 pairs", len(pairs) == 3, 3, len(pairs))
    coords = [[t.item() for t in p[0]] for p in pairs]
    run_test("DFS order", coords == [[0, 0], [0, 1], [1, 0]],
             [[0, 0], [0, 1], [1, 0]], coords)
    run_test("value[0] first elem=0", pairs[0][1][0].item() == 0)
    run_test("value[1] first elem=1", pairs[1][1][0].item() == 1)
    run_test("value[2] first elem=2", pairs[2][1][0].item() == 2)

    print("\nAll tests completed.")
