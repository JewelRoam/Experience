import torch

def group_random_select(group_ids):
    """
    Randomly select one element uniformly from each group, returning original indices.
    group_ids: (N,) integer tensor
    """
    # 1. Sort by group_ids so that elements of the same group become consecutive
    sorted_idx = torch.argsort(group_ids)
    sorted_groups = group_ids[sorted_idx]

    # 2. Find the start positions and lengths of each group
    # Mark the first element of each group
    start_mask = torch.cat([torch.tensor([True], device=group_ids.device),
                            sorted_groups[1:] != sorted_groups[:-1]])
    group_starts = torch.nonzero(start_mask, as_tuple=True)[0]          # shape (num_groups,)
    group_ends = torch.cat([group_starts[1:],
                            torch.tensor([len(sorted_groups)], device=group_ids.device)])

    # 3. Generate a random local offset (0 to group_size-1) for each group
    group_sizes = group_ends - group_starts
    rand_float = torch.rand(len(group_sizes), device=group_ids.device)
    local_offsets = (rand_float * group_sizes).long()   # uniform integer within each group

    # 4. Compute global positions in the sorted array
    global_pos_in_sorted = group_starts + local_offsets

    # 5. Map back to original indices
    original_indices = sorted_idx[global_pos_in_sorted]
    return original_indices


if __name__ == "__main__":
    print("Running group_random_select tests...\n")

    def run_test(name, condition, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # Test 1: One element per group
    print("Test 1: One element per group")
    group_ids = torch.tensor([0, 1, 2])
    result = group_random_select(group_ids)
    run_test("Returns 3 indices", result.shape == (3,), (3,), tuple(result.shape))
    run_test("Each group selects its only element", sorted(result.tolist()) == [0, 1, 2],
             [0, 1, 2], sorted(result.tolist()))

    # Test 2: All same group
    print("Test 2: All same group")
    group_ids = torch.tensor([5, 5, 5, 5])
    result = group_random_select(group_ids)
    run_test("Returns 1 index", result.shape == (1,), (1,), tuple(result.shape))
    run_test("Index is valid", 0 <= result.item() <= 3, "0-3", result.item())

    # Test 3: Two groups
    print("Test 3: Two groups")
    group_ids = torch.tensor([1, 0, 1, 0, 1])
    result = group_random_select(group_ids)
    run_test("Returns 2 indices", result.shape == (2,), (2,), tuple(result.shape))
    selected_groups = group_ids[result]
    run_test("Selects from both groups", sorted(selected_groups.tolist()) == [0, 1],
             [0, 1], sorted(selected_groups.tolist()))
    # Group 0 elements are at indices 1, 3
    # Group 1 elements are at indices 0, 2, 4
    g0_idx = result[selected_groups == 0].item()
    g1_idx = result[selected_groups == 1].item()
    run_test("Group 0 index valid", g0_idx in [1, 3], "{1, 3}", g0_idx)
    run_test("Group 1 index valid", g1_idx in [0, 2, 4], "{0, 2, 4}", g1_idx)

    # Test 4: Non-contiguous group ids
    print("Test 4: Non-contiguous group ids")
    group_ids = torch.tensor([10, 20, 10, 30, 20])
    result = group_random_select(group_ids)
    run_test("Returns 3 indices (3 unique groups)", result.shape == (3,), (3,), tuple(result.shape))
    selected_groups = group_ids[result]
    run_test("All groups represented", sorted(selected_groups.tolist()) == [10, 20, 30],
             [10, 20, 30], sorted(selected_groups.tolist()))

    # Test 5: Uniformity — run many times, check all candidates get selected
    print("Test 5: Uniformity check")
    group_ids = torch.tensor([0, 0, 0])
    counts = torch.zeros(3, dtype=torch.long)
    n_trials = 3000
    for _ in range(n_trials):
        idx = group_random_select(group_ids).item()
        counts[idx] += 1
    run_test("All elements selected at least once", (counts > 0).all().item(),
             "all > 0", counts.tolist())
    min_ratio = counts.min().item() / n_trials
    run_test(f"Min selection ratio >= 0.2 (got {min_ratio:.3f})", min_ratio >= 0.2,
             ">= 0.2", f"{min_ratio:.3f}")

    # Test 6: Single element
    print("Test 6: Single element")
    group_ids = torch.tensor([7])
    result = group_random_select(group_ids)
    run_test("Returns 1 index", result.shape == (1,), (1,), tuple(result.shape))
    run_test("Index is 0", result.item() == 0, 0, result.item())

    print("\nAll tests completed.")
