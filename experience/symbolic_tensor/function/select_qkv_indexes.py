import os
import torch
from typing import List, Tuple

from experience.symbolic_tensor.tensor_util.dump_view import dump_view


def _get_jaccard_similarity(a: List[str], b: List[str]) -> float:
    """Compute Jaccard similarity between two lists of keywords."""
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _filter_last_coordinate_eq_zero(view_dir: str) -> List[str]:
    """Walk the view directory and return paths to data files where the last coordinate is 0.

    In the qkv view layout, coordinate 0 = query, 1 = key, 2 = value.
    We only want the query files for similarity matching.
    e.g. view_dir/0/0/data.txt (row 0, coord 0=query) is included,
         view_dir/0/1/data.txt (row 0, coord 1=key) is excluded.
    """
    result = []
    for root, _dirs, files in os.walk(view_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            # Check that last coordinate directory is "0"
            rel = os.path.relpath(fpath, view_dir)
            parts = rel.split(os.sep)
            # parts: [coord0, coord1, ..., coordN, filename]
            if len(parts) >= 2 and parts[-2] == "0":
                real_path = os.path.realpath(fpath)
                if os.path.isfile(real_path) and os.path.getsize(real_path) > 0:
                    result.append(fpath)
    return sorted(result)


def _extract_coordinates(file_path: str, view_dir: str) -> List[int]:
    """Extract coordinate indices from a view file path.

    Given view_dir/0/1/2/data.txt, extracts [0, 1, 2].
    """
    rel = os.path.relpath(file_path, view_dir)
    parts = rel.split(os.sep)
    # Last part is the filename (data.txt), rest are coordinate dirs
    coord_parts = parts[:-1]
    return [int(p) for p in coord_parts]


def _unzip_to_tensor_list(coordinates: List[List[int]]) -> List[torch.Tensor]:
    """Unzip a list of coordinate tuples into a list of tensors, one per dimension.

    [[0,1], [2,3], [0,3]] -> [tensor([0,2,0]), tensor([1,3,3])]
    """
    if not coordinates:
        return []
    ndim = len(coordinates[0])
    return [torch.tensor([coord[d] for coord in coordinates], dtype=torch.long)
            for d in range(ndim)]


def select_qkv_indexes(
    weight_tensor: torch.Tensor,
    query_key_words: List[str],
    topk: int,
    random_noise: bool = True,
) -> List[torch.Tensor]:
    """
    Select top-k entries from an Experience tensor by Jaccard similarity,
    optionally adding Gaussian noise to similarity scores for exploration.

    Args:
        weight_tensor: An Experience symbolic tensor (last dim = 3: q, k, v).
        query_key_words: List of query keywords to match against.
        topk: Number of top matches to return.
        random_noise: If True, add Gaussian noise to similarity scores.
            Default True. Set False for deterministic test cases.

    Returns:
        A list of torch.Tensor[int], one per dimension of the tensor
        (excluding the last qkv dimension), containing the selected indices.
    """
    original_tensor_dir = os.path.join(
        weight_tensor.st_relative_to, weight_tensor.st_tensor_uid
    )
    qkv_data_view_dir = os.path.join(original_tensor_dir, "qkv_data_view")

    # Dump view only if not already done (cached)
    if not os.path.isdir(qkv_data_view_dir):
        dump_view(weight_tensor, qkv_data_view_dir, "txt")

    # Find query files (last coordinate == 0) in the view
    query_file_paths = _filter_last_coordinate_eq_zero(qkv_data_view_dir)

    # Compute Jaccard similarity for each file
    similarity_values: List[float] = []
    for query_file_path in query_file_paths:
        real_path = os.path.realpath(query_file_path)
        with open(real_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        query_file_key_words = [w for w in content.split("\n") if w.strip()]
        similarity = _get_jaccard_similarity(query_key_words, query_file_key_words)
        similarity_values.append(similarity)

    # Compute noise parameters
    if similarity_values:
        similarity_mean = sum(similarity_values) / len(similarity_values)
    else:
        similarity_mean = 0.0
    quarter_of_similarity_mean = similarity_mean / 4.0
    epsilon = 0.01
    noise_std = quarter_of_similarity_mean + epsilon

    # Optionally add Gaussian noise for exploration
    if random_noise:
        noisy_similarities = [
            sim + torch.normal(
                mean=torch.tensor(quarter_of_similarity_mean),
                std=torch.tensor(noise_std),
            ).item()
            for sim in similarity_values
        ]
    else:
        noisy_similarities = similarity_values

    # Pair paths with (noisy) similarities and select top-k descending
    paired: List[Tuple[str, float]] = list(zip(query_file_paths, noisy_similarities))
    paired.sort(key=lambda x: x[1], reverse=True)
    # print("paired", paired)
    selected = paired[:topk]
    selected_paths = [path for path, _ in selected]

    # Extract coordinates from selected file paths and unzip to tensor list
    coordinates = [_extract_coordinates(p, qkv_data_view_dir) for p in selected_paths]
    ret = _unzip_to_tensor_list(coordinates)
    if len(ret) == 0:
        return [torch.tensor([], dtype=torch.long)] * len(weight_tensor.shape)
    return ret


if __name__ == "__main__":
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

    print("Running tests for select_qkv_indexes...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # Test 1: Basic selection from a 2x3 experience tensor (shape [2, 3])
    # Last dim = 3 means q, k, v
    print("Test 1: Basic top-k selection")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a 2x3 tensor where last dim=3 (q,k,v)
        # Each string contains keywords (one per line)
        data = [
            ["python\nfunction\ndef", "key_a", "value_a"],   # row 0
            ["java\nclass\nobject", "key_b", "value_b"],     # row 1
        ]
        t = make_tensor(data, tmpdir)
        # shape is [2, 3]

        # Query for python-related keywords (random_noise=False for deterministic test)
        result = select_qkv_indexes(t, ["python", "function"], topk=1, random_noise=False)
        run_test("Returns list of tensors", isinstance(result, list))
        run_test("Two index tensors (2 dims)", len(result) == 2)
        # Row 0 has "python\nfunction\ndef" -> highest Jaccard with ["python", "function"]
        run_test("First dim index is 0", result[0].item() == 0)
        print(f"    Selected indices: dim0={result[0].tolist()}, dim1={result[1].tolist()}")

    # Test 2: Top-2 selection
    print("Test 2: Top-2 selection")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = [
            ["alpha\nbeta", "k0", "v0"],
            ["gamma\ndelta", "k1", "v1"],
            ["alpha\ngamma\nepsilon", "k2", "v2"],
        ]
        t = make_tensor(data, tmpdir)

        result = select_qkv_indexes(t, ["alpha", "beta"], topk=2, random_noise=False)
        run_test("Two index tensors", len(result) == 2)
        run_test("Two results each", len(result[0]) == 2)
        # Row 0 has jaccard(["alpha","beta"], ["alpha","beta"]) = 1.0
        # Row 2 has jaccard(["alpha","beta"], ["alpha","gamma","epsilon"]) = 1/4 = 0.25
        # Row 1 has jaccard(["alpha","beta"], ["gamma","delta"]) = 0.0
        run_test("Best match is row 0", result[0][0].item() == 0)
        print(f"    Selected: dim0={result[0].tolist()}, dim1={result[1].tolist()}")

    # Test 3: Cached view (second call reuses)
    print("Test 3: Cached view directory")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = [["hello\nworld", "k", "v"]]
        t = make_tensor(data, tmpdir)

        view_dir = os.path.join(tmpdir, t.st_tensor_uid, "qkv_data_view")
        result1 = select_qkv_indexes(t, ["hello"], topk=1, random_noise=False)
        run_test("View dir created", os.path.isdir(view_dir))
        result2 = select_qkv_indexes(t, ["world"], topk=1, random_noise=False)
        run_test("View dir still exists (cached)", os.path.isdir(view_dir))
        run_test("Same results shape", len(result1) == len(result2))

    # Test 4: Empty query keywords
    print("Test 4: Empty query returns zero similarity")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = [["keyword", "k", "v"]]
        t = make_tensor(data, tmpdir)
        result = select_qkv_indexes(t, [], topk=1, random_noise=False)
        run_test("Still returns results", len(result) == 2)

    # Test 5: Random noise changes selection order
    print("Test 5: Random noise exploration")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = [
            ["alpha\nbeta", "k0", "v0"],
            ["gamma\ndelta", "k1", "v1"],
            ["alpha\ngamma\nepsilon", "k2", "v2"],
        ]
        t = make_tensor(data, tmpdir)

        # With random_noise=True, results may vary across runs
        result_noisy = select_qkv_indexes(t, ["alpha", "beta"], topk=2, random_noise=True)
        run_test("Noisy result still returns 2 index tensors", len(result_noisy) == 2)
        run_test("Noisy result still returns topk=2 entries", len(result_noisy[0]) == 2)
        print(f"    Noisy selected: dim0={result_noisy[0].tolist()}, dim1={result_noisy[1].tolist()}")

    print("\nAll tests completed.")
