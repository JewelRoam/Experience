"""TextMerger: pack/unpack a list of (index, coefficient, content) frames into/from a single string.

Used by merge_forward to merge multiple symbolic tensor elements along an axis
into a single text representation.
"""

from typing import List, Tuple


kFrameMarker = "===ST-MERGED-CONTENT===ST-MERGED-CONTENT===ST-MERGED-CONTENT==="


def frame_to_str(frame: Tuple[int, float, str]) -> str:
    """Convert a single (index, coefficient, content) frame to a string block."""
    index, coefficient, content = frame
    indented_content = "\n".join("  " + line for line in content.splitlines())
    return (
        f"{kFrameMarker}\n"
        f"index: {index}\n"
        f"coefficient: {coefficient}\n"
        f"content:\n\n"
        f"{indented_content}"
    )


def pack(frames: List[Tuple[int, float, str]]) -> str:
    """Pack a list of (index, coefficient, content) frames into a single merged string."""
    return "\n".join(frame_to_str(frame) for frame in frames)


def unpack(merged: str) -> List[Tuple[int, float, str]]:
    """Unpack a merged string back into a list of (index, coefficient, content) frames.

    This is the inverse of pack().
    """
    if not merged.strip():
        return []

    # Split on the frame marker
    parts = merged.split(kFrameMarker)
    frames: List[Tuple[int, float, str]] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        lines = part.splitlines()
        index = None
        coefficient = None
        content_lines: List[str] = []
        in_content = False

        content_started = False
        for line in lines:
            if in_content:
                # Skip leading blank lines before actual content starts
                if not content_started:
                    if line.strip() == "":
                        continue
                    content_started = True
                # Content lines are indented with two spaces
                if line.startswith("  "):
                    content_lines.append(line[2:])
                else:
                    content_lines.append(line)
            elif line.startswith("index: "):
                index = int(line[len("index: "):])
            elif line.startswith("coefficient: "):
                coefficient = float(line[len("coefficient: "):])
            elif line.startswith("content:"):
                in_content = True

        if index is not None and coefficient is not None:
            frames.append((index, coefficient, "\n".join(content_lines)))

    return frames


class TextMerger:
    """Stateless class with pack/unpack as class methods."""

    pack = staticmethod(pack)
    unpack = staticmethod(unpack)


if __name__ == "__main__":
    print("Running TextMerger tests...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # Test 1: frame_to_str
    print("Test 1: frame_to_str")
    result = frame_to_str((0, 1.0, "hello\nworld"))
    run_test("contains marker", kFrameMarker in result)
    run_test("contains index", "index: 0" in result)
    run_test("contains coefficient", "coefficient: 1.0" in result)
    run_test("contains indented content", "  hello" in result)
    run_test("contains indented line 2", "  world" in result)

    # Test 2: pack single frame
    print("Test 2: pack single frame")
    packed = pack([(0, 1.0, "hello")])
    run_test("contains marker", kFrameMarker in packed)
    run_test("contains index", "index: 0" in packed)
    run_test("contains content", "  hello" in packed)

    # Test 3: pack multiple frames
    print("Test 3: pack multiple frames")
    frames = [(0, 1.0, "alpha"), (1, 0.5, "beta\ngamma"), (2, 0.25, "delta")]
    packed = pack(frames)
    run_test("3 markers", packed.count(kFrameMarker) == 3)

    # Test 4: unpack reverses pack (single)
    print("Test 4: unpack reverses pack (single)")
    frames = [(0, 1.0, "hello")]
    packed = pack(frames)
    unpacked = unpack(packed)
    run_test("length 1", len(unpacked) == 1)
    run_test("index", unpacked[0][0] == 0)
    run_test("coefficient", unpacked[0][1] == 1.0)
    run_test("content", unpacked[0][2] == "hello")

    # Test 5: unpack reverses pack (multiple)
    print("Test 5: unpack reverses pack (multiple)")
    frames = [(0, 1.0, "alpha"), (1, 0.5, "beta\ngamma"), (2, 0.25, "delta")]
    packed = pack(frames)
    unpacked = unpack(packed)
    run_test("length 3", len(unpacked) == 3)
    for i, (idx, coeff, text) in enumerate(frames):
        run_test(f"frame[{i}] index", unpacked[i][0] == idx)
        run_test(f"frame[{i}] coeff", unpacked[i][1] == coeff)
        run_test(f"frame[{i}] content", unpacked[i][2] == text)

    # Test 6: unpack empty string
    print("Test 6: unpack empty string")
    run_test("empty -> []", unpack("") == [])
    run_test("whitespace -> []", unpack("   \n  ") == [])

    # Test 7: round-trip with multiline content
    print("Test 7: round-trip multiline content")
    frames = [(3, 0.75, "line1\nline2\nline3")]
    unpacked = unpack(pack(frames))
    run_test("multiline preserved", unpacked[0][2] == "line1\nline2\nline3")

    # Test 8: TextMerger class methods
    print("Test 8: TextMerger class methods")
    frames = [(0, 1.0, "test")]
    packed = TextMerger.pack(frames)
    unpacked = TextMerger.unpack(packed)
    run_test("TextMerger.pack works", kFrameMarker in packed)
    run_test("TextMerger.unpack works", unpacked[0][2] == "test")

    # Test 9: round-trip with zero coefficient
    print("Test 9: zero coefficient")
    frames = [(5, 0.0, "zero coeff content")]
    unpacked = unpack(pack(frames))
    run_test("zero coeff preserved", unpacked[0][1] == 0.0)
    run_test("content preserved", unpacked[0][2] == "zero coeff content")

    # Test 10: content with special characters
    print("Test 10: special characters in content")
    frames = [(0, 1.0, "def foo():\n    return 'bar'\n# comment")]
    unpacked = unpack(pack(frames))
    run_test("code preserved", unpacked[0][2] == "def foo():\n    return 'bar'\n# comment")

    print("\nAll tests completed.")
