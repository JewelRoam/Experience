"""
Initialize dataset: 3 Python files and 3 corresponding Viba files.
  1) Sequential code: Python sequential vs Viba sequential
  2) Branch code: Python if/else vs Viba sum type
  3) Loop code: Python for-loop vs Viba pattern match
"""
import os

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")


def init_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)

    pairs = {
        # 1) Sequential code
        "seq.py": (
            "def greet(name: str) -> str:\n"
            "    greeting = 'Hello'\n"
            "    message = greeting + ', ' + name + '!'\n"
            "    return message\n"
        ),
        "seq.viba": (
            "greet :=\n"
            "    $message str\n"
            "    <- $name str\n"
            "    # inline\n"
            "    <- ($greeting <- 'Hello')\n"
            "    <- ($message <- $greeting + ', ' + $name + '!')\n"
        ),

        # 2) Branch code
        "branch.py": (
            "def classify(x: int) -> str:\n"
            "    if x > 0:\n"
            "        return 'positive'\n"
            "    elif x == 0:\n"
            "        return 'zero'\n"
            "    else:\n"
            "        return 'negative'\n"
        ),
        "branch.viba": (
            "classify :=\n"
            "    | 'positive'\n"
            "    | 'zero'\n"
            "    | 'negative'\n"
            "    <- $x int\n"
            "    # inline\n"
            "    <- Match[$x > 0 -> 'positive', $x == 0 -> 'zero', _ -> 'negative']\n"
        ),

        # 3) Loop code
        "loop.py": (
            "def double_all(items: list[int]) -> list[int]:\n"
            "    result = []\n"
            "    for item in items:\n"
            "        result.append(item * 2)\n"
            "    return result\n"
        ),
        "loop.viba": (
            "double_all :=\n"
            "    list[$doubled int]\n"
            "    <- $items list[int]\n"
            "    # inline\n"
            "    <- (list[$doubled <- $item * 2] <- $items list[$item int])\n"
        ),
    }

    for filename, content in pairs.items():
        filepath = os.path.join(DATASET_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Created {filepath}")


if __name__ == "__main__":
    print("Initializing dataset...\n")
    init_dataset()
    print("\nDataset ready.")
