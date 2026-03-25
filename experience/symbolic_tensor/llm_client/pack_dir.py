
import os
import seedir


def pack_dir(root_dir: str) -> str:
    # Get directory structure
    dir_struct = seedir.seedir(root_dir, printout=False)

    # Recursively find all files and their content, resolving symlinks
    file_sections = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # Resolve symlinks to get original file content
            real_path = os.path.realpath(file_path)
            relative_file_path = os.path.relpath(file_path, root_dir)
            try:
                with open(real_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            file_sections.append(
                f"## File: {relative_file_path}\n"
                f"```\n"
                f"{file_content}\n"
                f"```"
            )

    ret = (
        f"# Directories structure\n"
        f"```\n"
        f"{dir_struct}\n"
        f"```\n\n"
        f"# Files\n\n"
        + "\n\n".join(file_sections)
    )
    return ret


if __name__ == "__main__":
    import tempfile

    print("Running pack_dir tests...\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test structure
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "hello.txt"), "w") as f:
            f.write("hello world")
        with open(os.path.join(tmpdir, "subdir", "nested.txt"), "w") as f:
            f.write("nested content")

        # Create a symlink
        target = os.path.join(tmpdir, "hello.txt")
        link = os.path.join(tmpdir, "link.txt")
        os.symlink(target, link)

        result = pack_dir(tmpdir)

        assert "# Directories structure" in result
        print("  ok: contains directory structure header")

        assert "# Files" in result
        print("  ok: contains files header")

        assert "hello world" in result
        print("  ok: contains file content")

        assert "nested content" in result
        print("  ok: contains nested file content")

        assert "## File: hello.txt" in result
        print("  ok: contains relative file path")

        assert "## File: link.txt" in result
        print("  ok: symlink file listed")

        # Symlink should resolve to same content
        assert result.count("hello world") >= 2  # original + symlink
        print("  ok: symlink resolves to original content")

    print("\nAll tests passed.")
