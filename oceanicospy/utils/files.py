import os
import shutil
from pathlib import Path
import re

def verify_file(file_path):
    """
    Verifies if a file exists at the given file path.

    Parameters:
    file_path (str): The path of the file to be verified.

    Returns:
    bool: True if the file exists, False otherwise.
    """
    if os.path.isfile(file_path):
        print(f'the file {file_path} already exists')
        return True
    return False

def verify_link(file_name,target_path):
    """
    Verify if a file is already linked in the target path.

    Parameters:
    file_name (str): The name of the file.
    target_path (str): The path where the link should be checked.

    Returns:
    bool: True if the file is already linked, False otherwise.
    """
    if os.path.islink(f'{target_path}{file_name}'):
        print(f'The file {file_name} is already linked in {target_path}')
        return True
    return False

def create_link(file_name,source_path,target_path):
    """
    Creates a symbolic link from the source path to the target path.

    Parameters:
        file_name (str): The name of the file to be linked.
        source_path (str): The path where the file is located.
        target_path (str): The path where the symbolic link will be created.

    Returns:
        None: If the file already exists in the target path.
    """
    if verify_file(f'{target_path}{file_name}'):
        return None
    else:
        os.symlink(f'{source_path}{file_name}',f'{target_path}{file_name}')

def remove_link(file_name, target_path):
    """
    Remove a symbolic link from the target directory.

    Parameters
    ----------
    file_name : str
        Name of the symbolic link to remove.
    target_path : str
        Directory that contains the symbolic link.

    Returns
    -------
    None
    """
    link_path = os.path.join(target_path, file_name)
    if os.path.islink(link_path):
        os.unlink(link_path)
        print(f"Removed symbolic link: {link_path}")
    else:
        print(f"No symbolic link found at: {link_path}")


def deploy_input_file(
    filename: str,
    origin_dir: str,
    run_dir: str,
    use_link: bool | None,
) -> None:
    """
    Deploy an input file from an input directory to a run directory,
    either as a symbolic link or as a physical copy.

    Parameters
    ----------
    filename : str
        Name of the file to deploy (e.g. ``"winds.wnd"``).
    origin_dir : str
        Directory that contains the source file.  Must end with ``/``.
    run_dir : str
        Target run directory where the file or link should appear.
        Must end with ``/``.
    use_link : bool
        * ``True``  — remove any regular file at the target location and
          create a symbolic link pointing to the source.
        * ``False`` — remove any existing symbolic link and copy the file.

    Returns
    -------
    None
    """

    if use_link is None:
        return

    if use_link:
        if verify_file(f'{run_dir}{filename}'):
            os.remove(f'{run_dir}{filename}')
        if not verify_link(filename, run_dir):
            create_link(filename, origin_dir, run_dir)
    else:
        if verify_link(filename, run_dir):
            remove_link(filename, run_dir)
        shutil.copy2(f'{origin_dir}{filename}', run_dir)

def fill_files(file_path: str, replacements: dict):
    """
    Replaces placeholders in an existing .swn file with given values.
    If a value is an empty string, replaces placeholder with whitespace.
    """
    text = Path(file_path).read_text()

    def replace_placeholder(match):
        key = match.group(0)[1:]  # remove leading $
        if key in replacements:
            val = replacements[key]
            return str(val) if val else " "  # empty -> whitespace
        return match.group(0)  # leave untouched if not in replacements

    updated = re.sub(r"\$\w+", replace_placeholder, text)

    Path(file_path).write_text(updated)

def delete_line(file_name,string_to_find):
    with open(file_name, "r") as f:
        lines = f.readlines()
    with open(file_name, "w") as f:
        for line in lines:
            if string_to_find not in line.split():
                f.write(line)

def look_for_NGRID_linenumber(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'NGRID' in line.split():
            return i + 1  # Return 1-based line number
    return False

def count_NGRID_occurrences(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
    count = sum(1 for line in lines if 'NGRID' in line.split())
    return count

def duplicate_lines(file_name, start_line_number):
    with open(file_name, "r") as f:
        lines = f.readlines()

    # Adjust because Python lists are 0-based
    idx = start_line_number - 1

    # Get the lines to duplicate
    if idx < 0 or idx + 1 >= len(lines):
        raise IndexError("Invalid start_line_number")

    lines_to_duplicate = lines[idx:idx + 2]

    # Insert the duplicated lines after the original two
    lines = lines[:idx + 2] + lines_to_duplicate + lines[idx + 2:]

    # Write back to the file
    with open(file_name, "w") as f:
        f.writelines(lines)

def count_lines(file_name):
    with open(file_name, "r") as f:
        return sum(1 for _ in f)

def fill_files_only_once(file_name, dict_):
    replacements = {k: str(v) for k, v in dict_.items()}
    text = Path(file_name).read_text()

    for key, value in replacements.items():
        max_replacements = 2 if key == 'nest_id' else 1
        text = re.sub(r"\$"+re.escape(key), value, text, count=max_replacements)

    Path(file_name).write_text(text)
