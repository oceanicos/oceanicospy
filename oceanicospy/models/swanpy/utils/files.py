import os
import fileinput

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

    PArameters:
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

def fill_files(file_name,dict_,strict=True):
    """
    Replaces occurrences of keys with their corresponding values in a file.

    Args:
        file_name (str): The path of the file to be modified.
        dict (dict): A dictionary containing the key-value pairs to be replaced.

    Returns:
        None: This function does not return any value.

        Example:
            >>> fill_files('/path/to/file.txt', {'key1': 'value1', 'key2': 'value2'})
        """

    dict_to_use=dict_.copy()
    for key_,value_ in dict_.items():
        if (type(value_)==float) or (type(value_)==int):
            dict_to_use[key_]=str(value_)
        dict_to_use[key_]=str(value_)
    
    if strict == True:
        for key,value in dict_to_use.items():
            with fileinput.FileInput(file_name,inplace=True,backup='') as file:
                    for line in file:
                        line_splitted=[string.replace("'","") for string in line.split()]
                        if key in line_splitted:
                                line=line.replace(key,value)
                                print(line,end='')
                        else:
                            print(line,end='')
    else:
        for key,value in dict_to_use.items():  
            with fileinput.FileInput(file_name,inplace=True,backup='') as file:
                    for line in file:
                        print(line.replace(key,value),end="")


def delete_line(file_name,string_to_find):
    with open(file_name, "r") as f:
        lines = f.readlines()
    with open(file_name, "w") as f:
        for line in lines:
            if string_to_find not in line.split():
                f.write(line)

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