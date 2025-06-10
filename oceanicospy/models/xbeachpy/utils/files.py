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
    if os.path.islink(f'{target_path}/{file_name}'):
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

def fill_files(file_name,dict):
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
    for key,value in dict.items():               
        with fileinput.FileInput(file_name,inplace=True, backup='') as file:
            for line in file:
                print(line.replace(key,value),end='')