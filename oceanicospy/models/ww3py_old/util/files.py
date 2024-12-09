import os
import fileinput

def verify_files(file_path):
    if os.path.isfile(file_path):
        print(f'the file {file_path} already exists')
        return True
    return False

def verify_links(file_name,source_path,target_path):
    if os.path.isfile(f'{target_path}{file_name}') or os.path.islink(f'{target_path}{file_name}'):
        print(f'the {target_path}{file_name} is already linked') 
    else:
        os.symlink(f'{source_path}{file_name}',f'{target_path}{file_name}')

def fill_files(file_name,dict):
    for key,value in dict.items():               
        with fileinput.FileInput(file_name,inplace=True, backup='') as file:
            for line in file:
                print(line.replace(key,value),end='')