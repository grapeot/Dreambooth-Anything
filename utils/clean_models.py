from os import listdir, remove
from os.path import join, isfile
from shutil import rmtree
import re

def clean_one_checkpoint(path: dir):
    """
    Remove the content of the given dir except a subdirectory named "samples".
    """
    dirs = listdir(path)
    for d in dirs:
        if d == 'samples':
            continue
        path_to_remove = join(path, d)
        print(path_to_remove)
        if isfile(path_to_remove):
            remove(path_to_remove)
        else:
            rmtree(path_to_remove)

def clean_model(path: str):
    """
    Clean the model folder by removing all the serialized models except the latest one, leaving the generated samples only.
    The `path` is expected to be the root dir of the model, i.e. the output directory of training.
    """
    checkpoint_dirs = listdir(path)
    # find the latest
    checkpoint_numbers = sorted([int(d) for d in checkpoint_dirs if re.match(r'^\d*$', d) is not None and d != '0'])
    for num in checkpoint_numbers[:-1]:
        # if num % 5000 != 0:
        clean_one_checkpoint(join(path, str(num)))

if __name__ == '__main__':
    # clean_model('data/models/anythingv4_touhou100k_noclass_5e-8')
    clean_model('data/models/SDv2_UglySweater_2e-8_linear_warmup')
    # clean_model('data/models/sd21_dpeth_Lycoris500_5e-8_scheduler')