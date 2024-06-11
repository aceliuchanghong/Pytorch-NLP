# pip install easy-media-utils
from tree_utils.struct_tree_out import print_tree

path = r'../../TorchCliPublic'
exclude_dirs_set = {'using_files', '__init__.py', 'static', 'LICENSE', 'Generated', 'data', 'test', 'font', 'from',
                    'image'}
print_tree(directory=path, exclude_dirs=exclude_dirs_set)
