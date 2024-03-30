import glob
from os.path import basename, dirname, isfile, join

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
]

import glob
import importlib
import inspect
import os

# Get the current directory path where __init__.py resides
current_dir = os.path.dirname(__file__)

# Find all Python files (assuming they are DistNet versions) in the current directory
version_files = glob.glob(os.path.join(current_dir, "distnet_v*.py"))

# Loop through the version files and import classes dynamically
for version_file in version_files:
    version_name = os.path.splitext(os.path.basename(version_file))[0]
    module_name = f".{version_name}"
    module = importlib.import_module(module_name, package=__name__)
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and name.startswith("DistNet"):
            version_number = version_name.split("_")[1]
            setattr(__import__(__name__), f"DistNet_{version_number}", obj)

del (
    os,
    glob,
    importlib,
    inspect,
    current_dir,
    version_files,
    version_file,
    version_name,
    module_name,
    module,
    name,
    obj,
)
