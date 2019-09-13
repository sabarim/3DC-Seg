import importlib
import pkgutil

import sys


def import_submodules(package_name):
  package = sys.modules[package_name]
  for importer, name, is_package in pkgutil.walk_packages(package.__path__):
    # not sure why this check is necessary...
    if not importer.path.startswith(package.__path__[0]):
      continue
    name_with_package = package_name + "." + name
    importlib.import_module(name_with_package)
    if is_package:
      import_submodules(name_with_package)