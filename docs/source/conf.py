import importlib
import os
import sys
import inspect

sys.path.insert(0, os.path.abspath("../../src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "simpletensor"
copyright = "2024, Vikram Rangarajan, Ved Karamsetty"
author = "Vikram Rangarajan, Ved Karamsetty"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "numpydoc",
    "sphinx.ext.autosummary",
]
code_url = "https://github.com/VikramRangarajan/simpletensor/tree/main"


def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None

    mod = importlib.import_module(info["module"])
    if "." in info["fullname"]:
        objname, attrname = info["fullname"].split(".")
        obj = getattr(mod, objname)
        try:
            # object is a method of a class
            obj = getattr(obj, attrname)
        except AttributeError:
            # object is an attribute of a class
            return None
    else:
        obj = getattr(mod, info["fullname"])

    try:
        file = inspect.getsourcefile(obj)
        lines = inspect.getsourcelines(obj)
    except TypeError:
        # e.g. object is a typing.Union
        return None
    file = os.path.relpath(file, os.path.abspath("."))
    start, end = lines[1], lines[1] + len(lines[0]) - 1

    return f"{code_url}/{file}#L{start}-L{end}"


templates_path = ["_templates"]
exclude_patterns = ["_static"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navigation_with_keys": True,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "github_url": "https://github.com/VikramRangarajan/simpletensor/",
    "logo": {
        "text": "SimpleTensor",
    },
    "navigation_depth": -1,
}
html_static_path = []

autodoc_default_options = {
    "special-members": "__add__, __sub__, __mul__, __div__, __pow__, __matmul__",
}
numpydoc_show_class_members = False
autosummary_generate = True
autosummary_imported_members = True
