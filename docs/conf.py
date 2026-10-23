"""Sphinx configuration for the DecLib++ documentation.

The C++ API is extracted by Doxygen (``Doxyfile``) into XML, rendered by
Breathe, and organised into one page per class/file by Exhale. The Python
API is documented with ``autodoc``/``autosummary`` straight from docstrings
(NumPy style).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = DOCS_DIR.parent

# Make the pure-Python layer importable without installing the package.
sys.path.insert(0, str(REPO_ROOT / "python"))

# -- Project information ------------------------------------------------------

project = "DecLib++"
author = "DecLib++ developers"
copyright = "2026, DecLib++ developers"
release = "0.1.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "breathe",
    "exhale",
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]

# Treat every unresolved cross-reference as a warning (CI runs with -W).
nitpicky = True
nitpick_ignore = [
    ("py:class", "petsc4py.PETSc.Vec"),
    ("py:mod", "petsc4py"),
    ("py:class", "mpi4py.MPI.Comm"),
    # Breathe does not emit namespace targets for qualified names.
    ("cpp:identifier", "declib"),
    ("cpp:identifier", "std"),
    ("cpp:identifier", "std::size_t"),
    ("cpp:identifier", "size_t"),
    ("cpp:identifier", "Dim"),
]

# Types from PETSc, MPI and Kokkos have no Sphinx inventory to link to.
nitpick_ignore_regex = [
    ("cpp:identifier",
     r"(Vec|Mat|MatType|DM|SNES|KSP|MPI_Comm|MATAIJKOKKOS|Petsc[A-Za-z]+|DMStag[A-Za-z]*|Kokkos(::.*)?)"),
    ("cpp:identifier", r"declib::stag"),
]

# Default role for `single backticks`: inline code.
default_role = "code"
highlight_language = "cpp"

todo_include_todos = True

# -- Doxygen ------------------------------------------------------------------
# Run Doxygen here (rather than letting Exhale do it) so that the same
# Doxyfile drives local builds, CMake, CI and Read the Docs.

DOXYGEN_XML = DOCS_DIR / "_build" / "doxygen" / "xml"


def _run_doxygen() -> None:
    (DOCS_DIR / "_build" / "doxygen").mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["doxygen", "Doxyfile"], cwd=DOCS_DIR, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("doxygen not found on PATH; it is required to build the C++ docs") from exc


if os.environ.get("DECLIB_SKIP_DOXYGEN") != "1":
    _run_doxygen()

# -- Breathe ------------------------------------------------------------------

breathe_projects = {"DecLib++": str(DOXYGEN_XML)}
breathe_default_project = "DecLib++"
breathe_default_members = ("members",)
breathe_show_include = True

# -- Exhale -------------------------------------------------------------------

exhale_args = {
    "containmentFolder": "./api/cpp",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "C++ API reference",
    "doxygenStripFromPath": str(REPO_ROOT / "include"),
    "createTreeView": False,
    "exhaleExecutesDoxygen": False,
    "fullToctreeMaxDepth": 2,
    "afterTitleDescription": (
        "Generated from the Doxygen comments in ``include/declib``. "
        "Start with :cpp:class:`declib::DiscreteManifold`, :cpp:class:`declib::Form` "
        "and :cpp:class:`declib::LinearOperator`."
    ),
    "contentsDirectives": False,
}

# -- autodoc / autosummary (Python API) ---------------------------------------

autosummary_generate = True
autosummary_imported_members = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}
# The compiled extension and MPI/PETSc stack are not needed to read docstrings.
autodoc_mock_imports = ["declib._core", "petsc4py", "mpi4py"]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "sequence": ":class:`~collections.abc.Sequence`",
    "callable": ":class:`~collections.abc.Callable`",
    "Topology": ":class:`~declib.topology.Topology`",
    "StagTopology": ":class:`~declib.topology.StagTopology`",
    "Geometry": ":class:`~declib.geometry.Geometry`",
    "GridMapping": ":class:`~declib.mapping.GridMapping`",
    "Orientation": ":class:`~declib.core.Orientation`",
    "Bundle": ":class:`~declib.forms.Bundle`",
    "FormSpace": ":class:`~declib.forms.FormSpace`",
    "Form": ":class:`~declib.forms.Form`",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# Offline builds (no network for intersphinx): skip external inventories and
# don't flag references that only they can resolve.
if os.environ.get("DECLIB_DOCS_OFFLINE") == "1":
    intersphinx_mapping = {}
    nitpick_ignore_regex += [(r"py:(class|obj)", r"(collections\.abc\..*|enum\..*|True|False|None)")]

# -- HTML output --------------------------------------------------------------

html_theme = "furo"
html_title = f"DecLib++ {release}"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/celdredsandia/declib",
    "source_branch": "main",
    "source_directory": "docs/",
}

mathjax3_config = {
    "tex": {"macros": {"dd": r"\mathrm{d}", "hodge": r"\star"}},
}
