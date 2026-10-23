Writing documentation
=====================

How the docs are built
----------------------

.. code-block:: text

   include/declib/*.hpp ──Doxygen──▶ XML ──Breathe/Exhale──┐
                                                           ├──▶ Sphinx HTML
   python/declib/*.py ──autodoc/autosummary───────────────┘

* ``docs/Doxyfile`` controls what Doxygen extracts. Only XML is generated.
* ``docs/conf.py`` runs Doxygen, then Exhale writes ``docs/api/cpp/``
  (generated, not committed) and autosummary writes ``docs/api/generated/``.

Building locally
----------------

.. code-block:: bash

   pip install -r docs/requirements.txt   # also needs doxygen and graphviz
   make -C docs html                      # output in docs/_build/html
   make -C docs strict                    # what CI runs: warnings are errors

or through CMake:

.. code-block:: bash

   cmake -S . -B build -DDECLIB_BUILD_DOCS=ON
   cmake --build build --target docs

C++ comment style
-----------------

Use Javadoc-style ``/** ... */`` blocks with ``@brief``, ``@param``,
``@tparam``, ``@return``, and ``\f$ ... \f$`` for inline maths:

.. code-block:: cpp

   /**
    * @brief Apply the discrete exterior derivative \f$d_k\f$.
    * @tparam K    Degree of the input cochain.
    * @param omega Input \f$k\f$-cochain.
    * @return The \f$(k+1)\f$-cochain \f$d\omega\f$.
    */

Put implementation details in a ``detail`` namespace: the Doxyfile excludes
it from the reference.

Python docstring style
----------------------

Use `NumPy-style docstrings <https://numpydoc.readthedocs.io/en/latest/format.html>`_,
and cross-reference the C++ counterpart with ``:cpp:class:`` or
``:cpp:func:``.
