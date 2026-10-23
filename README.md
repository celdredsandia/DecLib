# DecLib
A C++ library for discrete exterior calculus

## Documentation

The docs combine Doxygen + Breathe + Exhale (C++ API) with Sphinx autodoc
(Python API).

```bash
sudo apt-get install doxygen graphviz
pip install -r docs/requirements.txt
make -C docs html          # or: make -C docs strict  (warnings are errors, as in CI)
# or via CMake:
cmake -S . -B build -DDECLIB_BUILD_DOCS=ON && cmake --build build --target docs
```

Set `DECLIB_DOCS_OFFLINE=1` to build without network access (skips intersphinx).
