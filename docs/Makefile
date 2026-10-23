# Sphinx makefile for DecLib++ docs.

SPHINXBUILD ?= sphinx-build
SPHINXOPTS  ?=
SOURCEDIR    = .
BUILDDIR     = _build

.PHONY: help html strict linkcheck clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# What CI runs: fail on any warning, keep going to report them all.
strict:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" -W --keep-going $(SPHINXOPTS)

linkcheck:
	@$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)" api/cpp api/generated
