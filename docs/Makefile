# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: all sphinx_help worktree html pdf deploy clean

all: html

help:
	@echo "Builds Watershed Workflow documentation"
	@echo "---------------------------------------"
	@echo "To build WW documentation from scratch (new repo setup, do this once):"
	@echo "  > make worktree"
	@echo ""
	@echo "To build input spec/user guide:"
	@echo "  > make"
	@echo ""
	@echo "To deploy to github:"
	@echo "  > make deploy"

sphinx_help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

worktree:
	cd deploy && git worktree add html gh-pages

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

pdf:
	@$(SPHINXBUILD) -M pdf "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

deploy_master:
	mkdir -p deploy/html/dev
	cp -r build/html/* deploy/html/dev/

deploy_stable:
	mkdir -p deploy/html/stable
	cp -r build/html/* deploy/html/stable/

deploy:
	cd deploy/html && \
	git add --all && \
	git commit -m "Deployment to gh-pages" && \
	git push origin gh-pages

clean:
	rm -rf build/html

allclean:
	rm -rf build/*


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
#%: Makefile
#	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

