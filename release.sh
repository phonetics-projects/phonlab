find . \( -name .git -o -name .claude \) -prune -o -name .ipynb_checkpoints -type d -prune -exec rm -rf {} + ; \
find . \( -name .git -o -name .claude \) -prune -o -name .DS_Store -type f -delete ; \
rm -rf phonlab.egg-info build && python3 -m build
