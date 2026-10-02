# Tests

Run before accepting changes (uses the source tree, not any installed phonlab):

    python -m pytest test/ -q

- `test_api.py` – every name in `phonlab/__init__.pyi` imports and is callable.
- `test_behavior.py` – functions checked against known answers on synthetic signals.
- `test_regression.py` – golden-output snapshots (`test/golden/*.npz`) of each
  analysis function on fixed example audio. A failure means output changed. If
  the change is intended, run `python -m pytest test/test_regression.py --update-golden`
  and commit the updated snapshots so the diff can be reviewed.
- `test_textgrid*.py`, `test_interpolate_measures.py` – TextGrid parser and tidy utilities.
