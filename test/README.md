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
- `test_prep_audio.py`, `test_smooth1d.py` – golden comparisons whose snapshots
  (`test/golden/prep_audio_golden.npz`, `smooth1d_golden.npz`) are produced by
  `make_prep_audio_golden.py` and `make_smooth1d_golden.py`.
- `test_module.ipynb` – interactive notebook for eyeballing graphics and timing
  (not run by pytest). It reads `s09003.wav`, `s1202a.wav` and `sf3_cln.csv` from this directory.
- `bench_track_formants_ifc.py` – benchmark and equivalence check for `track_formants()`'s
  `ifc*` methods (not run by pytest): times a git revision of `track_formants_.py` (default
  `master`) against the working tree on the example audio and reports whether the two give
  identical numbers. `python test/bench_track_formants_ifc.py --help` lists the options.
- `data/` holds TextGrid fixtures for the parser tests; `golden/` holds all snapshots.
