# v28 Refactor

This folder contains a modularized version of `v28.py`.

## How to run

1. Put your JSON files under `data/` (same as before).
2. Run:

```bash
python run_v28_refactor.py
```

Outputs will be written to:
- `ver28_all/` for ALL
- `ver28_negOnly/` for NEG_ONLY

## Bugfix

The output-directory routing is now robust even if `configure_output_dirs()` is
called with a path-like string. ALL and NEG_ONLY will never collapse into the
same folder under normal use.
