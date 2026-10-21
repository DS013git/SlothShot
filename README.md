# SlothShot

A local, offline librarian for screenshots. SlothShot reads text from images
with CPU ONNX OCR and indexes it in a local SQLite database; search arrives in
later phases.

## Phase 0 setup

Requires Python 3.10 or 3.11 on Windows 10/11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
slothshot doctor
```

By default, SlothShot looks in `%USERPROFILE%\Pictures\Screenshots` and stores
its database and model cache under `%LOCALAPPDATA%\SlothShot`. Copy
`config.example.toml` to `%APPDATA%\SlothShot\config.toml` to customize paths.
Run `slothshot doctor --config path\to\config.toml` to check a specific config.

Doctor reports whether the configured model cache contains files and checks the
Python environment and SQLite FTS5 support. Phase 1 bundles its OCR model files
with the RapidOCR package, so there is no separate model download during OCR.
Model licensing and dependency license details will be documented before
release.

## Read one screenshot

RapidOCR's small ONNX models are bundled with the installed package. The OCR
command checks that those local files exist before starting the engine, so it
does not silently fetch them during an OCR run.

```powershell
slothshot ocr "C:\path\to\screenshot.png"
slothshot ocr "C:\path\to\screenshot.png" --no-preprocess
```

Each output line includes RapidOCR's confidence score. Preprocessing bounds
large images, enlarges small screenshots, and converts them to grayscale.
Dark-mode inversion is opt-in with `ocr.invert_dark_mode = true`. On one
synthetic dark-terminal image it improved the model confidence score, but that
single example is not enough to turn it on automatically for all screenshots.

## Index a folder

The index command scans recursively, recognizes supported image formats, and
stores metadata and OCR text in the local SQLite database. It skips unchanged
files by SHA-256 and reuses OCR text when a file is renamed or copied.

```powershell
slothshot index "$env:USERPROFILE\Pictures\Screenshots"
slothshot index .\synthetic-screenshots --config .\config.toml
```

This phase prepares FTS5 and the embeddings table in SQLite. Keyword and
semantic search commands are introduced in later phases.
