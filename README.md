# SlothShot

A local, offline librarian for screenshots. SlothShot reads text from images
with CPU ONNX OCR, indexes it in SQLite, and finds screenshots by keywords or
meaning with local ONNX embeddings.

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

Doctor checks the configured embedding model's local cache, the Python
environment, and SQLite FTS5 support. Phase 1 bundles its OCR model files with
RapidOCR, so OCR needs no separate model download.

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
stores metadata, OCR text, and local text embeddings in SQLite. It skips
unchanged files by SHA-256 and reuses OCR and vectors when a file is renamed or
copied.

```powershell
slothshot index "$env:USERPROFILE\Pictures\Screenshots"
slothshot index .\synthetic-screenshots --config .\config.toml
```

The first indexing or semantic-search run downloads the configured embedding
model if needed. Later uses load it from the cache, with network access disabled.

## Search indexed text

Keyword search uses SQLite FTS5, ranks matches with BM25, and highlights the
matching text in the snippet. Word prefixes work while you type. SlothShot
first looks for all query terms; if none match together, it retries with any
term so longer queries can still return candidates.

```powershell
slothshot search "docker port" --mode keyword
slothshot search "address already in" --mode keyword --limit 5
slothshot search "docker port problem" --mode semantic
slothshot search "docker port problem" --mode hybrid
```

Use `--mode semantic` for meaning-based retrieval or `--mode hybrid` (the
default) to combine both ranked lists. Set `models.embedding` in TOML to choose
a different FastEmbed-supported model.

## Watch for new screenshots

The watcher scans the folder once on startup, then watches it recursively. It
waits for each image's size and modification time to settle before OCR, ignores
common temporary and partial file names, and sends work to one background
indexing thread. Stop it with Ctrl+C.

```powershell
slothshot watch
slothshot watch "C:\path\to\screenshots" --config .\config.toml
```

## Browse with the local UI

The Streamlit page binds to this computer only. It opens SQLite read-only,
caches small thumbnails in memory, and can launch a separate background index
command from **Index now**. Use the code block's copy button to copy a result's
path. Stop the UI with Ctrl+C.

```powershell
slothshot ui
slothshot ui --config .\config.toml
```
