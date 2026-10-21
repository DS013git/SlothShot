# Design notes

## Phase 0: local setup

SlothShot keeps settings, screenshots, OCR text, and model files on the user's
computer. By default, screenshots come from `Pictures/Screenshots`; the
database and model cache live beneath `%LOCALAPPDATA%\SlothShot` on Windows.
There is no service component.

Python's built-in `tomllib` reads configuration so setup does not add a TOML
dependency. `AppConfig` is immutable, resolves paths once, and validates values
before later indexing code can use them. The `doctor` command reports the
environment and checks SQLite FTS5 without loading OCR or embedding models.
Importing model packages can be slow and can trigger downloads, so doctor uses
package metadata only and never downloads models.

The first install pins direct dependencies so a fresh install has a reproducible
starting point. Model downloads are a separate first-use operation; Phase 1 and
Phase 4 will make those cache checks and offline behavior explicit. FastEmbed
and RapidOCR use ONNX Runtime for CPU inference; no PyTorch or GPU runtime is
part of this project.

`slothshot.sqlite3` is stored outside the repository. The ignore rules also
exclude screenshot and evaluation folders so private inputs are not accidentally
added to source control. Synthetic test images will be generated in a later
phase.

## Planned shape

The watcher and manual index command will call the same indexing path. SQLite
will own both metadata and FTS5 text search, while embeddings will be kept as
float32 blobs in that database. Semantic and keyword result lists will be
combined with reciprocal rank fusion. The UI will read this local database and
will not need the watcher to be active.
