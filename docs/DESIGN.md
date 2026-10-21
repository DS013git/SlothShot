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

## Phase 1: OCR

RapidOCR is loaded only when the first readable image is sent to the engine.
The reader reuses that engine for later images, catches file/model errors as
results, and logs elapsed time on successful and failed calls. Large images are
bounded before inference; optional preprocessing enlarges small screenshots,
uses grayscale, and increases contrast. Per-line confidence is retained for
later inspection instead of being collapsed into one score.

RapidOCR 3.10.0 bundles its small ONNX weights in the installed package. Before
initializing it, SlothShot checks for the three expected model files and logs
their local path. If a file is missing, the reader reports how to restore the
package instead of allowing the OCR library to download a replacement. The
installed RapidOCR default selects ONNX Runtime and has CUDA disabled; SlothShot
installs the CPU `onnxruntime` package.

Dark-mode inversion is an optional configuration setting. We compared it on a
generated terminal screenshot containing “ADDRESS ALREADY IN USE”; both the
normal and inverted image were recognized. Confidence was 0.97017 normally and
0.99098 after inversion. This one sample suggests inversion can help, but is
not broad enough evidence to enable it for every dark screenshot, so it remains
off by default pending a more varied evaluation.

The dependency resolver caught a Phase 0 pin mismatch before installation:
FastEmbed 0.9 excludes Pillow 13 on Python 3.11. Pillow is pinned to 12.3.0,
which satisfies both OCR and embedding dependencies in the resolved install.

## Phase 2: database and indexing

SQLite stores one row per image path, with a SHA-256 content hash, dimensions,
OCR text, confidence, timestamps, and a small status value. The hash is
indexed, but not unique: two paths may point to identical screenshot bytes.
When that happens, SlothShot reuses the OCR metadata from the existing hash and
does not run OCR a second time. This also makes moving a file safe: indexing the
new path restores its existing text before a later scan removes the old path.

The FTS5 table is an external-content index over image paths and OCR text.
SQLite triggers update FTS in the same transaction as the image row, so callers
cannot forget to synchronize the two. The embeddings table is present now and
uses a foreign key with cascading delete; embedding generation is added in
Phase 4.

Each image is hashed in 1 MiB blocks. OCR happens before a write transaction,
then that image and its FTS entry are committed together. A stop or crash can
lose work on the image currently being read, while earlier images stay saved.
Folder cleanup runs only after the scan has completed, and only removes records
under the scanned root. Invalid images are recorded with an error status so a
bad screenshot does not abort the rest of the folder.

## Phase 3: keyword search

Keyword search sends only sanitized word tokens to FTS5. Tokens are quoted and
given a prefix wildcard, so punctuation and words such as `OR` cannot become
FTS query syntax, while partially typed words still match. BM25 ranks the
results; snippets mark the matched OCR text for the CLI and later UI.

Short queries first require all terms, which favors precision. If no image
contains every term, the search retries with OR so a longer natural-language
query can still return candidates. This avoids treating sentence-like search
input as a strict intersection that silently produces no results.
