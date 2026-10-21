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

## Phase 4: semantic and hybrid search

FastEmbed 0.9.0's installed API accepts `cache_dir`, `threads`, `cuda`, and
`local_files_only` through `TextEmbedding`. Its supported-model metadata lists
`BAAI/bge-small-en-v1.5` as a 384-dimensional, MIT-licensed model. SlothShot
checks the configured Hugging Face snapshot for the model ONNX file and
tokenizer files before loading: a cached model is opened with
`local_files_only=True`; only a cache miss permits the first-use download.
Inference is CPU-only with two ONNX threads and one small embedding batch at a
time.

OCR text is chunked near 1,600 characters at word boundaries to stay within the
model's 512-token truncation limit. Each vector is stored as float32 bytes with
its model name and source text in SQLite. A schema migration adds that chunk
text to existing Phase 2 databases. Semantic search computes exact cosine
similarity against stored chunks and keeps the best chunk per image. Hybrid
search merges FTS5 BM25 and cosine-ranked lists using Reciprocal Rank Fusion;
ranking positions, rather than incomparable raw scores, determine the fused
score. This simple scan fits the intended few-thousand-image scale and avoids
an additional vector database.

## Phase 5: background watcher

The watcher queues its catch-up scan before starting the single SQLite worker,
and starts the filesystem observer before releasing that worker. This ordering
means events arriving during catch-up stay queued behind the scan and are then
processed in order. Catch-up and live file events use the same size-and-mtime
stability check before OCR. The watcher filters temporary suffixes and common
editor lock files, coalesces duplicate create/modify notifications briefly,
and handles file or directory moves and deletions.

Only the worker thread opens the database connection used for indexing; the
Streamlit UI and command-line search can keep their own reader connections.
SQLite commits each indexed image independently, so a shutdown or process crash
can be recovered by the next catch-up scan. Ctrl+C stops event observation,
drains queued tasks, and closes the worker's database connection. Any image that
has not settled before shutdown is picked up by the next catch-up scan.

## Phase 6: local Streamlit UI

The UI opens the existing SQLite file with `mode=ro` and enables SQLite's
`query_only` setting. Its search operations use short-lived read-only
connections, so searches work whether or not the background watcher is
running. The **Index now** button starts the existing CLI index command as a
separate child process and records its output in the local app-data folder;
it does not turn the UI's connection into a writer or hold the page while OCR
and embedding run.

Search results use three columns of cards with thumbnails, snippets, scores,
and local open-image and open-folder actions. Snippet text is HTML-escaped
before FTS highlight markers become `<mark>` elements. Streamlit's copyable
code block provides a browser-side copy button without sending paths to a
clipboard service. Thumbnails are reduced to a bounded size and cached in
memory; the source images are never copied or modified.

The sidebar reads totals and the last successful index time from SQLite. Its
pending count is a cached estimate of supported image paths not yet recorded
in the index; after an Index now run, use **Refresh stats** to clear the short
cache. The UI logs whether the configured embedding model is already present
in the local cache, and binds its server to `127.0.0.1` with usage statistics
disabled.

## Phase 7: synthetic evaluation and resource measurements

`make_sample_data.py` produces 100 or more screenshot-like PNGs with terminal,
code, chat, and settings content. It varies light and dark themes and includes
smaller text. After the regular screenshots it adds five "tricky" images that
stress known OCR weaknesses: tiny text, low contrast, an OTP-style message,
dense config lines, and a completely blank screen. The ground-truth YAML maps
natural-language queries to known filenames; the blank image has no query on
purpose, so it checks that empty OCR text does not crash anything. All
generated content is fictional and the generator never reads a screenshot
folder.

`eval_search.py` uses PyYAML's safe loader to validate a caller-provided query
mapping, then reports top-1, top-3, and top-5 hit rate plus mean reciprocal rank
for keyword, semantic, and hybrid search. Expected names are compared against
result basenames so private evaluation files can refer to screenshots without
exposing their full paths. It also prints every query a mode did not answer
at rank 1, which is how the failure gallery in `BENCHMARKS.md` was produced.
A private evaluation file named `private_eval*.yaml` is already in
`.gitignore`; pass it with `--ground-truth` and `--db` pointing at your real
index.

`bench.py` measures OCR elapsed time from each reader call, embedding time for
the indexed OCR text, total indexing wall time, and process RSS sampled during
indexing. It fully OCRs and embeds 100 regular plus 5 tricky generated
screenshots. To measure search
scaling without claiming 5,000 OCR model passes, it builds 100-, 1,000-, and
5,000-row temporary SQLite corpora by reusing OCR text and vectors from that
real synthetic index. Keyword, semantic, and hybrid query latency are warmed
and measured repeatedly; each corpus records its SQLite database size.

The benchmark reports the machine and Python version that actually ran it.
Benchmark results are machine-specific; the documented figures are observations
from that run, not promises for every Windows laptop. Private evaluation files
and generated screenshots stay under ignored local data folders.
