# Phase 7 measurements

Measurements below were collected on 2026-10-21 using only screenshots created
by `scripts/make_sample_data.py`. No personal screenshot folders were read.
The configured BGE model was already cached locally, and RapidOCR reported its
ONNX model files in the installed package. No model download was needed.

## Machine and method

- Windows build reported as `10.0.26200`
- Intel Core i3-1215U, 6 physical cores / 8 logical processors
- Python 3.11.15; 7.65 GiB RAM reported by psutil
- 105 generated PNGs at 1000 x 620 pixels (100 regular screenshots plus 5
  "tricky" ones); default OCR preprocessing enabled
- One complete index pass using the real OCR, embedding, and SQLite code paths
- Search databases at 100, 1,000, and 5,000 records reuse OCR text and vectors
  from that indexed synthetic corpus. Larger sizes measure search and database
  growth; they do not claim additional OCR or embedding inference.
- Search was warmed, then measured for five calls per mode and size. Values are
  median / nearest-rank p95 in milliseconds. OCR p95 uses nearest rank.
- Peak RSS was sampled from the current process every 50 ms during indexing.

## Indexing and resource use

| Measurement | Result |
| --- | ---: |
| Images indexed / failed | 105 / 0 |
| Total indexing wall time (OCR + embedding + DB) | 325.03 s (5 min 25 s) |
| OCR time per image, median | 3.022 s |
| OCR time per image, p95 | 3.344 s |
| Warm embedding time for 104 documents (one batch) | 8.31 s |
| Process RSS before model initialization | 82.5 MiB |
| Peak process RSS during indexing | 586.1 MiB |
| SQLite database after 105 images | 0.34 MiB |

OCR is the slow stage on this CPU: about 3 seconds per screenshot, so a new
screenshot becomes searchable roughly 3 to 4 seconds after it is saved. The
per-image embedding step during indexing took 0.02 to 0.06 s once the model was
loaded. The RSS measurement includes local OCR and embedding model
initialization.

An earlier run of this same benchmark (100 regular images, before the tricky
cases existed) measured a 8.9 s median OCR time and a 14.5 minute index pass.
A separate warm check later the same day measured about 2.9 s per image, so
the earlier figure was most likely slowed by other load on the machine (not
verified). Treat OCR time as sensitive to what else the CPU is doing.

## Search and database size

Each search cell is median / p95 milliseconds from five warmed calls.

| Images | SQLite size | Keyword | Semantic | Hybrid |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.30 MiB | 0.39 / 0.59 | 10.15 / 12.33 | 19.35 / 24.34 |
| 1,000 | 2.60 MiB | 1.81 / 1.98 | 54.13 / 78.00 | 34.77 / 54.88 |
| 5,000 | 12.73 MiB | 1.81 / 2.10 | 142.41 / 228.61 | 108.69 / 233.45 |

Semantic search scans stored embedding rows and calculates cosine similarity
with NumPy, so latency grows with the number of rows. Five repeats per cell is a
small sample, and the ordering of semantic versus hybrid at 1,000 and 5,000
images is within run-to-run noise (hybrid reuses the same scan). These timings
are a single local run, not a guarantee for another machine or disk.

## Synthetic retrieval evaluation

The truth set has 12 queries: eight for the repeated scenario templates (Docker
port binding, package installation, Git conflicts, SQLite locking, meeting
notes, Wi-Fi settings, Node server ports, a recipe) and four for the tricky
images (segmentation fault, payment confirmation, OTP message, package
dependencies). Measured on the 105 indexed screenshots:

| Mode | Top-1 hit | Top-3 hit | Top-5 hit | MRR |
| --- | ---: | ---: | ---: | ---: |
| Keyword | 0.833 | 0.833 | 0.833 | 0.839 |
| Semantic | 1.000 | 1.000 | 1.000 | 1.000 |
| Hybrid | 0.917 | 1.000 | 1.000 | 0.958 |

The sample generator repeats eight scenario templates with different window
numbers, and the truth set marks every matching screenshot as relevant. The
queries were written by the same author as the generator, so they are friendlier
than real life. These scores are a pipeline smoke check on synthetic text, not an
independent measure of general search quality.

## Failure gallery

Search failures, taken from the miss list printed by `eval_search.py`:

- **Keyword, "friend sent recipe ingredients":** no result at all. None of those
  words appear in the OCR text ("lentil soup", "cumin", "lemon"). Semantic search
  found the intended scenario. This is the case semantic search exists for.
- **Keyword, "npm server port already used":** the correct screenshot was only
  at rank 14. Keyword search falls back to OR matching, and "port" and "already"
  also match the Docker screenshots, which outrank the npm ones.
- **Hybrid, same query:** the correct screenshot moved to rank 2; a Docker
  screenshot stayed first because both scenarios contain "address already in
  use", which gives them a strong keyword rank that RRF adds to.
- **Repeated templates:** ranking among screenshots with identical visible text
  is arbitrary. The evaluation treats all repeats as valid hits and cannot
  estimate performance on the variety of a personal collection.

OCR failures, from running the OCR on the five tricky images (read directly
from the OCR output, not a statistical sample):

- **Tiny text (11 px):** `$ ./render --scene city` was read as
  `$./render-scene city`. Spacing and the double dash were lost, although the
  words that matter for search ("Segmentation fault (core dumped)") were read
  correctly.
- **Dense small text (15 px, four lines):** one of the four lines
  (`"express": "4.19.2", "zod": "3.23.8" },`) was dropped entirely, and the last
  line lost a space (`"test": "vitest"}`). Searching for `express` would miss
  this screenshot by keyword.
- **Low contrast (grey on grey):** all three lines were read correctly with
  confidence above 0.98. The preprocessing contrast step probably helps here
  (not isolated by an experiment).
- **OTP message:** read correctly, so the code `482915` is searchable. This is
  intentional, but it also means one-time codes are stored in plain text in the
  local database; see the privacy section of the README.
- **Blank screen:** zero lines and no error. OCR still spent about 3 s on it, so
  large numbers of empty screenshots cost as much time as text-heavy ones.

## Reproduction

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\bench.py --output data\phase7-bench --count 100 --repeats 5
.\.venv\Scripts\python.exe scripts\eval_search.py --ground-truth data\phase7-bench\screenshots\ground_truth.yaml --db data\phase7-bench\indexed.sqlite3
```

To use your own private query set, create a YAML file such as
`private_eval.yaml` (already gitignored) with entries like
`"the bank page about my loan": "Screenshot 2026-03-04 101500.png"` and run
`eval_search.py --ground-truth private_eval.yaml` against your real index.

Benchmark outputs and generated images are under the ignored `data/` folder.
PyYAML 6.0.3 is pinned for safe YAML parsing and is MIT licensed according to
[its PyPI project metadata](https://pypi.org/project/PyYAML/).
