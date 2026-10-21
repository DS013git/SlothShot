# SlothShot demo script (about 2 minutes)

The story to tell: **"My screenshots are private, so nothing here touches the internet. I'll prove it by turning Wi-Fi off."**

Everything below uses **synthetic** screenshots (generated fictional content), so you can record safely. Never record real chats, one-time passwords or bank pages.

The search results shown in this script were rehearsed on a real index before writing it. OCR takes about 3 s per image on an i3, so timings below are approximate. **Rehearse once** and trim to fit.

## Prepare (before you hit record, Wi-Fi still ON)

Run these once from the repository root in PowerShell with the venv active. This part may take a minute and a half while OCR reads six images.

```powershell
# 1. Make synthetic screenshots and pick a small, memorable demo set.
python scripts\make_sample_data.py --output data\demo\source --count 100
New-Item -ItemType Directory -Force data\demo\shots | Out-Null
"terminal_000","terminal_006","chat_004","chat_007","settings_005","tricky_otp_message" |
  ForEach-Object { Copy-Item "data\demo\source\$_.png" data\demo\shots\ }

# 2. A demo config: separate database, but reuse the real model cache so nothing re-downloads.
@'
[paths]
watch_folder = "shots"
data_dir = "appdata"
model_cache = '%LOCALAPPDATA%\SlothShot\models'
'@ | Set-Content data\demo\config.toml

# 3. Pre-flight: this is where the one-time model download happens, if it hasn't yet.
slothshot doctor --config data\demo\config.toml
```

Check that `doctor` says `Embedding model cache: ready` and `SQLite FTS5: available`. If the model is not cached yet, run `slothshot index data\demo\shots --config data\demo\config.toml` once while online, then delete `data\demo\appdata` so the demo indexes live.

Keep one image back for the live "new screenshot" moment: `data\demo\source\code_003.png` (a Python "database is locked" snippet).

Open three windows side by side: a terminal, File Explorer at `data\demo\shots`, and a browser. Open Task Manager's Performance tab if you want to show RAM.

## The script

| Time | What you do | What you say |
| --- | --- | --- |
| **0:00** | Show the Windows Wi-Fi flyout. Turn **Wi-Fi off** and let the viewer see the "no internet" icon. Optionally run `ping 1.1.1.1` to show it fails. | "First, I'm switching off my internet. Everything from here on runs on this laptop: an Intel i3, no GPU, 8 GB of RAM." |
| **0:15** | `slothshot index data\demo\shots --config data\demo\config.toml` | "SlothShot reads every screenshot with an open-source OCR model and stores the text locally. It never renames or moves my files." (Wait about 20 s. The log shows each image being read. Optionally glance at RAM in Task Manager.) |
| **0:40** | `slothshot search "address already in use" --mode keyword --config data\demo\config.toml` | "Exact words work. This finds the terminal screenshots." (Expect `terminal_000.png` and `terminal_006.png`.) |
| **0:55** | `slothshot search "docker port problem" --config data\demo\config.toml` | "Now by meaning. I never typed those words, but it finds the Docker error." (Expect `terminal_000.png` first.) |
| **1:10** | `slothshot search "friend sent recipe ingredients" --mode keyword --config data\demo\config.toml` then the same with `--mode hybrid` | "Keyword search finds nothing, because those words are not on the screen. The hybrid search understands what I mean." (Keyword prints `No matching screenshots found.` Hybrid returns `chat_007.png` first. This is the key moment.) |
| **1:30** | Start the UI: `slothshot ui --config data\demo\config.toml`. In the browser search `bank one time password code`. | "There's a local UI too. And yes, this finds the one-time password screenshot. That is exactly why this must never leave my computer." (Expect `tricky_otp_message.png` first.) |
| **1:45** | In a second terminal start `slothshot watch --config data\demo\config.toml`. Drag `data\demo\source\code_003.png` into `data\demo\shots` in Explorer, wait about 6 to 10 s, search `database is locked` in the UI. | "Fully lazy mode: I drop a new screenshot in and a few seconds later it's searchable. No clicking. At login this starts by itself." |
| **1:58** | Show the Wi-Fi icon still off. | "Still offline. Open-source models, runs on a cheap laptop, costs nothing, and my screenshots never leave this machine." |

## If something goes wrong

- **A search returns nothing unexpectedly.** Check that `index` finished (look for `0 failed`). Re-running `index` is safe.
- **The first search is slower than later ones.** The embedding model loads on first use. Run one throwaway search before recording.
- **The UI says there is nothing indexed.** Pass the same `--config data\demo\config.toml` to `slothshot ui`.
- **The UI will not start or shows an old page.** An earlier `slothshot ui` may still be running on port 8501. Press Ctrl+C in that terminal, then start it again.
- **Model download attempted while offline.** The model was not cached. Turn Wi-Fi on, run the Prepare step, and try again.
- **Dropped file does not appear.** Make sure `slothshot watch` is running against `data\demo\config.toml`. Look for `Indexed screenshot:` in its output.

## Clean up

```powershell
Remove-Item -Recurse -Force data\demo
```

`data\` is gitignored, so nothing from the demo can be committed by accident.

## Recording tips

- Record at 1080p with a large terminal font (18 pt or more).
- Export a short GIF of the 1:10 moment (keyword finds nothing, hybrid finds the recipe) for the top of the README.
- Do not claim speed numbers beyond what `docs/BENCHMARKS.md` measured: about 3 s of OCR per screenshot on this CPU.
