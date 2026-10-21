# SlothShot

A local, offline librarian for screenshots. Phase 0 sets up the project and
configuration; indexing and search arrive in later phases.

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

The first phase does not download models. Doctor reports whether model cache
files are present and checks the Python environment and SQLite FTS5 support.
Model licensing and dependency license details will be documented before
release.
