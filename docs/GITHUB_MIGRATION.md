# GitHub migration notes

The GitHub repository should prioritize source code, documentation, tests, and small result
summaries. Large generated files should stay local or be shared separately through GitHub
Releases, Git LFS, or an external artifact store.

## Keep in Git

- `src/`, `scripts/`, `tests/`, `templates/`, and root PowerShell entry points.
- `README.md`, `LICENSE`, and files under `docs/`.
- Small JSON metric summaries that document the reported results.

## Keep out of Git

- `.venv/`, `.uv-cache/`, and dependency caches.
- `work/data/`, because the datasets have their own sources and licenses.
- `work/*.csv` and `work/*.json`, because they are generated caches.
- `outputs/**/*.joblib` and prediction CSVs, unless intentionally published with Git LFS or
  as a release artifact.
- Poster drafts, old zip packages, PDFs, and duplicate packaged copies.

## Recreate on another computer

```powershell
git clone REPOSITORY_URL
cd heartabnormaldetection
.\setup.ps1
.\run_app.ps1
```

For full retraining, download the datasets described in `docs/DATASET.md` and run the
appropriate pipeline:

```powershell
.\run_pipeline.ps1
.\run_research.ps1
```
