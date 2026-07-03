# AIPE Operations

This file keeps repeatable local operations out of chat history.

## Verify The Stack

```bash
docker compose exec api python -m scripts.verify_stack
```

Checks:

- API health endpoint
- active `LLM_MODEL`
- `DEFAULT_PROJECT`
- project profile loading
- terminology and style guide loading
- Qdrant point count for the profile collection

Use a specific profile:

```bash
docker compose exec api python -m scripts.verify_stack --project-id wwm/zh-en
```

If project dependencies are installed in the local Python environment, the same
script can also run as `python3 -m scripts.verify_stack`.

## Run Checks

Docker runtime, matching the API container:

```bash
python3 -m scripts.run_checks
```

Target selected tests:

```bash
python3 -m scripts.run_checks tests/test_project_profiles.py tests/test_ops_scripts.py
```

Local Python runtime:

```bash
python3 -m scripts.run_checks --local
```

## Add A Project Profile

```bash
python3 -m scripts.create_project_profile nrc/zh-en \
  --game "New Game" \
  --language-pair ZH-EN \
  --style-guide data/style_guide/new_game.md \
  --terminology data/terminology/new_game_terms.xlsx \
  --qdrant-collection nrc_en_corpus \
  --web-search-prefix "New Game"
```

The script writes `data/projects/<game>/<source-target>/profile.json` and stores asset
paths relative to the profile directory. It refuses to overwrite an existing
profile unless `--force` is passed.

## Import SDLTM Incrementally

Dry run first:

```bash
python3 -m scripts.import_sdltm_incremental /path/to/TM_DIR \
  --collection yanyun_corpus \
  --progress data/progress/sdltm_incremental_import.json
```

Commit after reviewing the report:

```bash
python3 -m scripts.import_sdltm_incremental /path/to/TM_DIR \
  --collection yanyun_corpus \
  --progress data/progress/sdltm_incremental_import.json \
  --commit
```

The import script deduplicates incoming SDLTM pairs, scans existing Qdrant
payload before embedding, skips equal-or-better existing pairs, and reuses
existing vectors for quality upgrades.

When running through Docker, mount the TM directory into the one-off container:

```bash
docker compose run --rm \
  -v "/path/to/TM_DIR:/tm:ro" \
  --entrypoint python api \
  -m scripts.import_sdltm_incremental /tm \
  --collection yanyun_corpus
```

## Clean Docker Leftovers

Dry run:

```bash
python3 -m scripts.cleanup_dev_stack --include-build-cache
```

Execute conservative cleanup:

```bash
python3 -m scripts.cleanup_dev_stack --include-build-cache --commit
```

This removes stopped compose containers, dangling images, and optionally Docker
build cache. It does not remove Qdrant data by default.

Volume removal is explicit and destructive:

```bash
python3 -m scripts.cleanup_dev_stack --include-volumes --commit
```
