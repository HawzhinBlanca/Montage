#############################################################
#  W E E K  1   S E C U R I T Y  &  R E F A C T O R  S P R I N T
#############################################################

# 0. Safety pre-flight: stop if any uncommitted work
- run: git diff --quiet || (echo "❌ Uncommitted changes" && exit 1)

# 1. Strip secrets from history & working tree
- run: |
    pip install git-filter-repo detect-secrets
    # scrub .env*. Remove any key-like patterns
    git filter-repo --path .env --path .env.example --invert-paths --force
    # commit placeholder
    echo "OPENAI_API_KEY=<YOUR_KEY>" > .env.example
    git add .env.example && git commit -m "chore(secrets): placeholder envs"

# 2. Rotate & externalise secrets
- add_precommit_hook:
    repo: https://github.com/zricethezav/gitleaks
    rev: v8.18.1
    hooks: [gitleaks]

- write_file: scripts/rotate_keys.sh
  mode: 755
  content: |
    #!/usr/bin/env bash
    echo "Rotate keys in provider consoles, then update AWS Secrets Manager:"
    # aws secretsmanager put-secret-value --secret-id montage/openai --secret-string ...

# 3. Introduce secret manager loader
- create_python_module: src/utils/secret_loader.py
  description: |
    from aws_secretsmanager_caching import SecretCache, SecretCacheConfig
    def get(name: str): ...

- patch_file: src/providers/*  # swap os.getenv -> secret_loader.get

# 4. Enforce project structure
- mkdir: [src/core, src/providers, src/utils, src/cli, scripts, config]
- move_files:
    patterns:
      src/run_pipeline.py: src/cli/run_pipeline.py
      src/analyze_video.py: src/core/analyze_video.py
      src/highlight_selector.py: src/core/highlight_selector.py
      src/ffmpeg_utils.py: src/utils/ffmpeg_utils.py
      src/resolve_mcp.py: src/providers/resolve_mcp.py

# 5. Structured logging & Sentry
- add_dependency: sentry-sdk==2.5.0
- patch_file: src/core/__init__.py
  - append: |
      import logging, sentry_sdk, os
      sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"), traces_sample_rate=0.1)
      logging.basicConfig(
          format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
          level=os.getenv("LOG_LEVEL","INFO"))

# 6. Custom exception hierarchy
- create_python_module: src/core/errors.py
  description: |
    class MontageError(Exception): ...
    class SecretError(MontageError): ...
    class ValidationError(MontageError): ...

- refactor_exceptions:
    root: src/
    base: MontageError

# 7. Tests must pass database-less
- write_file: tests/conftest.py
  content: |
    import pytest, os
    @pytest.fixture(autouse=True, scope="session")
    def env_defaults(monkeypatch):
        monkeypatch.setenv("DATABASE_URL","sqlite:///:memory:")
        monkeypatch.setenv("REDIS_URL","redis://localhost:6379/0")

# 8. Pre-commit config (black, ruff, detect-secrets)
- write_file: .pre-commit-config.yaml
  content: |
    repos:
      - repo: https://github.com/psf/black
        rev: 24.3.0
        hooks: [{id: black}]
      - repo: https://github.com/astral-sh/ruff-pre-commit
        rev: v0.4.6
        hooks: [{id: ruff}]
      - repo: https://github.com/zricethezav/gitleaks
        rev: v8.18.1
        hooks: [{id: gitleaks}]
      - repo: https://github.com/Yelp/detect-secrets
        rev: v1.4.0
        hooks: [{id: detect-secrets}]

# 9. CI update: add secret scan + logging test
- update_github_actions:
    workflow: ci
    add_step_after: "Run tests"
    new_step: |
      - name: Secret scan
        run: gitleaks detect --no-git -v

# 10. Commit & push
- git_commit:
    branch: week1-security-refactor
    message: |
      feat(security,structure): secret purge, secret loader, logging,
      new project layout, pre-commit hooks

#############################################################
#  G A T E S
#  • `pytest -q` green
#  • `gitleaks detect` returns 0
#  • `ruff` + `black --check` clean
#  • No file under src/ at repo root; matches new layout
#############################################################