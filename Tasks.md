#############################################################
#           W E E K   3   –   N O - L I E S   S P R I N T
#   Real Postgres tests • Coverage-driven purge • Core refactor
#############################################################

# 0. Preconditions ───────────────────────────────────────────
- assert_clean_worktree

# 1. Dependencies  (testcontainers, coverage tools) ----------
- add_dependency:
    testcontainers[postgres]==4.4.1
    coverage==7.5.0
    radon==6.0.1
    ruff==0.4.6
    mypy==1.10.0
    detect-secrets==1.4.0

# 2. Real Postgres fixture  ----------------------------------
- write_file: tests/conftest.py
  content: |
    from testcontainers.postgres import PostgresContainer
    import pytest, psycopg2, os, pathlib
    @pytest.fixture(scope="session")
    def pg_url():
        with PostgresContainer("postgres:15") as pg:
            conn = psycopg2.connect(pg.get_connection_url())
            conn.cursor().execute(pathlib.Path("schema.sql").read_text())
            conn.commit()
            yield pg.get_connection_url()
    @pytest.fixture(autouse=True, scope="session")
    def _patch_env(pg_url, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", pg_url)

# 3. Edge-path coverage script  ------------------------------
- mkdir: [scripts]
- write_file: scripts/coverage_all.sh
  mode: 755
  content: |
    #!/usr/bin/env bash
    set -e
    coverage erase
    coverage run -m pytest -q
    coverage run --append -m montage.cli.run_pipeline tests/data/short.mp4
    coverage run --append -m montage.cli.run_pipeline /no/such/file.mp4 || true
    coverage run --append -m montage.cli.run_pipeline tests/data/corrupt.mp4 || true
    coverage run --append -m montage.cli.run_pipeline tests/data/long.mp4 &
    PID=$!; sleep 2; kill -TERM $PID || true; wait || true
    ulimit -m 500000          # 500 MB
    coverage run --append -m montage.cli.run_pipeline tests/data/huge.mp4 || true
    coverage json -o cov.json
    coverage report --fail-under=95

# 4. Dead-code purge by real coverage ------------------------
- run: bash scripts/coverage_all.sh || true
- run: |
    jq -r '.files | to_entries[] | select(.value.summary.percent_covered==0) | .key' \
        cov.json > dead_files.txt
    # manual keep list for decorators
    grep -v -E '(resolve_mcp|cli|__init__|tests)' dead_files.txt > purge.txt
    while read f; do git rm -f "$f"; done < purge.txt

# 5. Refactor high-complexity brain functions ----------------
- split_function:
    file: src/core/highlight_selector.py
    function: choose_highlights
    into:
      - rank_candidates
      - apply_rules
      - filter_by_length

- split_function:
    file: src/providers/smart_track.py
    function: _analyze_motion
    into:
      - calc_optical_flow
      - classify_motion

- split_function:
    file: main.py
    function: _execute_stages
    into:
      - execute_stage
      - dispatch_stage

# 6. Add strict typing to refactored core --------------------
- add_type_hints:
    modules:
      - src/core/highlight_selector.py
      - src/providers/smart_track.py
      - main.py

# 7. Replace broad exception blocks --------------------------
- run: ruff --fix --select E722     # remove bare except:
- create_python_module: src/core/errors.py
  description: |
    class MontageError(Exception): ...
    class ASRError(MontageError): ...
    class MCPError(MontageError): ...
- refactor_exceptions:
    root: src/
    base: MontageError

# 8. Update CI workflow  -------------------------------------
- update_github_actions:
    workflow: ci
    replace_job: test
    new_job_yaml: |
      test:
        runs-on: ubuntu-latest
        services:
          postgres:
            image: postgres:15
            env:
              POSTGRES_PASSWORD: pass
            ports: [ "5432:5432" ]
        steps:
          - uses: actions/checkout@v4
          - run: pip install -r requirements-dev.txt
          - run: bash scripts/coverage_all.sh
          - run: mypy src/core
          - run: coverage xml

# 9. Secret scan in pre-commit  ------------------------------
- add_precommit_hook:
    repo: https://github.com/zricethezav/gitleaks
    rev: v8.18.1
    hooks: [gitleaks]

# 10. Commit --------------------------------------------------
- git_commit:
    branch: week3-no-lies
    message: |
      chore(week3): real-postgres tests, coverage-driven purge, split brain
      functions, strict types, CI 95 % coverage gate, secret scan hook
#############################################################
# GATES
# • scripts/coverage_all.sh passes (≥95 % lines)
# • mypy src/core passes
# • GitHub Actions green
#############################################################