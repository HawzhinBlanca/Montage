import subprocess
import os

def test_run_pipeline_help():
    env = os.environ.copy()
    env.setdefault("JWT_SECRET_KEY", "test-key")
    env.setdefault("DATABASE_URL", "sqlite:///:memory:")
    result = subprocess.run([
        'python', '-m', 'montage.cli.run_pipeline', '--help'
    ], capture_output=True, text=True, env=env)
    assert result.returncode == 0
    assert '--plan-only' in result.stdout 