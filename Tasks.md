How to apply and verify the fixes locally
	1.	Patch the code
	•	tests/conftest.py

import os, pytest

@pytest.fixture(autouse=True, scope="session")
def _bootstrap_env():
    os.environ.setdefault("JWT_SECRET_KEY", "test-secret")


	•	montage/cli/run_pipeline.py

-from ..core.pipeline import run_pipeline
+from montage.core.pipeline import run_pipeline


	•	smart_track.py (around L211)

bbox = self._single_frame_face_crop(frame)
self._last_bbox = bbox
return bbox


	•	memory_manager.py (around L627)

import gc
if self._should_collect():
    freed = gc.collect()
    self.logger.debug("GC freed %d objects", freed)


	•	memory_manager.py (around L712)

self.logger.warning("High memory pressure: %.1f MB avail", available_mb)
self._downgrade_quality_mode()


	2.	Run the validation suite

# stub-scan must be zero
python scripts/stub_scan.py > stub_scan.out
cat stub_scan.out                     # expect “0”

# unit tests
pytest -q | tee pytest_summary.txt    # expect all dots

# coverage
coverage run -m pytest
coverage report > coverage_after.txt
cat coverage_after.txt                # should remain ≥ 85 %


	3.	What to send back
	•	stub_scan.out
	•	pytest_summary.txt
	•	coverage_after.txt

Once those three show 0 stubs, all tests green, and coverage ≥ 85 %, the repo is back in compliance and the branch can be merged.