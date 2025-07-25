# Minimal conftest.py for isolated unit tests
# No Docker dependencies
import pytest

# Override any fixtures that might cause Docker issues
@pytest.fixture(scope="session")
def docker_services():
    """Mock docker services to prevent container startup"""
    return None

@pytest.fixture
def pg_container():
    """Mock postgres container"""
    return None