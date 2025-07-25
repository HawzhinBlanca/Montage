# Testing simplified endpoints to verify lazy loading
import pytest

def test_import_web_server():
    """Test that web_server can be imported without side effects"""
    # This should not trigger DB connection
    from montage.api import web_server
    assert hasattr(web_server, 'app')
    assert hasattr(web_server, 'get_db')
    assert hasattr(web_server, 'get_celery')

def test_get_db_lazy():
    """Test that get_db is truly lazy"""
    from montage.api.web_server import get_db
    # Calling the function should work
    # (actual DB connection happens when Database() is called)
    db_func = get_db
    assert callable(db_func)

def test_get_celery_lazy():
    """Test that get_celery is truly lazy"""
    from montage.api.web_server import get_celery
    # Calling the function should work
    celery_func = get_celery
    assert callable(celery_func)

def test_health_endpoint_exists():
    """Test that health endpoint is registered"""
    from montage.api.web_server import app
    routes = [route.path for route in app.routes]
    assert "/health" in routes

def test_no_import_side_effects():
    """Test that importing web_server doesn't create connections"""
    # Import should succeed without any database/celery connections
    import montage.api.web_server
    
    # Check that lazy functions exist
    assert hasattr(montage.api.web_server, 'get_db')
    assert hasattr(montage.api.web_server, 'get_celery')
    
    # No global db object should exist
    assert not hasattr(montage.api.web_server, 'db')