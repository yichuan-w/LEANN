"""Pytest configuration and fixtures for LEANN tests."""

import os
import pytest


@pytest.fixture(autouse=True)
def test_environment():
    """Set up test environment variables."""
    # Mark as test environment to skip memory-intensive operations
    os.environ["CI"] = "true" 
    yield
    

@pytest.fixture(scope="session", autouse=True) 
def cleanup_session():
    """Session-level cleanup to ensure no hanging processes."""
    yield
    
    # Basic cleanup after all tests
    try:
        import psutil
        import os
        
        current_process = psutil.Process(os.getpid())
        children = current_process.children(recursive=True)
        
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
                
        # Give them time to terminate gracefully
        psutil.wait_procs(children, timeout=3)
        
    except Exception:
        # Don't fail tests due to cleanup errors
        pass