import httpx
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--agent-url",
        default="http://localhost:9009",
        help="Agent URL (default: http://localhost:9009)",
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires servers running)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests requiring running servers"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --run-integration is specified."""
    if config.getoption("--run-integration"):
        # Run integration tests
        return
    
    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)

@pytest.fixture(scope="session")
def agent(request):
    """Agent URL fixture. Agent must be running before tests start."""
    url = request.config.getoption("--agent-url")

    try:
        response = httpx.get(f"{url}/.well-known/agent-card.json", timeout=2)
        if response.status_code != 200:
            pytest.exit(f"Agent at {url} returned status {response.status_code}", returncode=1)
    except Exception as e:
        pytest.exit(f"Could not connect to agent at {url}: {e}", returncode=1)

    return url
