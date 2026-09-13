def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires live Ollama model and/or web search; excluded from default runs (see addopts)",
    )