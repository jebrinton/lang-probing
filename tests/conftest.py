"""Project-level pytest configuration.

Registers custom markers used by individual test modules so that
`pytest -m gpu` / `pytest -m slow` work without warnings.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: requires a CUDA GPU + model load (Llama-3.1-8B + SAE)"
    )
    config.addinivalue_line(
        "markers", "slow: takes more than a few seconds to run"
    )
