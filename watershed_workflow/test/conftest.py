import pytest


def pytest_addoption(parser):
    parser.addoption('--plot', action='store_true', default=False,
                     help='Show matplotlib plots during test runs')


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'network: marks tests that require live network access (deselect with -m "not network")'
    )
