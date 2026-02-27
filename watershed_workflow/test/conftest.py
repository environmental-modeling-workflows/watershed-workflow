def pytest_addoption(parser):
    parser.addoption('--plot', action='store_true', default=False,
                     help='Show matplotlib plots during test runs')
