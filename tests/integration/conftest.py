"""
Overrides tests/conftest.py's _hermetic_database fixture (same name, so
pytest resolves this closer one for every test under this directory) --
integration tests need a REAL psycopg2.connect, but only ever against
TEST_DATABASE_URL on the throwaway instance (see each test file's own
_real_db fixture, which explicitly sets logger.DATABASE_URL = TEST_DATABASE_URL
before connecting). This file exists solely to neutralize the parent
conftest's poison pill here -- it does not set up or connect to anything
itself, and never touches the real DATABASE_URL.
"""
import pytest


@pytest.fixture(autouse=True)
def _hermetic_database():
    yield
