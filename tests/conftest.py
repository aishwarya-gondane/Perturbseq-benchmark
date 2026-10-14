import pytest

from psbench.simulate import toy_dataset


@pytest.fixture(scope="session")
def toy():
    return toy_dataset(seed=1)
