import pytest

def test_import_hpo_learner():
    """
    Basic smoke-test:
    Ensures that the new HPO module can be imported
    without syntax errors or missing dependencies.
    """
    import rose.hpo.hpo_learner
    assert True

