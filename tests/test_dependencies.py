"""The installed package's dependency metadata must tell the truth.

The NumPy backend imports SciPy at runtime, and the compiled extension is
built against NumPy 2 headers so that a single wheel serves NumPy 1.x and
NumPy 2.x. Both facts belong in the runtime requirements that pip sees:
SciPy declared where it is actually used, NumPy without an upper bound that
would force a downgrade on environments already using NumPy 2.
"""
import importlib.metadata

import pytest
from packaging.requirements import Requirement


def runtime_requirements():
    try:
        metadata = importlib.metadata.metadata("psnr_hvsm")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("psnr_hvsm is not installed as a distribution")
    return [
        Requirement(requirement)
        for requirement in metadata.get_all("Requires-Dist") or []
        if "extra ==" not in requirement
    ]


def test_scipy_is_a_runtime_dependency():
    names = {requirement.name.lower() for requirement in runtime_requirements()}
    assert "scipy" in names


def test_numpy_runtime_requirement_has_no_upper_bound():
    numpy_requirements = [
        requirement
        for requirement in runtime_requirements()
        if requirement.name.lower() == "numpy"
    ]
    assert numpy_requirements, "numpy must be a runtime dependency"
    for requirement in numpy_requirements:
        upper_bounds = [
            str(specifier)
            for specifier in requirement.specifier
            if specifier.operator in ("<", "<=")
        ]
        assert not upper_bounds, f"numpy is capped at {upper_bounds}"
