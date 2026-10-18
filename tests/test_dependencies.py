"""The installed package's dependency metadata must tell the truth.

The NumPy backend imports SciPy at runtime, and the compiled extension is
built against NumPy 2 headers so that a single wheel serves NumPy 1.x and
NumPy 2.x. Both facts belong in the runtime requirements that pip sees:
SciPy declared where it is actually used, NumPy without an upper bound that
would force a downgrade on environments already using NumPy 2.

The same metadata declares which interpreters the package supports. Pip
refuses to install on an excluded interpreter, so an honest range is what
keeps unsupported Pythons out and supported ones installed.
"""
import importlib.metadata
import re

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

# The supported CPython range, 3.10 through 3.15.
SUPPORTED_PYTHON_MINORS = frozenset(range(10, 16))

PYTHON_CLASSIFIER = re.compile(r"Programming Language :: Python :: 3\.(\d+)$")


def distribution_metadata():
    try:
        return importlib.metadata.metadata("psnr_hvsm")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("psnr_hvsm is not installed as a distribution")


def runtime_requirements():
    return [
        Requirement(requirement)
        for requirement in distribution_metadata().get_all("Requires-Dist") or []
        if "extra ==" not in requirement
    ]


def declared_python_minors():
    return {
        int(match.group(1))
        for classifier in distribution_metadata().get_all("Classifier") or []
        if (match := PYTHON_CLASSIFIER.search(classifier))
    }


def test_version_is_at_least_0_3_0():
    # 0.3.0 is the release that lifted the Python floor to 3.10, removed the
    # NumPy <2 cap and added the BT.601 precision option; every later release
    # must declare a version at least as high.
    from packaging.version import Version

    assert Version(distribution_metadata()["Version"]) >= Version("0.3.0")


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


def test_requires_python_declares_the_supported_floor():
    requires_python = SpecifierSet(distribution_metadata()["Requires-Python"])
    assert requires_python.contains("3.10.0")
    assert requires_python.contains("3.15.0")
    assert not requires_python.contains("3.9.18"), "3.9 is no longer supported"


def test_classifiers_declare_exactly_the_supported_range():
    assert declared_python_minors() == SUPPORTED_PYTHON_MINORS
