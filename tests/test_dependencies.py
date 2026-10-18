"""The installed package's dependency metadata must tell the truth.

The NumPy backend imports SciPy at runtime, and the compiled extension is
built against NumPy 2 headers so that a single wheel serves NumPy 1.x and
NumPy 2.x. Both facts belong in the runtime requirements that pip sees:
SciPy declared where it is actually used, NumPy without an upper bound that
would force a downgrade on environments already using NumPy 2.

The same metadata declares which interpreters the package supports. Pip
refuses to install on an excluded interpreter, so an honest range is what
keeps unsupported Pythons out and supported ones installed.

A requirement that pip cannot resolve is as unusable as one that is missing:
on Windows the MSVC runtime DLLs come from `msvc-runtime`, which ships
per-CPython-ABI wheels and has none for 3.15, so neither the build nor the
runtime requirements may ask for it on CPython 3.15.
"""
import importlib.metadata
import pathlib
import re

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 has no tomllib in the stdlib.
    tomllib = None

# The supported CPython range, 3.10 through 3.15.
SUPPORTED_PYTHON_MINORS = frozenset(range(10, 16))

PYTHON_CLASSIFIER = re.compile(r"Programming Language :: Python :: 3\.(\d+)$")

# Requirements that pip installs on Windows only, and the newest supported
# interpreter each of them has a wheel for. A Windows-only requirement is
# declared for 3.10 up to the value recorded here; a package whose wheels stop
# earlier must not be asked for on the interpreters beyond it. Adding one means
# adding a line here, which keeps the decision (and its reason) visible.
WINDOWS_ONLY_NEWEST_WHEELED_MINOR = {
    # msvc-runtime publishes per-ABI wheels (cp311-cp314 in 14.44.35112); no
    # cp315 wheel exists, so it stops at 3.14.
    "msvc-runtime": 14,
}

# A Windows interpreter as pip would report it, to evaluate markers against
# every supported interpreter regardless of the one running the tests.
WINDOWS_ENVIRONMENT = {
    "os_name": "nt",
    "platform_machine": "AMD64",
    "platform_release": "10",
    "platform_system": "Windows",
    "platform_version": "10.0.26100",
    "sys_platform": "win32",
}


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


def python_version_for_minor(minor):
    return f"3.{minor}"


def applies_on_windows(requirement, minor):
    """Would pip install this requirement on Windows CPython 3.<minor>?"""
    if requirement.marker is None:
        return True
    environment = dict(WINDOWS_ENVIRONMENT, python_version=python_version_for_minor(minor))
    return requirement.marker.evaluate(environment)


def windows_only_requirements():
    """The runtime requirements pip installs on Windows but not on Linux."""
    return {
        requirement.name.lower(): requirement
        for requirement in runtime_requirements()
        if requirement.marker is not None
        and applies_on_windows(requirement, min(SUPPORTED_PYTHON_MINORS))
    }


def build_system_requirements():
    pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("pyproject.toml is not next to the tests directory")
    text = pyproject.read_text()
    if tomllib is not None:
        return tomllib.loads(text)["build-system"]["requires"]
    return build_system_requirements_without_tomllib(text)


def build_system_requirements_without_tomllib(text):
    """Read `[build-system].requires` on Python 3.10, which has no tomllib."""
    section = re.search(
        r"^\[build-system\](?P<body>.*?)(?=^\[|\Z)", text, re.MULTILINE | re.DOTALL
    )
    assert section, "[build-system] not found in pyproject.toml"
    requires = re.search(
        r"requires\s*=\s*\[(?P<items>.*?)\]", section.group("body"), re.DOTALL
    )
    assert requires, "[build-system].requires not found in pyproject.toml"
    return re.findall(r"""['"]([^'"]+)['"]""", requires.group("items"))


def test_windows_only_requirements_are_limited_to_interpreters_with_wheels():
    """A Windows-only requirement must not be declared where it cannot install.

    `msvc-runtime` is the only such requirement: it puts the MSVC runtime DLLs
    into `sys.prefix` and `Scripts`, and its wheels are tagged per CPython ABI
    with no cp315 wheel published. Declaring it for Windows on 3.15 therefore
    left the newest supported interpreter uninstallable, which is what broke
    the win-amd64-cp315 wheel build. It must stop at 3.14, while 3.10-3.14
    keep it: the wheel links against the CRT and does not bundle it.
    """
    windows_only = windows_only_requirements()
    assert set(windows_only) == set(WINDOWS_ONLY_NEWEST_WHEELED_MINOR), (
        "a new Windows-only requirement was declared; record the newest "
        "supported interpreter that has a wheel for it in "
        "WINDOWS_ONLY_NEWEST_WHEELED_MINOR"
    )
    for name, newest_wheeled_minor in WINDOWS_ONLY_NEWEST_WHEELED_MINOR.items():
        for minor in SUPPORTED_PYTHON_MINORS:
            applies = applies_on_windows(windows_only[name], minor)
            assert applies == (minor <= newest_wheeled_minor), (
                f"{name} is {'declared' if applies else 'not declared'} on "
                f"Windows CPython {python_version_for_minor(minor)}"
            )


def test_msvc_runtime_is_declared_only_on_windows():
    """The CRT is a Windows concern; other platforms must not be asked for it."""
    requirement = windows_only_requirements()["msvc-runtime"]
    linux = dict(
        WINDOWS_ENVIRONMENT,
        os_name="posix",
        platform_machine="x86_64",
        platform_system="Linux",
        python_version="3.10",
        sys_platform="linux",
    )
    assert not requirement.marker.evaluate(linux)


def test_build_requirements_do_not_include_msvc_runtime():
    """The MSVC runtime is not a build tool, so the build must not need it.

    `msvc-runtime` copies CRT DLLs into an environment prefix; it contributes
    nothing to compiling or linking the extension, because the MSVC or clang
    toolchain already provides the CRT headers and import libraries. CMake
    also installs only the FFTW DLLs into the wheel, so the wheel has never
    carried the CRT and the runtime requirement covers the user.

    Listing it in `[build-system].requires` also made the build environment
    itself unresolvable: with no cp315 wheel published, the isolated
    environment for CPython 3.15 could not satisfy it and the wheel build
    failed before compiling anything.
    """
    names = {
        Requirement(text).name.lower() for text in build_system_requirements()
    }
    assert "msvc-runtime" not in names
