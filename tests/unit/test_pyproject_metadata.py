from __future__ import annotations

from pathlib import Path
import tomllib


def _load_pyproject() -> dict:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    return tomllib.loads(pyproject.read_text())


def _load_pre_commit_config() -> str:
    config = Path(__file__).resolve().parents[2] / ".pre-commit-config.yaml"
    return config.read_text()


def test_python_and_runtime_dependencies_are_explicit() -> None:
    data = _load_pyproject()

    assert data["project"]["requires-python"] == ">=3.11,<3.13"

    dependencies = data["project"]["dependencies"]
    assert "google-cloud-storage>=3.9.0" in dependencies
    assert "soundfile>=0.13.1" in dependencies
    assert not any(dep.startswith("tensorflow") for dep in dependencies)


def test_tensorflow_extras_are_platform_specific_and_conflicting() -> None:
    data = _load_pyproject()

    optional = data["project"]["optional-dependencies"]
    assert set(optional) == {"tf-macos", "tf-linux-cpu", "tf-linux-gpu"}
    assert optional["tf-macos"] == [
        "tensorflow-macos>=2.16,<2.17; sys_platform == 'darwin' and platform_machine == 'arm64'",
        "tensorflow-metal>=1.1; sys_platform == 'darwin' and platform_machine == 'arm64'",
    ]
    assert optional["tf-linux-cpu"] == [
        "tensorflow>=2.20,<2.21; sys_platform == 'linux'",
    ]
    assert optional["tf-linux-gpu"] == [
        "tensorflow[and-cuda]>=2.20,<2.21; sys_platform == 'linux'",
    ]

    assert data["tool"]["uv"]["conflicts"] == [
        [
            {"extra": "tf-macos"},
            {"extra": "tf-linux-cpu"},
        ],
        [
            {"extra": "tf-macos"},
            {"extra": "tf-linux-gpu"},
        ],
        [
            {"extra": "tf-linux-cpu"},
            {"extra": "tf-linux-gpu"},
        ],
    ]


def test_dev_dependency_group_contains_project_tooling() -> None:
    data = _load_pyproject()

    dev = data["dependency-groups"]["dev"]
    assert dev == [
        "honcho>=1.1.0",
        "httpx>=0.28.1",
        "pre-commit>=4.3.0",
        "pyright>=1.1.408",
        "pytest>=9.0.2",
        "pytest-asyncio>=1.3.0",
        "pytest-cov>=6.0",
        "pytest-watch>=4.2",
        "ruff>=0.15.5",
        "supervisor>=4.3.0",
    ]


def test_pyright_configuration_targets_python_source_scopes() -> None:
    data = _load_pyproject()

    assert data["tool"]["pyright"] == {
        "include": ["src/humpback", "scripts", "tests"],
        "extraPaths": ["src"],
        "pythonVersion": "3.11",
        "typeCheckingMode": "standard",
    }


def test_pre_commit_runs_pyright_via_uv() -> None:
    config = _load_pre_commit_config()

    assert "id: pyright" in config
    assert "entry: uv run pyright" in config
    assert "pass_filenames: false" in config
    assert (
        "files: ^(src/humpback/.*\\.py|scripts/.*\\.py|tests/.*\\.py|"
        "pyproject\\.toml|\\.pre-commit-config\\.yaml)$"
    ) in config
