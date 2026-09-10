import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MLX_LM_REQUIREMENT = (
    "mlx-lm @ "
    "git+https://github.com/LibraxisAI/mlx-lm-mirror.git"
    "@b5b2865007f864ef5a6378f88159c2bbb733ea43"
)


def _dynamic_dependency_files(pyproject):
    project = pyproject["project"]
    if project.get("dependencies") is not None:
        raise AssertionError("dependencies must have one dynamic setuptools owner")
    if project.get("dynamic", []).count("dependencies") != 1:
        raise AssertionError(
            "project.dependencies must be declared dynamic exactly once"
        )

    files = pyproject["tool"]["setuptools"]["dynamic"]["dependencies"]["file"]
    if files != ["requirements.txt"]:
        raise AssertionError("requirements.txt must remain the sole dependency source")
    return files


def _require_one_canonical_shared_lm(requirements):
    candidates = [
        requirement
        for requirement in requirements
        if requirement.partition("@")[0].strip().lower().replace("_", "-") == "mlx-lm"
    ]
    if candidates != [EXPECTED_MLX_LM_REQUIREMENT]:
        raise AssertionError(
            "mlx-lm must have exactly one canonical immutable source requirement"
        )
    return candidates[0]


def test_setuptools_metadata_propagates_the_exact_shared_lm_requirement():
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)

    dependency_files = _dynamic_dependency_files(pyproject)
    requirements = [
        line.strip()
        for path in dependency_files
        for line in (REPO_ROOT / path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert _require_one_canonical_shared_lm(requirements) == EXPECTED_MLX_LM_REQUIREMENT
    assert "transformers>=5.14.0" in requirements


@pytest.mark.parametrize(
    "replacement",
    [
        [],
        [EXPECTED_MLX_LM_REQUIREMENT, EXPECTED_MLX_LM_REQUIREMENT],
        ["mlx-lm>=0.31.3"],
    ],
    ids=["missing", "duplicate-owner", "unqualified-source"],
)
def test_shared_lm_contract_rejects_missing_or_ambiguous_ownership(replacement):
    requirements = ["mlx>=0.32.0", *replacement, "transformers>=5.14.0"]

    with pytest.raises(
        AssertionError,
        match="exactly one canonical immutable source requirement",
    ):
        _require_one_canonical_shared_lm(requirements)
