"""Keep source version and PyPI-facing documentation consistent."""

import re
from pathlib import Path

import mavats

ROOT = Path(__file__).resolve().parents[1]


def test_source_version_matches_package_metadata():
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'^version = "([^"]+)"$', metadata, re.MULTILINE)[1]
    assert mavats.__version__ == version


def test_readme_install_command_and_release_notes_match_version():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"mavats=={mavats.__version__}" in readme
    notes = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## {mavats.__version__} " in notes


def test_readme_guide_links_are_absolute_and_point_to_existing_release_sources():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    prefix = f"https://github.com/dx-li/MaVaTS/blob/v{mavats.__version__}/"
    links = re.findall(r"\]\(([^)]+)\)", readme)
    assert links
    for link in links:
        assert link.startswith(("https://", "#")), link
        if link.startswith("https://github.com/dx-li/MaVaTS/blob/"):
            assert link.startswith(prefix), link
        if link.startswith(prefix):
            path = link.removeprefix(prefix).split("#", 1)[0]
            assert (ROOT / path).is_file(), path
