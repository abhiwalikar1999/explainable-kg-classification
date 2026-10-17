"""The Streamlit demo runs end to end, and its cloud install stays pinned."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "app" / "demo.py"


def pins(*files) -> dict:
    """Package -> version for every `name==version` line in the files."""
    lines = [
        line.split("#")[0].strip()
        for name in files
        for line in (ROOT / name).read_text(encoding="utf-8").splitlines()
    ]
    return dict(line.split("==") for line in lines if "==" in line)


def test_cloud_requirements_use_the_project_pins():
    cloud = pins("app/requirements.txt")
    project = pins("requirements.txt", "requirements-demo.txt")
    assert "streamlit" in cloud
    assert cloud.items() <= project.items()


def test_demo_runs_and_switches_person():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(DEMO), default_timeout=60).run()
    assert not app.exception
    assert len(app.selectbox[0].options) == 36

    app.selectbox[0].select_index(5).run()
    assert not app.exception
    assert [m.label for m in app.metric] == [
        "True group", "Decision tree", "LightGBM"
    ]
    # The tree path is listed and ends in a predicted group.
    assert any(md.value.startswith("- →") for md in app.markdown)
