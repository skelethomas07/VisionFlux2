import pytest


def test_application_modules_import_without_running_analysis():
    pytest.importorskip("streamlit")
    import app  # noqa: F401
    import pipeline.analyzer  # noqa: F401
    import pipeline.review  # noqa: F401
    import ui.figures  # noqa: F401
    import ui.measurement_canvas  # noqa: F401
