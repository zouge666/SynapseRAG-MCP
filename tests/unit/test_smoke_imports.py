"""Smoke tests for the initial project skeleton."""


def test_top_level_packages_import() -> None:
    import core
    import ingestion
    import libs
    import mcp_server
    import observability

    assert core is not None
    assert ingestion is not None
    assert libs is not None
    assert mcp_server is not None
    assert observability is not None
