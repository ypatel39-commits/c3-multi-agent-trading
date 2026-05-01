def test_smoke():
    """Sanity check: package imports without exploding."""
    import c3_multi_agent_trading  # noqa: F401

    assert c3_multi_agent_trading.__version__
