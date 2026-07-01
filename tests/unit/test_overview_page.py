from observability.dashboard.pages.overview import _component_card, _metadata_summary


def test_component_card_uses_compact_configured_summary() -> None:
    card = _component_card(
        {
            "name": "Embedding",
            "provider": "local",
            "detail": "local-hash",
            "status": "configured",
            "metadata": {"dimensions": 128},
        }
    )

    assert "✓ Configured" in card
    assert "dimensions: 128" in card
    assert "<div class=\"synapserag-component-card\"" in card


def test_component_card_marks_disabled_items_without_status_spinner() -> None:
    card = _component_card(
        {"name": "Evaluation", "provider": "none", "detail": "disabled", "status": "disabled", "metadata": {}}
    )

    assert "○ Disabled" in card
    assert "status--disabled" in card


def test_metadata_summary_is_inline() -> None:
    assert _metadata_summary({"persist_path": "data/chroma", "dimension": 128}) == "persist_path: data/chroma · dimension: 128"
