from types import SimpleNamespace

import pytest

from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory


class FakeSentenceSplitter(BaseSplitter):
    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        return [part.strip() for part in text.split(".") if part.strip()]


class FakeFixedSplitter(BaseSplitter):
    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        size = self.settings.chunk_size
        return [text[index : index + size] for index in range(0, len(text), size)]


@pytest.fixture(autouse=True)
def reset_factory() -> None:
    SplitterFactory.unregister_provider("sentence")
    SplitterFactory.unregister_provider("fixed")
    yield
    SplitterFactory.unregister_provider("sentence")
    SplitterFactory.unregister_provider("fixed")


def test_factory_creates_registered_provider_from_splitter_settings() -> None:
    SplitterFactory.register_provider("sentence", FakeSentenceSplitter)
    settings = SimpleNamespace(provider="sentence")

    splitter = SplitterFactory.create(settings)

    assert isinstance(splitter, FakeSentenceSplitter)
    assert splitter.split_text("One. Two.") == ["One", "Two"]


def test_factory_creates_registered_provider_from_project_settings() -> None:
    SplitterFactory.register_provider("fixed", FakeFixedSplitter)
    settings = SimpleNamespace(splitter=SimpleNamespace(provider="fixed", chunk_size=3))

    splitter = SplitterFactory.create(settings)

    assert isinstance(splitter, FakeFixedSplitter)
    assert splitter.split_text("abcdefg") == ["abc", "def", "g"]


def test_factory_creates_registered_provider_from_mapping() -> None:
    SplitterFactory.register_provider("fixed", FakeFixedSplitter)
    settings = {"splitter": SimpleNamespace(provider="fixed", chunk_size=2)}

    splitter = SplitterFactory.create(settings)

    assert isinstance(splitter, FakeFixedSplitter)
    assert splitter.split_text("abcd") == ["ab", "cd"]


def test_factory_rejects_unknown_provider() -> None:
    settings = SimpleNamespace(provider="missing")

    with pytest.raises(ValueError, match="unsupported splitter provider: missing"):
        SplitterFactory.create(settings)
