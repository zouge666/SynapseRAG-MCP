from __future__ import annotations

from collections.abc import Mapping

from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory


class RecursiveSplitter(BaseSplitter):
    default_chunk_size = 1000
    default_chunk_overlap = 200
    separators = ("\n```", "\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ")

    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        if not isinstance(text, str):
            raise ValueError("splitter text must be a string")
        normalized = text.strip()
        if not normalized:
            return []
        chunk_size = self._positive_int("chunk_size", self.default_chunk_size)
        chunk_overlap = self._non_negative_int("chunk_overlap", self._non_negative_int("overlap", self.default_chunk_overlap))
        if chunk_overlap >= chunk_size:
            raise ValueError("splitter.chunk_overlap must be smaller than chunk_size")
        chunks: list[str] = []
        for block in self._markdown_blocks(normalized):
            if self._is_fenced_code_block(block):
                chunks.append(block)
            else:
                chunks.extend(self._split_recursive(block, chunk_size, 0))
        merged = self._merge(chunks, chunk_size, chunk_overlap)
        return [chunk for chunk in merged if chunk]

    def _markdown_blocks(self, text: str) -> list[str]:
        blocks: list[str] = []
        current: list[str] = []
        code: list[str] = []
        in_code = False
        for line in text.splitlines():
            if line.startswith("```"):
                if in_code:
                    code.append(line)
                    blocks.append("\n".join(code).strip())
                    code = []
                    in_code = False
                else:
                    if current:
                        blocks.append("\n".join(current).strip())
                        current = []
                    code = [line]
                    in_code = True
                continue
            if in_code:
                code.append(line)
                continue
            if line.startswith("#") and line[1:2] in {" ", "#"}:
                if current:
                    blocks.append("\n".join(current).strip())
                current = [line]
                continue
            current.append(line)
        if code:
            blocks.append("\n".join(code).strip())
        if current:
            blocks.append("\n".join(current).strip())
        return [block for block in blocks if block]

    def _split_recursive(self, text: str, chunk_size: int, separator_index: int) -> list[str]:
        if len(text) <= chunk_size or self._is_fenced_code_block(text):
            return [text]
        if separator_index >= len(self.separators):
            return self._split_by_size(text, chunk_size)
        separator = self.separators[separator_index]
        parts = self._split_by_separator(text, separator)
        if len(parts) == 1:
            return self._split_recursive(text, chunk_size, separator_index + 1)
        result: list[str] = []
        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue
            result.extend(self._split_recursive(stripped, chunk_size, separator_index + 1))
        return result

    def _split_by_separator(self, text: str, separator: str) -> list[str]:
        parts = text.split(separator)
        if len(parts) == 1:
            return [text]
        result = [parts[0]]
        for part in parts[1:]:
            result.append(f"{separator}{part}")
        return result

    def _split_by_size(self, text: str, chunk_size: int) -> list[str]:
        return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]

    def _merge(self, parts: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
        chunks: list[str] = []
        current = ""
        for part in parts:
            if current and self._starts_heading(part):
                chunks.append(current.strip())
                current = ""
            candidate = self._join(current, part)
            if current and len(candidate) > chunk_size:
                chunks.append(current.strip())
                current = self._overlap_text(current, chunk_overlap)
                current = self._join(current, part)
                while len(current) > chunk_size and not self._is_fenced_code_block(current):
                    chunks.append(current[:chunk_size].strip())
                    current = self._overlap_text(current[:chunk_size], chunk_overlap) + current[chunk_size:]
            else:
                current = candidate
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _join(self, left: str, right: str) -> str:
        if not left:
            return right.strip()
        if not right:
            return left.strip()
        return f"{left.rstrip()}\n\n{right.lstrip()}"

    def _overlap_text(self, text: str, chunk_overlap: int) -> str:
        if chunk_overlap <= 0:
            return ""
        return text[-chunk_overlap:].lstrip()

    def _is_fenced_code_block(self, text: str) -> bool:
        stripped = text.strip()
        return stripped.startswith("```") and stripped.endswith("```") and stripped.count("```") >= 2

    def _starts_heading(self, text: str) -> bool:
        stripped = text.lstrip()
        return stripped.startswith("# ") or stripped.startswith("## ") or stripped.startswith("### ")

    def _positive_int(self, name: str, default: int) -> int:
        value = self._setting(name, default)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"splitter.{name} must be a positive integer")
        return value

    def _non_negative_int(self, name: str, default: int) -> int:
        value = self._setting(name, default)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"splitter.{name} must be a non-negative integer")
        return value

    def _setting(self, name: str, default: object) -> object:
        if isinstance(self.settings, Mapping):
            return self.settings.get(name, default)
        return getattr(self.settings, name, default)


SplitterFactory.register_provider("recursive", RecursiveSplitter)
