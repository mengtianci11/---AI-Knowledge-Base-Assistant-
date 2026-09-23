"""分块策略单元测试。"""
import pytest

from app.chunking import split_text, split_text_fixed, split_text_recursive


class TestSplitFixed:
    def test_basic_windows(self):
        text = "a" * 1000
        chunks = split_text_fixed(text, chunk_size=300, overlap=50)
        assert len(chunks) > 1
        # 相邻块存在重叠区域（滑窗）
        assert text[250:300] in chunks[0] and text[250:300] in chunks[1]

    def test_no_blank_chunks(self):
        chunks = split_text_fixed("   \n\n  ", chunk_size=10, overlap=2)
        assert chunks == []

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            split_text_fixed("xxx", chunk_size=-1)
        with pytest.raises(ValueError):
            split_text_fixed("xxx", chunk_size=10, overlap=10)


class TestSplitRecursive:
    def test_paragraph_merged(self):
        text = "\n\n".join(["第一段内容。" * 30, "第二段内容。" * 30])
        chunks = split_text_recursive(text, chunk_size=300, overlap=50)
        assert chunks
        assert all(len(c) <= 300 + 1 for c in chunks)

    def test_long_paragraph_falls_back_to_window(self):
        text = "长" * 900
        chunks = split_text_recursive(text, chunk_size=300, overlap=50)
        assert len(chunks) >= 2
        assert sum(len(c) for c in chunks) >= 900

    def test_short_text_single_chunk(self):
        assert split_text_recursive("短文本。", chunk_size=300, overlap=50) == ["短文本。"]

    def test_empty(self):
        assert split_text_recursive("  \n ", chunk_size=300, overlap=50) == []


class TestSplitDispatch:
    def test_strategies(self):
        text = "今天天气不错，适合学习。\n\n" * 50
        fixed = split_text(text, "fixed", 100, 20)
        recursive = split_text(text, "recursive", 100, 20)
        assert fixed and recursive