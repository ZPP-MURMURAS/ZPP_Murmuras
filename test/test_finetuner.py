import pytest
from src.bert_finetuning.finetuner import __assert_init as ft__assert_init \
    , create_custom_tags, __align_labels_with_tokens as ft__align_labels_with_tokens


class TestFinetuner:
    def test_assert_init(self):
        with pytest.raises(AssertionError):
            ft__assert_init()

    def test_create_custom_tags_append_zer0(self):
        tokens = ['O', 'N/A']
        assert create_custom_tags(tokens) == ['O', 'O']

    def test_create_custom_tags_append_bi(self):
        tokens = ['B', 'I']
        assert create_custom_tags(tokens) == ['B-B', 'I-B', 'B-I', 'I-I']

    def test_align_labels_with_tokens_no_changes(self):
        labels = [1, 2, 3, 4]
        word_ids = [0, 1, 2, 3]
        assert ft__align_labels_with_tokens(labels, word_ids, True) == [1, 2, 3, 4]

    def test_align_labels_with_tokens_bi_split(self):
        labels = [1, 2, 3, 4, 4, 4]
        word_ids = [0, 1, 2, 3, 3, 3]
        assert ft__align_labels_with_tokens(labels, word_ids, True) == [1, 2, 3, 4, 4, 4]

    def test_align_labels_with_tokens_no_bi_split(self):
        labels = [1, 2, 3, 4, 4, 4]
        word_ids = [0, 1, 2, 3, 3, 3]
        assert ft__align_labels_with_tokens(labels, word_ids, False) == [1, 3, 5, 7, 8, 8]