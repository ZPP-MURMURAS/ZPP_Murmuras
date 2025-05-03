import pytest
import sys
import os
import numpy as np
import logging

from src.pipeline_benchmark.pipelines.bert_pipeline import _labeled_text_to_coupon, NER_ENTITY_GROUP, NER_SCORE, NER_TEXT, \
                                                           TAG_PRODUCT_NAME, TAG_DISCOUNT_TEXT, TAG_VALIDITY_TEXT, TAG_ACTIVATION_TEXT


class TestBertPipeline:

    @pytest.mark.parametrize("labeled_text, strategy, coupon", [
        (
            [
                {
                    NER_ENTITY_GROUP: TAG_PRODUCT_NAME,
                    NER_TEXT: 'A',
                    NER_SCORE: 0.5
                },
                {
                    NER_ENTITY_GROUP: TAG_PRODUCT_NAME,
                    NER_TEXT: 'B',
                    NER_SCORE: 0.7
                },
                {
                    NER_ENTITY_GROUP: TAG_DISCOUNT_TEXT,
                    NER_TEXT: 'C',
                    NER_SCORE: 0.7
                },
                {
                    NER_ENTITY_GROUP: TAG_DISCOUNT_TEXT,
                    NER_TEXT: 'D',
                    NER_SCORE: 0.5
                },
                {
                    NER_ENTITY_GROUP: TAG_VALIDITY_TEXT,
                    NER_TEXT: 'E',
                    NER_SCORE: 0.5
                }
            ], 
            'first', 
            {"product_name": "A", "discount_text": "C", "valid_until": "E", "activation_text": ""}),
        (
            [
                {
                    NER_ENTITY_GROUP: TAG_PRODUCT_NAME,
                    NER_TEXT: 'A',
                    NER_SCORE: 0.5
                },
                {
                    NER_ENTITY_GROUP: TAG_PRODUCT_NAME,
                    NER_TEXT: 'B',
                    NER_SCORE: 0.7
                },
                {
                    NER_ENTITY_GROUP: TAG_DISCOUNT_TEXT,
                    NER_TEXT: 'C',
                    NER_SCORE: 0.7
                },
                {
                    NER_ENTITY_GROUP: TAG_DISCOUNT_TEXT,
                    NER_TEXT: 'D',
                    NER_SCORE: 0.5
                },
                {
                    NER_ENTITY_GROUP: TAG_ACTIVATION_TEXT,
                    NER_TEXT: 'E',
                    NER_SCORE: 0.5
                }
            ], 
            'concat', 
            {"product_name": "A B", "discount_text": "C D", "valid_until": "", "activation_text": "E"}),
        (
            [
                {
                    NER_ENTITY_GROUP: TAG_PRODUCT_NAME,
                    NER_TEXT: 'A',
                    NER_SCORE: 0.5
                },
                {
                    NER_ENTITY_GROUP: TAG_PRODUCT_NAME,
                    NER_TEXT: 'B',
                    NER_SCORE: 0.7
                },
                {
                    NER_ENTITY_GROUP: TAG_DISCOUNT_TEXT,
                    NER_TEXT: 'C',
                    NER_SCORE: 0.7
                },
                {
                    NER_ENTITY_GROUP: TAG_DISCOUNT_TEXT,
                    NER_TEXT: 'D',
                    NER_SCORE: 0.5
                },
                {
                    NER_ENTITY_GROUP: TAG_VALIDITY_TEXT,
                    NER_TEXT: 'E',
                    NER_SCORE: 0.5
                }
            ], 
            'top_score', 
            {"product_name": "B", "discount_text": "C", "valid_until": "E", "activation_text": ""}),
        ])
    def test_labeled_text_to_coupon(self, labeled_text, strategy, coupon):
        assert _labeled_text_to_coupon(labeled_text, strategy) == coupon

