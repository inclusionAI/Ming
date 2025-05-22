# -*- encoding: utf-8 -*-
# Time: 2024/12/02 19:58:58
# Desc:

import os
import re
import yaml
from typing import Optional, List
from enum import Enum

from .text_norm.normalizer import Normalizer


CN_REGEX = "\u4e00-\u4E27\u4E29-\u4E3E\u4E42-\u9fa4"  # 跳过笔画\u4E28\u4E3F\u4E40\u4E41


class SentencePieceType(str, Enum):
    END_OF_SENTENCE = "END_OF_SENTENCE"


def split_with_separator(regx_sep, text):
    """difference with `re.split`: include separator
    """
    split_list = []
    start = 0
    for match in re.finditer(regx_sep, text):
        end = match.span()[1]
        assert end > start
        split_list.append(text[start: end])
        start = end
    split_list.append(text[start:])  # could be empty string
    return split_list


def split(text, split_pattern, split_cn_length=None):
    """
    Args:
        text
    Return:
        (split_list: List[str], remain: str)
    """
    split_list = split_with_separator(split_pattern, text)
    remain = split_list.pop(-1)  # 最后一项为不完整子句，可能为空字符串

    # 针对末尾的不完整子句，若满足中文字数条件，也添加进split_list
    if split_cn_length is not None:
        text_split = re.search(f"^[。！？，{CN_REGEX}]" + "{" + f"{split_cn_length}," + "}", remain)
        if text_split:
            text_split = text_split.group()
            split_list.append(text_split)
            remain = remain[len(text_split):]
    return split_list, remain


class SentenceNormalizer(Normalizer):
    def __init__(self, config={}):
        self.config = config

    def normalize(self, text, context: str = ""):
        text = self.preprocess(text)
        text, norm_details = self.normalize_regular(text, is_en=False)
        text = self.postprocess(text, custom=self.config["postprocess"])
        text = text[len(context):]
        return text


class SentenceManager:
    def __init__(self, tokenizer, normalizer, config):
        """
        Args:
            tokenizer: tokenizer为必填参数，因为对英文等特殊符号，tokenize和拼接操作不是可交换的
        """

        self.split_pattern = "|".join(config["split_token"])
        self.split_cn_length = config["split_cn_length"]

        self.tokenizer = tokenizer
        self.normalizer = normalizer

        self.context: Optional[str] = ""
        self.cache: List[int] = []
        self.output_queue: List[List[int]] = []

    def put(self, token_id):
        text = self.tokenizer.decode([*self.cache, token_id])
        split_list, remain = split(text, self.split_pattern, split_cn_length=self.split_cn_length)
        assert split_list or remain

        if split_list:
            normalized_split_list = [
                self.normalizer.normalize(x) if i < len(split_list) else self.normalizer.normalize(x, self.context) 
                for i, x in enumerate(split_list)
            ]
            if len(normalized_split_list[-1]) == len(split_list[-1]):
                self.context = split_list[-1]
            token_ids_list = [self.tokenizer.encode(x) for x in normalized_split_list if x]
            self.output_queue.extend(token_ids_list)
            if re.search(f"({self.split_pattern})$", split_list[-1]):
                self.output_queue.append(SentencePieceType.END_OF_SENTENCE)

        if remain:
            token_ids_remain = self.tokenizer.encode(remain)
            self.cache = token_ids_remain
        else:
            self.cache = []

    def get(self):
        if self.output_queue:
            return self.output_queue.pop(0)
        else:
            return None
