# -*- encoding: utf-8 -*-
"""
@Auth: xuelyuxin.xlx
@Time: 2023/03/07 11:00:59
@Desc: A tool to perform string-related method
"""

import re
from typing import Union

# 定义标准中文标点，tts中文文本中的标点一般统一映射为标准中文标点
PUNC_STANDARD = "。！？，"

PUNC_PRE_NORM = ""
REGEX_CN = "\u4e00-\u4E27\u4E29-\u4E3E\u4E42-\u9fa4"  # 跳过笔画\u4E28\u4E3F\u4E40\u4E41
REGEX_EN = "a-zA-Z"
REGEX_NUM = "0-9"

BLANK_CHAR = "\t\n\r\f"
PUNC_MAP_EN2CN = {
    "……": "。",
    "…" : "。",
    "!" : "！",
    "?" : "？",
    ";" : "；",
    ":" : "：",
    "," : "，",
    "(" : "（",
    ")" : "）"
}
PUNC_MAP_OTHER2CN = {
    "﹐" : "，",
    "﹔" : "；",
    "｡"  : "。"
}
PUNC_MAP_STANDARD = {
    "；" : "。",
    "：" : "，",
    "、" : "，"
}


class StringOperator:
    @classmethod
    def replace_punc_en2cn(cls, string: str) -> str:
        """replace english punctuations with chinese punctuations

        "." is not replaced, because "." could represent decimal or date, eg. 12.30, 
        and normally would not be mistaken with "。"
        """
        string = cls.replace(string, PUNC_MAP_EN2CN)
        string = re.sub(r"(\")(.*?)(\")", r"“\2”", string)
        return string

    @classmethod
    def replace(cls, string: str, map_dict: dict) -> str:
        """replace chars in `string` based on `map_dict`
        Args:
            string: original string
            map_dict: the mapping dict used to perform replacement
        """
        for pattern, target in map_dict.items():
            string = re.sub(f"{pattern}", target, string)
        return string

    @classmethod
    def delete(cls, string: str, delete: Union[str, re.Pattern]) -> str:
        """delete chars in `string` matched by `delete`
        """
        if isinstance(delete, str): 
            delete = re.compile(delete)
        string = re.sub(delete, "", string)
        return string

    @classmethod
    def delete_space(cls, string) -> str:
        """delete space in string
        1. 把除了英文之间的空格去掉，即去除非英文的前后的空格
        2. 英文之间的多个空格换成单个空格
        """
        string = re.sub("(?<=[^a-zA-Z])[ ]+", "", string)
        string = re.sub("[ ]+(?=[^a-zA-Z])", "", string)
        string = re.sub("(?<=[a-zA-Z])[ ]+(?=[a-zA-Z])", " ", string)
        return string

    @classmethod
    def replace_2u(cls, string: str) -> str:
        """TODO 转unicode字符"""
        pass

    @classmethod
    def delete_comma_in_number(cls, string: str) -> str:
        """delete comma of number in string

        eg. xxx12,345,678.123,xxx -> xxx12345678.123xxx
        """
        string = re.sub(r"(?<=\d),(?=\d{3})", "", string)
        return string

    @classmethod
    def replace_F2H(cls, string: str) -> str:
        """全角转半角

        Args:
            string: unicode字符串
        """
        # 单个unicode字符 全角转半角
        def F2H(char):
            inside_code = ord(char)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符则返回原来的字符
                return char
            return chr(inside_code)

        return "".join([F2H(char) for char in string])
    
    @classmethod
    def split(cls, pattern: str, text: int) -> list:
        """split text with matched $pattern

        different with re.split, the matched string is reserved
        """
        output = []
        start = 0
        for match in re.finditer(pattern, text):
            end = match.span()[1]
            output.append(text[start:end])
            start = end
        if start != len(text):
            output.append(text[start:])
        return output
    
    @classmethod
    def is_cn(cls, text):
        """判断text是否是纯中文
        """         
        if re.match(f"[{REGEX_CN}]+$", text):
            return True
        else:
            return False
    
    @classmethod
    def is_en(cls, text):
        """判断text是否是纯英文
        """         
        if re.match(f"([{REGEX_EN}]+$)|([{REGEX_EN}]+['][{REGEX_EN}]+$)", text):
            return True
        else:
            return False
    
    @classmethod
    def is_num(cls, text):
        """判断text是否是纯数字
        """
        if re.match(f"[\d]+$", text):
            return True
        else:
            return False


if __name__ == "__main__":

    from solutions.multimodal.adaspeech.recipes.asr.tools.get_logger import get_logger
    logger = get_logger("string_operator")
    samples = [
        ("replace", "123", {"2": "1", "3": "1"}),
        ("replace", "123", {"\d": "1"})
    ]
    for s in samples:
        if s[0] == "replace":
            logger.debug(f"raw: {s[1]}")
            logger.debug(f"replace: {StringOperator.replace(string=s[1], map_dict=s[2])}")