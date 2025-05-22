# -*- encoding: utf-8 -*-
"""
@Auth: xuelyuxin.xlx
@Time: 2023/05/12 14:55:44
@Desc: 正则模块
    `Normalizer.normalize_regular`输出内容只包含简体中文、英文、tts相关标点的文本
@Usge: null
"""

import re
from typing import List, Dict
from .utils.chronology import RE_DATE
from .utils.chronology import RE_DATE2
from .utils.chronology import RE_TIME, RE_TIME_2, RE_TIME_3
from .utils.chronology import RE_TIME_RANGE
from .utils.chronology import replace_date
from .utils.chronology import replace_date2
from .utils.chronology import replace_time, replace_time_nohour
from .utils.num import RE_DECIMAL_NUM
from .utils.num import RE_DEFAULT_NUM
from .utils.num import RE_FRAC
from .utils.num import RE_INTEGER
from .utils.num import RE_NUMBER
from .utils.num import RE_PERCENTAGE
from .utils.num import RE_POSITIVE_QUANTIFIERS
from .utils.num import RE_POSITIVE_QUANTIFIERS_2
from .utils.num import RE_RANGE
from .utils.num import RE_DIGITS
from .utils.num import RE_LICENSE_PLATE
from .utils.num import replace_default_num_with_altone, replace_default_num_without_altone
from .utils.num import replace_frac
from .utils.num import replace_negative_num
from .utils.num import replace_number
from .utils.num import replace_percentage
from .utils.num import replace_positive_quantifier
from .utils.num import replace_positive_quantifier_2
from .utils.num import replace_range
from .utils.num import replace_license_plate
from .utils.phonecode import RE_MOBILE_PHONE
from .utils.phonecode import RE_NATIONAL_UNIFORM_NUMBER
from .utils.phonecode import RE_TELEPHONE
from .utils.phonecode import replace_mobile
from .utils.phonecode import replace_phone
from .utils.quantifier import RE_TEMPERATURE
from .utils.quantifier import replace_temperature
from .utils.address import RE_ADDRESS_room, RE_ADDRESS
from .utils.address import replace_address_room, replace_address
from .utils.currency import RE_CURRENCY, RE_CURRENCY_2
from .utils.currency import replace_currency, replace_currency_2
from .utils.en_num import normalize_numbers as en_normalize_numbers
from .utils.string_operator import BLANK_CHAR, PUNC_MAP_OTHER2CN, PUNC_MAP_STANDARD, PUNC_STANDARD, REGEX_CN  # noqa
from .utils.string_operator import StringOperator as stringop


def add_blank(match_obj):
    return " ".join(list(match_obj.group(0)))


def convert_date(string):
    nums = re.findall("[\d]+", string)
    if len(nums) == 3:
        year, month, day = [RE_NUMBER.sub(replace_number, w) for w in nums]
        return f"{year}年{month}月{day}日"
    elif len(nums) == 2:
        month, day = [RE_NUMBER.sub(replace_number, w) for w in nums]
        return f"{month}月{day}日"
    else:
        return string


class Normalizer:
    """文本正则
    """

    @classmethod
    def substitute(cls, pattern: re.Pattern, replace_func, text: str, trace: list):
        for matchobj in pattern.finditer(text):
            origin_word = matchobj.group(0)
            new_word = replace_func(matchobj)
            trace.append({"origin_word": origin_word, "new_word": new_word})
            text = text.replace(origin_word, new_word)
        return text

    @classmethod
    def preprocess(cls, text: str) -> str:
        """正则前的预处理

        1. 繁体转简体
        2. 过滤不影响正则的标点符号等，包括
            a. 数字间的逗号
            b. 空格（除英文之间外）
            c. 空白字符\t\n\r\f
        3. 英文转小写
        """
        text = stringop.replace_F2H(text)  # 统一转半角
        text = stringop.delete_comma_in_number(text)  # 去掉数字之间的逗号
        # text = stringop.delete_space(text)
        text = re.sub(rf"[{BLANK_CHAR}]", "，", text)  # 去掉不影响正则的字符
        # text = text.lower()
        # 处理特殊符号
        text = re.sub(r"㎡", "m²", text)
        text = text.replace("㎡", "m²")
        text = text.replace("cm²", "平方厘米")
        text = text.replace("m²", "平方米")
        # text = text.replace("&lt;", "<")
        # text = text.replace("&gt;", ">")
        # text = text.replace("&amp;", "&")
        # text = text.replace("&quot;", "\"")
        # text = text.replace("&apos;", "'")
        text = re.sub(r">(\d)", r"大于\1", text)
        text = re.sub(r"<(\d)", r"小于\1", text)
        text = re.sub(r"=", "等于", text)
        text = re.sub(r"(?<=\d)ml(?![a-zA-Z])", "毫升", text)
        text = re.sub(r"(?<=\d)mmHg(?![a-zA-Z])", "毫米汞柱", text)
        text = re.sub(r"([0-9.]+元)(-)([0-9.]+元)", "\\1至\\3", text)
        return text

    @classmethod
    def postprocess(cls, text: str, custom: List[Dict] = None) -> str:
        """正则后处理，只包括删除和替换操作
            正则后的文本中的字符类型只包括：
            1. 中文 2. 英文 3. 标点`，。！？`
        """
        if custom is not None:
            for map_dict in custom:
                text = stringop.replace(text, map_dict)
            return text
        # 标点符号统一为中文标点`。！？，`
        text = stringop.replace_punc_en2cn(text)
        text = stringop.replace(text, PUNC_MAP_OTHER2CN)
        # 进一步把中文标点统一为标准中文标点`。！？，`
        text = stringop.replace(text, PUNC_MAP_STANDARD)
        # 处理连续句号"。"
        text = re.sub(r"。+", "。", text)
        # 处理正则后的 "/"
        text = re.sub("/", "每", text)
        # 处理正则后的[~～]+
        text = re.sub(r"～+", "～", text)
        text = re.sub(r"~+", "~", text)
        # text = re.sub(r"[~～](?=[0-9])", "至", text)
        text = re.sub(r"[~～]", "。", text)
        # 删除除了中文、英文、数字、标准中文标点、@break外的其他符号
        text = stringop.delete(text, f"[^{PUNC_STANDARD}{REGEX_CN}A-Za-z0-9@]")
        text = stringop.delete(text, "@(?!break)")  # 删除@符号，`@break`除外
        return text
    
    @classmethod
    def custom(cls, text: str, *, interpret_as: str) -> str:
        text = cls.preprocess(text)
        if not text: 
            return ""  # 对于预处理后为空的字符串直接返回
        text = cls.normalize_custom(text, interpret_as=interpret_as)
        text = cls.postprocess(text)
        return text
    
    @classmethod
    def regular(cls, text: str) -> str:
        text = cls.preprocess(text)
        if not text: 
            return ""  # 对于预处理后为空的字符串直接返回
        text = cls.normalize_regular(text)
        text = cls.postprocess(text)
        return text

    @classmethod
    def normalize_custom(cls, text: str, *, interpret_as: str) -> str:
        """指定正则

        仅针对`interpret_as`进行正则匹配

        Args:
            interpret_as: string in ['cardinal', 'currency', 'digits', 'telephone', 'address', 'date', 'time', 'id']
        """
        assert interpret_as in [
            "cardinal",
            "currency",
            "digits",
            "telephone",
            "address",
            "date",
            "time",
            "id",
            "measure",
            "punctuation"
        ], f"""
            interpret_as:{interpret_as} not supported
        """.strip()

        if interpret_as == "cardinal":
            text = text.replace(",", "")
            text = RE_NUMBER.sub(replace_number, text)  # 正负整数小数
            text = RE_FRAC.sub(replace_frac, text)
            text = RE_PERCENTAGE.sub(replace_percentage, text)
        elif interpret_as == "currency":
            text = RE_CURRENCY.sub(replace_currency, text)
            text = RE_CURRENCY_2.sub(replace_currency_2, text)
            text = text.replace(",", "")
            text = RE_NUMBER.sub(replace_number, text)  # 正负整数小数
            text = RE_FRAC.sub(replace_frac, text)
        elif interpret_as == "digits":
            text = RE_DIGITS.sub(replace_default_num_without_altone, text)
        elif interpret_as == "telephone":
            text = RE_MOBILE_PHONE.sub(replace_mobile, text)
            text = RE_TELEPHONE.sub(replace_phone, text)
            text = RE_NATIONAL_UNIFORM_NUMBER.sub(replace_phone, text)
            text = RE_DIGITS.sub(replace_default_num_with_altone, text)
        elif interpret_as == "address":
            text = text.replace("-", "杠")
            text = RE_ADDRESS_room.sub(replace_address_room, text)
            text = RE_ADDRESS.sub(replace_address, text)
        elif interpret_as == "date":
            text = RE_DATE.sub(replace_date, text)  # 年月日
            text = RE_DATE2.sub(replace_date2, text)  # YY/MM/DD 或者 YY-MM-DD
            text = convert_date(text)
            text = text.replace("-", "至")
        elif interpret_as == "time":
            text = RE_TIME_RANGE.sub(replace_time, text)  # 8:30-12:30
            text = RE_TIME.sub(replace_time, text)  # 12:30:58
        elif interpret_as == "id":
            text = RE_DIGITS.sub(replace_default_num_with_altone, text)
            text = text.replace("_", "下划线").replace("-", "杠").upper()
            text = re.sub("[a-zA-Z]+", add_blank, text)
        elif interpret_as == "measure":
            text = text.replace("㎡", "m²")
            text = text.replace("cm²", "平方厘米")
            text = text.replace("m²", "平方米")
            text = text.replace("cm", "厘米")
            text = text.replace("mm", "毫米")
            text = text.replace("m", "米")
            text = text.replace("kg", "千克")
            text = text.replace("g", "克")
        elif interpret_as == "punctuation":
            text = re.sub("…+", "省略号", text)
            text = re.sub("\"|“|”", "双引号", text)
            text = re.sub("'|‘|’", "单引号", text)
            text = re.sub("（|\\(", "左括号", text)
            text = re.sub("）|\\)", "右括号", text)
            text = re.sub("!|！", "叹号", text)
            for sym, txt in zip(
                ['……', '…', '!', '"', '#', '$', '%', '&', '‘', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_'],  # noqa
                ['省略号', '省略号', '叹号', '双引号', '井号', 'dollar', '百分号', 'and', '单引号', '左括号', '右括号', '星号', '加号', '逗号', '杠', '点', '斜杠', '冒号', '分号', '小于', '等号', '大于', '问号', 'at', '左方括号', '反斜线', '右方括号', '脱字符', '下划线']  # noqa
            ):
                text = text.replace(sym, txt)
        return text

    @classmethod
    def normalize_regular(cls, text: str, is_en: bool = False, return_details: bool = False) -> str:
        """通用正则

        包含了所有可能情况的正则匹配
        输出的文本中只包含中文、英文、tts所需的标点("，。！？")
        """
        trace = []
        if is_en is True:
            text = en_normalize_numbers(text)
            text = text.replace(".", "。")
            text = text.replace(",", "，")
        else:
            text = cls.substitute(RE_DATE, replace_date, text, trace)
            text = cls.substitute(RE_DATE2, replace_date2, text, trace)
            text = re.sub(r"(?<=[\d%])[-~](?=\d)", "至", text)  # 先判断日期（2023-01-02），并跟减号负号区分
            text = cls.substitute(RE_TIME_RANGE, replace_time, text, trace)
            text = cls.substitute(RE_TIME, replace_time, text, trace)
            text = cls.substitute(RE_CURRENCY, replace_currency, text, trace)
            text = cls.substitute(RE_CURRENCY_2, replace_currency_2, text, trace)
            text = cls.substitute(RE_TEMPERATURE, replace_temperature, text, trace)
            text = cls.substitute(RE_LICENSE_PLATE, replace_license_plate, text, trace)
            text = cls.substitute(RE_FRAC, replace_frac, text, trace)
            text = cls.substitute(RE_PERCENTAGE, replace_percentage, text, trace)
            text = cls.substitute(RE_MOBILE_PHONE, replace_mobile, text, trace)
            text = cls.substitute(RE_TELEPHONE, replace_phone, text, trace)
            text = cls.substitute(RE_NATIONAL_UNIFORM_NUMBER, replace_phone, text, trace)
            text = cls.substitute(RE_RANGE, replace_range, text, trace)
            text = cls.substitute(RE_INTEGER, replace_negative_num, text, trace)
            text = cls.substitute(RE_DECIMAL_NUM, replace_number, text, trace)
            text = cls.substitute(RE_POSITIVE_QUANTIFIERS_2, replace_positive_quantifier_2, text, trace)
            text = cls.substitute(RE_POSITIVE_QUANTIFIERS, replace_positive_quantifier, text, trace)
            text = cls.substitute(RE_DEFAULT_NUM, replace_default_num_with_altone, text, trace)
            text = cls.substitute(RE_NUMBER, replace_number, text, trace)

        return text, trace
