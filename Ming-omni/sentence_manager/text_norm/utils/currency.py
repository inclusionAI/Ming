import re

from .num import num2str

RE_CURRENCY = re.compile(r"([¥$])((\d+)(\.\d+)?)")
unit2str = {"¥": "人民币", "$": "美元"}


def replace_currency(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """
    unit = match.group(1)
    number = match.group(2)

    result = f"{num2str(number)}{unit2str[unit]}"
    return result


RE_CURRENCY_2 = re.compile(r"((\d+)(\.\d+)?)(RMB|rmb)")


def replace_currency_2(match) -> str:
    """
    Args:
        match (re.Match)
    Returns:
        str
    """

    number = match.group(1)

    result = f"{num2str(number)}人民币"
    return result


if __name__ == "__main__":

    sent = "1000.13rmb"
    print(RE_CURRENCY_2.sub(replace_currency_2, sent))
