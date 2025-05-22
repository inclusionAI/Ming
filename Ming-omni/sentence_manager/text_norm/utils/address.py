import re

from .num import num2str, verbalize_digit

RE_ADDRESS_room = re.compile(r"(\d+)(室)?$")
RE_ADDRESS = re.compile(r"(\d+)")


def replace_address_room(match):

    num = match.group(1)
    result = verbalize_digit(num, alt_one=True)
    if match.group(2):
        result += match.group(2)
    return result


def replace_address(match):

    num = match.group(1)

    return num2str(num)


if __name__ == "__main__":

    sent = "五常街道庭院5幢4单元201"
    sent = RE_ADDRESS_room.sub(replace_address_room, sent)
    sent = RE_ADDRESS.sub(replace_address, sent)
    print(sent)
