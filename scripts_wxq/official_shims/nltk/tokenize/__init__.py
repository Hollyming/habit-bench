import re


def word_tokenize(text):
    return re.findall(r"[A-Za-z0-9_]+|[^\w\s]", text or "")
