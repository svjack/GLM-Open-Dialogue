import requests
from predict_t5_choose_simple import *

def generate_func(zh_sentence, max_times):
    assert type(zh_sentence) == type("")
    assert type(max_times) == type(0)
    zh_question = zh_sentence
    l = generate_seq(zh_question, max_times = max_times)
    assert type(l) == type([])
    return {
        "Dialogue Context": l
    }
