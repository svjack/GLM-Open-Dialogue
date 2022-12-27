import requests
from easynmt import EasyNMT
import os

trans_model = EasyNMT("m2m_100_1.2B")

import gradio as gr

def translate_func(src_question, src_lang, tgt_lang):
    assert type(src_question) == type("")
    assert type(src_lang) == type("")
    assert type(tgt_lang) == type("")
    if "[SEP]" in src_question:
        src_question = list(filter(lambda xx: xx ,map(lambda x: x.strip() ,
        src_question.split("[SEP]"))))
    else:
        src_question = [src_question]
    tgt_question = trans_model.translate(
        src_question,
        source_lang=src_lang, target_lang = tgt_lang
    )
    assert type(tgt_question) == type([])
    tgt_question = "[SEP]".join(tgt_question)
    return {
        "Target Question": tgt_question
    }
