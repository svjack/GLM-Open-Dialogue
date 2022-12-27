import os
#os.system("pip install flagai==1.5.0")

import gradio as gr
from translate import *
from generate import *

example_sample = [
    ["Which keyboard do you prefer?", "en", 2],
    ["東京と京都のどちらが住みやすいですか。", "ja", 2],
    ["程序员要掌握哪些技能?", "zh", 2],
]

def demo_func(question, lang, max_times):
    assert type(question) == type("")
    max_times = max(int(max_times), 1)
    if not lang.startswith("zh"):
        zh_question = translate_func(question, lang, "zh")
        zh_question = zh_question["Target Question"]
    else:
        zh_question = question
    gen_output = generate_func(zh_question, max_times)
    gen_output = gen_output["Dialogue Context"]
    assert type(gen_output) == type([])
    zh_gen_output = "[SEP]".join(gen_output)
    if not lang.startswith("zh"):
        lang_question = translate_func(zh_gen_output, "zh", lang)
        lang_question = lang_question["Target Question"]
    else:
        lang_question = zh_gen_output
    l = list(filter(lambda y: y ,map(lambda x: x.strip() ,lang_question.split("[SEP]"))))
    assert type(l) == type([])
    return {
        "Dialogue Context": l
    }


demo = gr.Interface(
        fn=demo_func,
        inputs=[gr.Text(label = "Question"),
                gr.Text(label = "Language", value = "en"),
                gr.Number(label = "Forward Times", value = 2)
        ],
        outputs="json",
        title=f"Open Dialogue Generator by GLM ⚡️ demonstration",
        examples=example_sample if example_sample else None,
        cache_examples = False
    )

demo.launch(server_name=None, server_port=None)
