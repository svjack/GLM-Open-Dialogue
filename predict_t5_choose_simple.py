device = "cpu"
#device = "cuda:0"
assert device.startswith("cpu") or device.startswith("cuda")

negative_prompts = [
        "是的",
        "真的",
        "谢",
        "泻药",
        ",",
        "谢邀",
        "没有",
        "有",
        "啊",
    ] + ["不知道", "不想" , "不会", "怎么知道", "你说呢", "是吗？", "不", "否定的",] + \
    ["一", "1", "二", "2",]

from fix_dialogue_context import *

from simplet5 import SimpleT5
import os
import sys
import numpy as np
import pandas as pd
import re
import pathlib
import shutil
from tqdm import tqdm
from copy import deepcopy

import pandas as pd
import numpy as np
import jieba.posseg as posseg
#import jionlp as jio


model = SimpleT5()
if device.startswith("cuda"):
    model.load_model(
    model_dir = "svjack/T5-daliy-dialogue",
    use_gpu = True)
else:
    model.load_model(
    model_dir = "svjack/T5-daliy-dialogue",
    use_gpu = False)

choose_model = SimpleT5()
if device.startswith("cuda"):
    choose_model.load_model(
    model_dir = "svjack/T5-dialogue-choose",
    use_gpu = True)
else:
    choose_model.load_model(
    model_dir = "svjack/T5-dialogue-choose",
    use_gpu = False)

collect_model = SimpleT5()
if device.startswith("cuda"):
    collect_model.load_model(
    model_dir = "svjack/T5-dialogue-collect-v6",
    use_gpu = True)
else:
    collect_model.load_model(
    model_dir = "svjack/T5-dialogue-collect-v6",
    use_gpu = False)

import torch
import json
from flagai.model.glm_model import GLMModel
from flagai.data.tokenizer import Tokenizer
from flagai.model.predictor.predictor import Predictor

model_name = 'GLM-large-ch'
#device = "cuda:0"
#device = "cpu"
g_model = GLMModel.from_pretrain(model_name=model_name,
                                   download_path="./state_dict/")
tokenizer = Tokenizer.from_pretrained(model_name)
#tokenizer = Tokenizer.from_pretrained("GLM-large-ch", only_download_config=False)
if device.startswith("cuda"):
    g_model.cuda(torch.cuda.current_device())
predictor = Predictor(g_model, tokenizer)

import jieba
def repeat_to_one_f(x):
    req = None
    for token in jieba.lcut(x):
        #print("req :", req)

        if len(set(token)) == 1:
            token = token[0]
        if req is None:
            req = token
        else:

            if (token in req and token not in [',', '，', '、', ' ']) or (req and token in [',', '，', '、', ' '] and req[-1] in [',', '，', '、', ' ']):
                continue
            else:
                while req.endswith(token[0]):
                    token = token[1:]
                req = req + token
    if req is None:
        return ""
    return req.strip()

def repeat_to_one_fb(x):
    return sorted(map(repeat_to_one_f, [x, "".join(jieba.lcut(x)[::-1])]),
                 key = len
                 )[0]

#repeat_to_one = repeat_to_one_fb
repeat_to_one = repeat_to_one_f

#@tranquilize(method='post')
def gen_answer_by_glm(question, mask_string = "gMASK", config_string = ""):
    text = '问题 {} 回答：[{}]'.format(question, mask_string)
    if not config_string:
        kwargs = {}
    else:
        kwargs = json.loads(config_string)
    assert type(kwargs) == type({})
    output=predictor.predict_generate_randomsample(text, **kwargs)
    return output
#### file end

#import synonyms
from rapidfuzz import fuzz
def rec_construct(query, max_times = 5, dec_times = 1, **kwargs):
    assert dec_times >= 1
    req = [query]
    for i in range(max_times):
        #print(req)
        in_ = "[SEP]".join(req)
        assert type(kwargs) == type({})
        kwargs["num_return_sequences"] = dec_times
        kwargs["num_beams"] = max(dec_times, kwargs.get("num_beams", 2))
        out_l = model.predict(in_, **kwargs)
        out = sorted(out_l, key = lambda x: fuzz.ratio(x, req[-1]), reverse = True)[0]
        if out in req:
            break
        req.append(out)
    return req

def label_source_to_l(l):
    assert type(l) == type([])
    def rule_justify(x):
        need = []
        if any(map(lambda y: y in x, "?？")):
            need.append("ask")
        if any(map(lambda y: y in x, ",.，。、")):
            need.append("answer")
        if not need:
            need.append("answer")
        return need
    return list(map(lambda x: (rule_justify(x), x), l))

def nearest_qa_pair_seek(l):
    assert type(l) == type([])
    req = []
    for i in range(len(l)):
        assert len(l[i]) == 2
        labels = l[i][0]
        text = l[i][1]
        is_answer = "answer" in labels
        if not is_answer:
            continue
        pre_l = l[:i+1]
        ask_text = ""
        for ele in pre_l[::-1]:
            assert type(ele) == type((1,))
            if "ask" in ele[0]:
                ask_text = ele[1]
                assert type(ask_text) == type("")
                break
        req.append((ask_text, text))
    return req

def process_context(context):
    import re
    req = re.sub(r"[\[\<|][A-Za-z]+[\]|\>]" ,"" ,context).replace("<>", "")
    if "回答:" in req:
        req = "".join(req.split("回答:")[1:])
    req = req.strip()
    return req

def glm_generator(header_question,
                            question,
                            answer, gen_times = 10, out_max_length = None):
    if out_max_length is None:
        out_max_length = len(answer)
    c_list = []
    for i in tqdm(range(gen_times)):
        ask_question = "关于{}的问题,{}".format(header_question.replace("?", "").replace("？", ""),
            question)
        if i == 0:
            print(ask_question)
        #context = call_glm(ask_question, "MASK", out_max_length = out_max_length)
        context = gen_answer_by_glm(ask_question, "MASK", json.dumps(
            {"out_max_length": out_max_length}
        ))
        context = process_context(context)
        c_list.append((context, 1.0))
    return c_list

def rec_construct_longer(question, max_times):
    l0 = rec_construct(question, max_times = max_times,
    dec_times=2)
    if len(l0) >= max_times:
        return l0
    l1 = rec_construct(question, max_times = max_times,
    dec_times=5)
    l = sorted([l0, l1], key = len, reverse = True)[0]
    return l

def batch_as_list(a, batch_size = int(100000)):
    req = []
    for ele in a:
        if not req:
            req.append([])
        if len(req[-1]) < batch_size:
            req[-1].append(ele)
        else:
            req.append([])
            req[-1].append(ele)
    return req

def predict_format(context ,all_candidates):
    all_candidates_text = "\n".join(map(lambda t2: "选项{}:{}".format(t2[0],
                                                                   t2[1]
                                                                   ), enumerate(all_candidates)))
    prompt_format = '''
    根据如下上下文，选择最优的后续句子
    上下文：{}\n{}
    答案：
    '''
    prompt = prompt_format.format(context, all_candidates_text)
    return prompt

def predict_format_hqa(header_question, question, all_candidates):
    ask_question = "关于{}的问题,{}".format(header_question.replace("?", "").replace("？", ""),
        question)
    prompt = predict_format(ask_question ,all_candidates)
    return prompt


def shorten_exists(l, sim_threshold = 70, slice_size = 2):
    req = []
    for ele in l:
        if not req:
            req.append(ele)
        else:
            if max(map(lambda x: fuzz.ratio(x[:slice_size], ele[:slice_size]), req)) < sim_threshold:
                req.append(ele)
    return req

def shorten_list(l, sim_threshold = 30, outer_append = []):
    req = []
    outer_append = outer_append + negative_prompts
    l = l + outer_append
    l = sorted(l)
    #print(l)
    for ele in l:
        if ele in outer_append:
            req.append((ele, 100))
        else:
            if not req:
                req.append((ele, 100))
            else:
                val = fuzz.ratio(ele, req[-1][0])
                #print(req, val, sim_threshold)
                if val < sim_threshold:
                    req.append((ele, val))
    req = list(filter(lambda t2:
    max(map(lambda x: fuzz.ratio(x, t2[0]), outer_append)) < sim_threshold, req))
    return req

def choose_in_rec_manner(hq, q, all_ans, max_size = 4, print_it = False):
    assert type(hq) == type("")
    assert type(q) == type("")
    assert type(all_ans) == type([])
    from copy import deepcopy
    all_ans = deepcopy(all_ans)
    while len(all_ans) > max_size:
        l = batch_as_list(all_ans, max_size)
        req = []
        for ele in l:
            assert type(ele) == type([])
            if len(ele) == 1:
                req += ele
            else:
                assert len(ele) > 1
                prompt = predict_format_hqa(hq, q, ele)
                assert type(prompt) == type("")
                out = choose_model.predict(prompt)
                if print_it:
                    print("-*" * 10)
                    print(prompt)
                    print(out)
                assert type(out) == type([])
                assert len(out) == 1
                req += out
        assert len(req) > 0
        all_ans = deepcopy(req)
    assert len(all_ans) > 0
    if len(all_ans) == 1:
        return all_ans[0]
    else:
        assert len(all_ans) > 1
        #print("-*" * 10)
        #print(all_ans)
        prompt = predict_format_hqa(hq, q, all_ans)
        #print(prompt)
        assert type(prompt) == type("")
        out = choose_model.predict(prompt)
        #print(out)
        if print_it:
            print("-*" * 10)
            print(prompt)
            print(out)
        assert type(out) == type([])
        assert len(out) == 1
        return out[0]

def choose_in_rec_manner_bagging(hq, q, all_ans, max_size = 4,
    bagging_times = 5, return_cnt = False
):
    assert bagging_times > 0
    all_ans = list(set(all_ans))
    all_ans = list(map(lambda t2: t2[0] ,shorten_list(all_ans, outer_append = [hq])))
    req = []
    for i in range(bagging_times):
        all_ans = pd.Series(all_ans).sample(frac = 1.0).values.tolist()
        out = choose_in_rec_manner(hq, q, all_ans, max_size)
        req.append(out)
    if return_cnt:
        return pd.Series(req).value_counts().to_dict()
    return pd.Series(req).value_counts().index.tolist()[0]

def shorten_e(a, e):
    e_req = []
    hq = a[0]
    for e_ele in e:
        q, ans, ans_l = e_ele
        ans_l = pd.Series(shorten_list(
        list(map(lambda t2: t2[0] ,ans_l)),
        outer_append = [hq]
        )).map(
        lambda t2: t2[0]
        ).values.tolist()
        out = choose_in_rec_manner(hq, q,
                    ans_l
                                )
        assert type(out) == type("")
        e_req.append(
            (q, ans, [(out, 1.0)])
        )
    return e_req

def generate_pair(question, max_times = 5,
    gen_times = 10, use_shorten_e = True, exist_f = True):
    from copy import deepcopy
    l = rec_construct_longer(question, max_times = max_times)
    ori_l = deepcopy(l)
    l_labled = label_source_to_l(l)
    pair_list = nearest_qa_pair_seek(l_labled)
    last_labled = l_labled[-1]
    if "ask" in last_labled[0]:
        pair_list.append(
            (last_labled[1], last_labled[1])
        )
    pair_list_glm_list = []
    for qst, ans in tqdm(pair_list):
        ll = glm_generator(l[0],
                                    qst,
                                    ans, gen_times = gen_times)
        pair_list_glm_list.append(
        (qst, ans, ll)
        )
    ori_pair_list_glm_list = deepcopy(pair_list_glm_list)
    if use_shorten_e:
        pair_list_glm_list = shorten_e(ori_l, pair_list_glm_list)
    pair_list_glm_dict = dict(map(lambda t3: (t3[1], t3[2]), pair_list_glm_list))
    #l = list(map(lambda x: pair_list_glm_dict.get(x, x), l))
    l_labled_rev_dict = dict(map(lambda t2: (t2[1], t2[0]), l_labled))
    req_l = []
    for ele in l:
        if "ask" in l_labled_rev_dict.get(ele, []):
            if ele in pair_list_glm_dict:
                req_l.append(ele)
                req_l.append(pair_list_glm_dict[ele])
            else:
                req_l.append(ele)
        else:
            req_l.append(pair_list_glm_dict[ele])

    req_l_mapped = list(map(lambda x: x if type(x) == type("") else x[0][0], req_l))
    req_l_mapped_ = []
    for ele in req_l_mapped:
        if ele not in req_l_mapped_:
            req_l_mapped_.append(ele)
    req_l_mapped = req_l_mapped_
    req_l_mapped = list(map(repeat_to_one, req_l_mapped))
    if exist_f:
        req_l_mapped = shorten_exists(req_l_mapped)
    return req_l_mapped ,ori_l, req_l, l_labled, pair_list_glm_list, ori_pair_list_glm_list

def add_book_by_ner(source_list, list_for_add):
    assert type(source_list) == type([])
    assert type(list_for_add) == type([])
    all_books = list(set(reduce(lambda a, b: a + b ,
                       map(lambda x: re.findall("《(.*)》", x), source_list))))
    all_books = list(filter(lambda x: x, all_books))
    if not all_books:
        return list_for_add
    def single_rp(x):
        x = x.replace("《", "").replace("》", "")
        for xx in sorted(all_books, key = len, reverse = True):
            if xx in x:
                x = x.replace(xx, "《{}》".format(xx))
        return x
    list_for_add = list(map(single_rp, list_for_add))
    return list_for_add

rp_list = "介词 介词_方位介词 代词 副词 叹词 疑问词 连词".split()
def process_one_sent(input_, drop_prob = 1.0):
    assert type(input_) == type("")
    input_ = " ".join(map(lambda y: y.word.strip() ,filter(lambda x: x.flag != "x" ,
    posseg.lcut(input_))))
    #input_l = " ".join(filter(lambda x: lexicon_sentiment(x) == 0.5, input_l))
    x = ner(input_)
    assert type(x) == type([])
    #print(x)
    input_ = " ".join(map(lambda t2: "[{}]".format(t2[1]) if
    (t2[1] in rp_list and np.random.rand() <= drop_prob) else t2[0], x))
    #print(input_)
    input_ = input_.replace("[", "").replace("]", "")
    return input_

def predict_split(sp_list, cut_tokens = True, drop_prob = 1.0, rp_dict = {}):
    assert type(sp_list) == type([])
    if cut_tokens:
        src_text = '''
            根据下面的上下文进行分段：
            上下文：{}
            答案：
            '''.format(" ".join(
            map(lambda x:process_one_sent(x, drop_prob = drop_prob) ,sp_list)
            ))
    else:
        src_text = '''
            根据下面的上下文进行分段：
            上下文：{}
            答案：
            '''.format("".join(sp_list))
    for k, v in rp_dict.items():
        src_text = src_text.replace(k, v)
    print(src_text)
    pred = collect_model.predict(src_text)[0]
    pred = list(filter(lambda y: y ,map(lambda x: x.strip() ,pred.split("分段:"))))
    return pred

def dialogue_context_collect_func(input_):
    assert type(input_) == type([])
    ori_input = deepcopy(input_)
    if not input_:
        return []
    input_ = deepcopy(input_)
    if len(input_) >= 2:
        input_1 = dialogue_context_fix_func(input_[1:])
        input_1 = predict_split(input_1)
        assert type(input_1) == type([])
        assert type(input_[0]) == type("")
        input_ = [input_[0]] + input_1
    #sp_list = input_
    #input_ = predict_split(sp_list)
    if len(input_) >= 2:
        input_1 = dialogue_context_fix_func(input_[1:])
        assert type(input_1) == type([])
        assert type(input_[0]) == type("")
        input_ = [input_[0]] + input_1
    input_ = add_book_by_ner(ori_input, input_)
    return input_

def generate_seq(question, max_times = 10, single_step_times = 1,
exist_f = True, break_length = 256, fix_it = True):
    from copy import deepcopy
    req = []
    for i in tqdm(range(max_times)):
        a, b, c, d, e, f = generate_pair(question, max_times=single_step_times)
        assert type(a) == type([])
        question = "".join(a)
        if not req:
            req.append("[SEP]".join(a))
        else:
            req.append("[SEP]".join([req[-1]] + a[1:]))
        if len(req[-1]) >= break_length:
            break
    req = req[-1].split("[SEP]")
    req = list(map(repeat_to_one, req))
    if exist_f:
        req = shorten_exists(req)
    if fix_it:
        req = dialogue_context_collect_func(req)
    if exist_f:
        req = shorten_exists(req)
    return req

'''
a_seq = generate_seq("程序员要掌握哪些技能?")
a, b, c, d, e, f = generate_pair("程序员要掌握哪些技能?")
en_a, zh_a = back_trans(a_seq)
a_seq = generate_seq("你喜欢看奥特曼吗?")
a, b, c, d, e, f = generate_pair("你喜欢看奥特曼吗?")
a_seq = generate_seq("朱元璋是怎么当上皇帝的?")
a, b, c, d, e, f = generate_pair("朱元璋是怎么当上皇帝的?")
a_seq = generate_seq("如何做西红柿炒鸡蛋?")
a, b, c, d, e, f = generate_pair("如何做西红柿炒鸡蛋?")
a_seq = generate_seq("凯撒是如何死亡的?")
a, b, c, d, e, f = generate_pair("凯撒是如何死亡的?")
a_seq = generate_seq("当厨师需要看哪些书?")
a, b, c, d, e, f = generate_pair("当厨师需要看哪些书?")
a_seq = generate_seq("GTA有哪些玩法?")
a, b, c, d, e, f = generate_pair("GTA有哪些玩法?")
a_seq = generate_seq("GTA有哪些有趣的玩法?")
a, b, c, d, e, f = generate_pair("GTA有哪些有趣的玩法?")
a_seq = generate_seq("你玩过哪些策略类游戏?")
a, b, c, d, e, f = generate_pair("你玩过哪些策略类游戏?")
a_seq = generate_seq("推特的CEO是谁?")
a, b, c, d, e, f = generate_pair("推特的CEO是谁?")
text = "D e l s a S p o s a 糖 果 色 系 列 婚 纱 ， 粉 蓝 红 紫 ， 俏 皮 又 清 新 ， 你 喜 欢 吗 ？".replace(" ", "")
text
a_seq = generate_seq(text)
a, b, c, d, e, f = generate_pair(text)
#### with context
a_seq = generate_seq("一个新人厨师正在准备厨师证考试,当厨师需要看哪些书?")
a, b, c, d, e, f = generate_pair("一个新人厨师正在准备厨师证考试,当厨师需要看哪些书?")
a_seq = generate_seq("清朝的《明史》对于朱元璋称帝的问题语焉不详,朱元璋是怎么当上皇帝的?")
a, b, c, d, e, f = generate_pair("清朝的《明史》对于朱元璋称帝的问题语焉不详,朱元璋是怎么当上皇帝的?")
a_seq = generate_seq("STEAM选出了今年的年度游戏,其中有GTA5,GTA有哪些的玩法?")
a, b, c, d, e, f = generate_pair("STEAM选出了今年的年度游戏,其中有GTA5,GTA有哪些的玩法?")
a_seq = generate_seq("最近有一个有名的歌曲叫做《奥特曼和小怪兽》,你喜欢看奥特曼吗?")
a, b, c, d, e, f = generate_pair("最近有一个有名的歌曲叫做《奥特曼和小怪兽》,你喜欢看奥特曼吗?")
a_seq = generate_seq("我玩过光荣公司出品的《三国志10》,你玩过哪些策略类游戏?")
a, b, c, d, e, f = generate_pair("我玩过光荣公司出品的《三国志10》,你玩过哪些策略类游戏?")
a_seq = generate_seq("听说凯撒死之后屋大维为他报了仇,凯撒是怎么死的?")
a, b, c, d, e, f = generate_pair("听说凯撒死之后屋大维为他报了仇,凯撒是怎么死的?")
a_seq = generate_seq("我昨天在饭店尝到了很好吃的一道菜,如何做西红柿炒鸡蛋?")
a, b, c, d, e, f = generate_pair("我昨天在饭店尝到了很好吃的一道菜,如何做西红柿炒鸡蛋?")
a_seq = generate_seq("听说最近很多人感染了新型冠状病毒,怎么能够较好地保护自己呢?")
a, b, c, d, e, f = generate_pair("听说最近很多人感染了新型冠状病毒,怎么能够较好地保护自己呢?")
a_seq = generate_seq("听说中国大陆的麦当劳都改名为金拱门了,你觉得麦当劳和肯德基哪一个好吃呢?")
a, b, c, d, e, f = generate_pair("听说中国大陆的麦当劳都改名为金拱门了,你觉得麦当劳和肯德基哪一个好吃呢?")
a_seq = generate_seq("有一本故事书描写，苏丹阿尔斯兰击败了阿莱克休斯二世，统一了阿拉伯世界，" + \
"你觉得阿尔斯兰伟大吗？")
a, b, c, d, e, f = generate_pair("有一本故事书描写，苏丹阿尔斯兰击败了阿莱克休斯二世，统一了阿拉伯世界，" + \
"你觉得阿尔斯兰伟大吗？")
'''
