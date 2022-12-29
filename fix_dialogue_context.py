import pandas as pd
import numpy as np
import json
import jieba.posseg as posseg
from functools import reduce
from tqdm import tqdm

from rapidfuzz import fuzz
from paddlenlp import Taskflow
ner = Taskflow("ner")

def any_flag_drop(input_str, unique_flag_list = ["x"]):
    jieba_cut_l = list(map(lambda y: (y.word, y.flag), posseg.lcut(input_str)))
    req = []
    pre_flag = None
    for w, f in jieba_cut_l:
        if pre_flag not in unique_flag_list:
            req.append(w)
            pre_flag = f
        else:
            if f == pre_flag:
                continue
            else:
                req.append(w)
                pre_flag = f
    return "".join(req)

def extract_last_span(input_str, sp_f_list = ["x"]):
    jieba_cut_l = list(map(lambda y: (y.word, y.flag), posseg.lcut(input_str)))
    jieba_cut_l_res = jieba_cut_l[::-1]
    req = []
    for i, (w, f) in enumerate(jieba_cut_l_res):
        if i == 0:
            req.append((w, f))
        else:
            if f in sp_f_list:
                break
            else:
                req.append((w, f))
    return req[::-1]

def maintain_last_span_justifyer(input_str, last_span_t2_list,):
    w_m_collection = ["?", ".", "。", "？"]
    f_m_collection = ["n", "r", "I", "i", "eng"]
    f_d_collection = ["v"]
    if last_span_t2_list[-1][1] in f_d_collection:
        return False
    length_ratio_threshold = 0.8
    if last_span_t2_list[-1][0] in w_m_collection:
        return True
    if any(map(lambda t2: any(map(lambda x :t2[1].startswith(x),
                          f_m_collection)), last_span_t2_list)):
        return True
    if (sum(map(lambda t2: len(t2[0]), last_span_t2_list)) / len(input_str)) >= length_ratio_threshold:
        return True
    return False

def add_book_should_or_not(input_str):
    book_symbol_list = ["《", "》"]
    if all(map(lambda x: x not in input_str, book_symbol_list)):
        return False
    should_fix = False
    sym_list = list(filter(lambda x: x in book_symbol_list, list(input_str)))
    if len(sym_list) % 2 != 0:
        should_fix = True
    else:
        sym_df = pd.DataFrame(np.asarray([sym_list]).reshape([-1, 2])).drop_duplicates()
        if sym_df.shape[0] != 1:
            should_fix = True
        else:
            if not (sym_df.iloc[0, 0] == book_symbol_list[0] and sym_df.iloc[0, 1] == book_symbol_list[1]):
                should_fix = True
    return should_fix

def add_book_by_drop(input_str, ner_book_cate_list = ["事件类",
                                                      "事件类_实体",
                                                      "作品类_实体"]):
    book_symbol_list = ["《", "》"]
    if all(map(lambda x: x not in input_str, book_symbol_list)):
        return input_str
    should_fix = False
    sym_list = list(filter(lambda x: x in book_symbol_list, list(input_str)))
    if len(sym_list) % 2 != 0:
        should_fix = True
    else:
        sym_df = pd.DataFrame(np.asarray([sym_list]).reshape([-1, 2])).drop_duplicates()
        if sym_df.shape[0] != 1:
            should_fix = True
        else:
            if not (sym_df.iloc[0, 0] == book_symbol_list[0] and sym_df.iloc[0, 1] == book_symbol_list[1]):
                should_fix = True
    if not should_fix:
        return input_str
    #input_str_dropped = "".join(filter(lambda x: x not in book_symbol_list, list(input_str)))
    input_str_dropped = input_str
    ner_t2_list = ner(input_str_dropped)
    #print(ner_t2_list)
    req = []
    for w, f in ner_t2_list:
        #if any(map(lambda x: f.startswith(x) ,ner_book_cate_list)):
        if f == "w":
            continue
        if f in ner_book_cate_list:
            req.append("《{}》".format(w))
        else:
            req.append(w)
    return "".join(req)

def dialogue_context_fix_func(input_):
    from copy import deepcopy
    input_ = deepcopy(input_)
    assert type(input_) in [type([]), type({})]
    if type(input_) == type({}):
        input_ = input_["Dialogue Context"]
    assert type(input_) == type([])
    input_ = list(map(any_flag_drop, input_))
    req = []
    for ele in input_:
        ele = ele.strip()
        if not ele:
            continue
        last_span_t2_list = extract_last_span(ele)
        last_span_string = "".join(map(lambda t2: t2[0], last_span_t2_list))
        assert ele.endswith(last_span_string)
        should_maintain_last_span = maintain_last_span_justifyer(ele, last_span_t2_list,)
        if should_maintain_last_span:
            pass
        else:
            ele = ele[:-1 * len(last_span_string)].strip()
        ele = add_book_by_drop(ele)
        ele = ele.strip()
        if ele.endswith(",") or ele.endswith("，"):
            ele = ele[:-1] + "。"
        if ele:
            req.append(ele)
    return req
