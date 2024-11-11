import sys
import os
import re # 推理
from subprocess import Popen
import traceback
import torch
import json
import yaml
import argparse
import numpy as np # 推理
import librosa # 推理
import LangSegment # 推理
from time import time as ttime # 推理
from pathlib import Path
from text import chinese
import re

from gradio import processing_utils

import logging
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logging.getLogger("multipart.multipart").setLevel(logging.ERROR)

import os.path
## MP!!! 增加导入路径
sys.path.append(os.getcwd() + "/GPT_SoVITS")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.my_utils import clean_path
from tools.asr.config import asr_dict # 格式化
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule  # 推理
from GPT_SoVITS.module.models import SynthesizerTrn  # 推理
from GPT_SoVITS.feature_extractor import cnhubert # 推理
from GPT_SoVITS.text.cleaner import clean_text # 推理
from GPT_SoVITS.text import cleaned_text_to_sequence # 推理
from transformers import AutoModelForMaskedLM, AutoTokenizer # 推理
from tools.my_utils import load_audio # 推理
from GPT_SoVITS.module.mel_processing import spectrogram_torch # 推理

punctuation = set(['!', '?', '…', ',', '.', '-'," "])

now_dir = os.getcwd()
tmp = os.path.join(now_dir, "TEMP")

version=os.environ.get("version","v2")

# dict_language = {
#     "中文": "all_zh",#全部按中文识别
#     "英文": "en",#全部按英文识别#######不变
#     "日文": "all_ja",#全部按日文识别
#     "中英混合": "zh",#按中英混合识别####不变
#     "日英混合": "ja",#按日英混合识别####不变
#     "多语种混合": "auto",#多语种启动切分识别语种
# }

dict_language_v1 = {
    "中文": "all_zh",#全部按中文识别
    "英文": "en",#全部按英文识别#######不变
    "日文": "all_ja",#全部按日文识别
    "中英混合": "zh",#按中英混合识别####不变
    "日英混合": "ja",#按日英混合识别####不变
    "多语种混合": "auto",#多语种启动切分识别语种
}
dict_language_v2 = {
    "中文": "all_zh",#全部按中文识别
    "英文": "en",#全部按英文识别#######不变
    "日文": "all_ja",#全部按日文识别
    "粤语": "all_yue",#全部按中文识别
    "韩文": "all_ko",#全部按韩文识别
    "中英混合": "zh",#按中英混合识别####不变
    "日英混合": "ja",#按日英混合识别####不变
    "粤英混合": "yue",#按粤英混合识别####不变
    "韩英混合": "ko",#按韩英混合识别####不变
    "多语种混合": "auto",#多语种启动切分识别语种
    "多语种混合(粤语)": "auto_yue",#多语种启动切分识别语种
}
dict_language = dict_language_v1 if version =='v1' else dict_language_v2

infer_device = "cpu"
if torch.cuda.is_available():
    infer_device = "cuda"
else:
    infer_device = "cpu"

# 推理参数
splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)

cnhubert.cnhubert_base_path = cnhubert_base_path

ssl_model = cnhubert.get_model()
infer_is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
if infer_is_half == True:
    ssl_model = ssl_model.half().to(infer_device)
else:
    ssl_model = ssl_model.to(infer_device)

bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)

bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if infer_is_half == True:
    bert_model = bert_model.half().to(infer_device)
else:
    bert_model = bert_model.to(infer_device)

tokenizer = AutoTokenizer.from_pretrained(bert_path)

dtype=torch.float16 if infer_is_half == True else torch.float32

# <<EOF - 停顿辅助 - 
def turn_pause_directive_to_placeholder(content, placeholder='$$Pad$daP$$'):
    pad_directive = re.compile(r'(<\|停顿:(\d+(?:\.\d+)?)\|>)')
    all_matches = pad_directive.findall(content)
    replaced = re.sub(pad_directive, placeholder, content)
    return replaced, all_matches

def turn_placeholder_to_pause_directive(content, all_matches, placeholder='$$Pad$daP$$'):
    for i in range(len(all_matches)):
        content = content.replace(placeholder, all_matches[i][0], 1)
    return content

def turn_pause_directive_placeholder_included_text_to_slice(content, placeholder='$$Pad$daP$$'):
    return content.split(placeholder)

def rjust_texts_with_pause_directive(texts):
    new_texts = []
    pauses_map = {}
    for text in texts:
        pending_text, sub_matches = turn_pause_directive_to_placeholder(text)
        if len(sub_matches) > 0:
            items = turn_pause_directive_placeholder_included_text_to_slice(pending_text)
            for i, item in enumerate(items):
                new_texts.append(item)
                if i < len(sub_matches):
                    pauses_map[len(new_texts)-1] = sub_matches[i][1]
        else:
            new_texts.append(text)
    return new_texts, pauses_map
# EOF - 停顿辅助 - 

# 推理辅助
def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

def clean_text_inf(text, language, version):
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

def get_bert_inf(phones, word2ph, norm_text, language):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(infer_device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if infer_is_half == True else torch.float32,
        ).to(infer_device)

    return bert

def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(infer_device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T

def get_phones_and_bert(text,language, version, final=False):
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        language = language.replace("all_","")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # 因无法区别中日韩文汉字,以用户输入为准
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "zh":
            if re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"zh",version)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = get_bert_feature(norm_text, word2ph).to(infer_device)
        elif language == "yue" and re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"yue",version)
        else:
            phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if infer_is_half == True else torch.float32,
            ).to(infer_device)
    elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
        textlist=[]
        langlist=[]
        LangSegment.setfilters(["zh","ja","en","ko"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日韩文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text,language,version,final=True)

    return phones,bert.to(dtype),norm_text

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return  "\n".join(opts)

def cut4(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip(".").split(".")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    inp = inp.strip("\n")
    punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)

def process_text(texts):
    _text=[]
    if all(text in [None, " ", "\n",""] for text in texts):
        raise ValueError("请输入有效文本")
    for text in texts:
        if text in  [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text

def replace_consecutive_punctuation(text):
    punctuations = ''.join(re.escape(p) for p in punctuation)
    pattern = f'([{punctuations}])([{punctuations}])+'
    result = re.sub(pattern, r'\1', text)
    return result

def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    maxx=audio.abs().max()
    if(maxx>1):audio/=min(2,maxx)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

if infer_device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    if (
            ("16" in gpu_name and "V100" not in gpu_name.upper())
            or "P40" in gpu_name.upper()
            or "P10" in gpu_name.upper()
            or "1060" in gpu_name
            or "1070" in gpu_name
            or "1080" in gpu_name
    ):
        infer_is_half=False

if(infer_device=="cpu"):infer_is_half=False

cache= {}
def get_tts_wav(output_dir, seed, gpt_weight_path, sovits_weight_path, ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut="按中文句号。切", pad_duration=0, top_k=20, top_p=0.6, temperature=0.6, ref_free = False, speed=1,if_freeze=False,inp_refs=None):
    global version, dict_language
    if output_dir is None or output_dir == "":
        output_dir = tmp

    # MP 对应 change_gpt_weights逻辑
    hz = 50
    dict_s1 = torch.load(gpt_weight_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if infer_is_half == True:
        t2s_model = t2s_model.half()
    # t2s_model = t2s_model.to(device)
    t2s_model = t2s_model.to(infer_device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    
    # MP 对应 change_sovits_weights逻辑
    # hps = utils.get_hparams(stage=2) # ！！！ get_hparams需要解析命令行, 在infer-web.py中不能添加utils.get_hparams中相关的参数...否则报错，可以考虑修改utils.get_hparams
    dict_s2 = torch.load(sovits_weight_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"
    version = hps.model.version
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_weight_path):
        del vq_model.enc_q
    if infer_is_half == True:
        # vq_model = vq_model.half().to(device)
        vq_model = vq_model.half().to(infer_device)
    else:
        # vq_model = vq_model.to(device)
        vq_model = vq_model.to(infer_device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    dict_language = dict_language_v1 if version =='v1' else dict_language_v2
   
    # 这里开始接近原始代码
    t = []
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True
    t0 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]
    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
        print("实际输入的参考文本:", prompt_text)
    text = text.strip("\n")
    text = replace_consecutive_punctuation(text)
    # if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text
    
    print("实际输入的目标文本:", text)
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if infer_is_half == True else np.float32,
    )
    
    pad_duration_wav = np.zeros(
        int(hps.data.sampling_rate * pad_duration),## 5秒 同码率下sampling_rate为1秒码率
        dtype=np.float16 if infer_is_half == True else np.float32,
    )
    
    if not ref_free:
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                raise OSError("参考音频在3~10秒范围外，请更换！")
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if infer_is_half == True:
                wav16k = wav16k.half().to(infer_device)
                zero_wav_torch = zero_wav_torch.half().to(infer_device)
            else:
                wav16k = wav16k.to(infer_device)
                zero_wav_torch = zero_wav_torch.to(infer_device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(infer_device)
    
    # with torch.no_grad():
    #     wav16k, sr = librosa.load(ref_wav_path, sr=16000)
    #     if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
    #         raise OSError("参考音频在3~10秒范围外，请更换！")
    #     wav16k = torch.from_numpy(wav16k)
    #     zero_wav_torch = torch.from_numpy(zero_wav)
    #     if infer_is_half == True:
    #         # wav16k = wav16k.half().to(device)
    #         wav16k = wav16k.half().to(infer_device)
    #         # zero_wav_torch = zero_wav_torch.half().to(device)
    #         zero_wav_torch = zero_wav_torch.half().to(infer_device)
    #     else:
    #         # wav16k = wav16k.to(device)
    #         wav16k = wav16k.to(infer_device)
    #         # zero_wav_torch = zero_wav_torch.to(device)
    #         zero_wav_torch = zero_wav_torch.to(infer_device)
    #     wav16k = torch.cat([wav16k, zero_wav_torch])
    #     ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
    #         "last_hidden_state"
    #     ].transpose(
    #         1, 2
    #     )  # .float()
    #     codes = vq_model.extract_latent(ssl_content)
   
    #     prompt_semantic = codes[0, 0]
    t1 = ttime()
    t.append(t1-t0)
    
    if (how_to_cut == "凑四句一切"):
        text = cut1(text)
    elif (how_to_cut == "凑50字一切"):
        text = cut2(text)
    elif (how_to_cut == "按中文句号。切"):
        text = cut3(text)
    elif (how_to_cut == "按英文句号.切"):
        text = cut4(text)
    elif (how_to_cut == "按标点符号切"):
        text = cut5(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    # TODO <directive:x> 尖括号用于非文本范围
    #      [directive:音量]text[/directive:音量] 用于范围，嵌套处理方式需要着重考虑
    #      增强: text 增加类似 <directive:停顿> 的指令，用于控制停顿时长
    #      增强: text 增加类似 <directive:笑声> 的指令，用于控制停顿时长
    #      增强: text 增加类似 <directive:感叹> 的指令，用于控制停顿时长
    #      增强: text 增加类似 [directive:音量]text[/directive:音量] 的指令，用于控制音量大小
    #      增强: text 增加类似 [directive:音调]text[/directive:音调] 的指令，用于控制音调升降
    #      增强: text 增加类似 [directive:情绪]text[/directive:情绪] 的指令，用于控制配置文本情绪
    print("实际输入的目标文本(切句后):", text)
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    if not ref_free:
        phones1,bert1,norm_text1=get_phones_and_bert(prompt_text, prompt_language, version)
    
    # 停顿辅助
    texts, pauses_map = rjust_texts_with_pause_directive(texts)
    
    # for text in texts:
    for i_text, text in enumerate(texts):
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        if (text[-1] not in splits): text += "。" if text_language != "en" else "."
        print("实际输入的目标文本(每句):", text)
        phones2,bert2,norm_text2=get_phones_and_bert(text, text_language, version)
        print("前端处理后的文本(每句):", norm_text2)
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1+phones2).to(infer_device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(infer_device).unsqueeze(0)

        bert = bert.to(infer_device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(infer_device)

        t2 = ttime()
        # cache_key="%s-%s-%s-%s-%s-%s-%s-%s"%(ref_wav_path,prompt_text,prompt_language,text,text_language,top_k,top_p,temperature)
        # print(cache.keys(),if_freeze)
        if(i_text in cache and if_freeze==True):pred_semantic=cache[i_text]
        else:
            with torch.no_grad():
                pred_semantic, idx = t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if ref_free else prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=hz * max_sec,
                )
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                cache[i_text]=pred_semantic
        t3 = ttime()
        refers=[]
        if(inp_refs):
            for path in inp_refs:
                try:
                    refer = get_spepc(hps, path.name).to(dtype).to(infer_device)
                    refers.append(refer)
                except:
                    traceback.print_exc()
        if(len(refers)==0):refers = [get_spepc(hps, ref_wav_path).to(dtype).to(device)]
        audio = (vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers,speed=speed).detach().cpu().numpy()[0, 0])
        max_audio=np.abs(audio).max()#简单防止16bit爆音
        if max_audio>1:audio/=max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
        t.extend([t2 - t1,t3 - t2, t4 - t3])
        t1 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % 
           (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3]))
           )
    # yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
    #     np.int16
    # )
    od = os.path.join(output_dir, seed)
    filename = od + '/infered.wav'
    Path(od).mkdir(exist_ok=True, parents=True)
    print("save to file:", filename)
    processing_utils.audio_to_file(hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
        np.int16
    ), filename, "wav")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        help="path of json config file",
    )
    args = parser.parse_args()
    logging.info(str(args))
    if (args.config_file is None):
        logging.error("config_file is missing")
        raise ValueError("config_file is missing")
    with open(args.config_file, "r") as f:
        config = json.load(f)
    output_dir = config["output_dir"]
    seed = config["seed"]
    gpt_weight_path = config["gpt_weight_path"]
    sovits_weight_path = config["sovits_weight_path"]
    ref_wav_path = config["ref_wav_path"]
    prompt_text = config["prompt_text"]
    prompt_language = config["prompt_language"]
    text = config["text"]
    text_language = config["text_language"]
    how_to_cut = config["how_to_cut"]
    pad_duration = config["pad_duration"]
    top_k = config["top_k"]
    top_p = config["top_p"]
    temperature = config["temperature"]
    ref_free = config["ref_free"]
    speed = config["speed"]
    if_freeze = config["if_freeze"]
    inp_refs = config["inp_refs"]
    
    if speed is None:
        speed = 1
    
    if if_freeze is None:
        if_freeze = False
    
    version = config["version"]
    if version is None or version == "":
        version = "v2"
    dict_language = dict_language_v1 if version =='v1' else dict_language_v2
    
    print("seed:", seed)
    print("gpt_weight_path:", gpt_weight_path)
    print("sovits_weight_path:", sovits_weight_path)
    print("ref_wav_path:", ref_wav_path)
    print("prompt_text:", prompt_text)
    print("prompt_language:", prompt_language)
    print("text:", text)
    print("text_language:", text_language)
    print("how_to_cut:", how_to_cut)
    print("pad duration:", pad_duration)
    print("top_k:", top_k)
    print("top_p:", top_p)
    print("temperature:", temperature)
    print("ref_free:", ref_free)
    print("speed:", speed)
    print("if_freeze", if_freeze)
    print("inp_refs", inp_refs)
    
    if pad_duration is None:
        pad_duration = 0
    try:
        get_tts_wav(output_dir,seed, gpt_weight_path, sovits_weight_path, ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut, pad_duration, top_k, top_p, temperature, ref_free, speed, if_freeze, inp_refs)
    except Exception as e:
        logging.error(str(e))
        os._exit(9875)
    