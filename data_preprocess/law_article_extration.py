import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import re
import random
import jieba
import pickle


re_xf_article = re.compile(r'(《.?中.?华.?人.?民.?共.?和.?国.?刑.?法.?》|《中华人民共和国刑事诉讼法》|《中华人民共和刑法》|〈〈中华人民共和国刑法〉〉|《中华人民人和国刑法》|《中华人民共和国刑罚》|《中华人民人民共和国刑法》|《中华人民共和国共和国刑法》|《中某人民共和国刑法》|《刑法》|《中华人民共国刑法》|《中国人民共和国刑法》|《中华人名共和国刑法》|《中华人民和共和国刑法》|《中华人民年共和国刑法》|《中华人某共和国刑法》|《中华某某共和国刑法》|《中华华人民共和国刑法》|《华人民共和国刑法》|《中华人民共和国人民共和国刑法》|《中华人民共和国国刑法》|《中华某共和国刑法》|《中华人民共和国刑法|《中华人民共和囯刑法》|中华人民共和国刑法》|《中民共和国刑法》).*?(判决|裁定|《|[之的]规定|条。$|款。$|项。$)')
re_noise_fact = re.compile(r'开庭审理过程中亦无异议。|开庭审理过程中无异议。') 

lmir_dic = pickle.load(open('lmir_article_dic.pkl', 'rb'))
with open('/work/mayixiao/similar_case/LeCaRD/LeCaRD_github/data/others/stopword.txt', 'r') as g:
    words = g.readlines()
stopwords = [i.strip() for i in words]
stopwords.extend(['.','（','）','-'])

def raw2jieba(s):
    tem = " ".join(jieba.cut(s, cut_all=False)).split()
    q = [i for i in tem if not i in stopwords]
    return q

# filter legal documents
def filt_line(lines):
    filt_dics = []
    cnt = 0
    for line in lines[:]:
        dic = eval(line)
        if len(dic['content']) <= 100:
            continue
    #         cnt += 1
    #         print(dic['content'])
        elif len(dic['fact']) < 60 or (len(dic['fact']) < 100 and re_noise_fact.search(dic['fact'])):
            continue
    #         cnt += 1
    #         print(dic['fact'])
    #         print(dic['content'])
        elif len(dic['reason']) <= 20:
            continue
    #         cnt += 1
    #         print(dic['reason'])
        filt_dics.append(dic)
    return filt_dics

# 抽取法条
def get_articles(s):
    num_dic = {'一':1, '二':2, '三':3, '四': 4, '五':5, '六':6, '七':7, '八':8, '九':9, '零':0}
    size_dic = {'十':10, '百':100}
    re_anum = re.compile(r'第[一二三四五六七八九十百零]+条')
    raw_article_nums = re.findall(re_anum, s)
    article_nums = []
    for raw_num in raw_article_nums:
        article_num = 0
        tmp_num = 1
        raw_num = raw_num[1:]
        for raw_char in raw_num:
            if raw_char in num_dic: tmp_num = num_dic[raw_char]
            elif raw_char in size_dic: 
                article_num += tmp_num*size_dic[raw_char]
                tmp_num = 0
            elif raw_char == '条': article_num += tmp_num
        article_nums.append(article_num)
    return article_nums
    


def add_info(filt_dics, fid):
    dics = []
    adic = {} # article: [docids]
    idx = 0
    for dic in filt_dics[:]:
        if dic['reason'] == '': continue
        try:
            start_index = dic['content'].index(dic['reason']) #+ len(dic['reason'])
        except:
            l = len(dic['reason'])//2
            while dic['reason'][:l] not in dic['content']:
                l = l // 2
            start_index = dic['content'].index(dic['reason'][-1:]) #+ l
        if re_xf_article.search(dic['content'][start_index:]):
            dic['xf_article'] = get_articles(re_xf_article.search(dic['content'][start_index:]).group())

            # add fine_grained article index
            dic['fg_article_idx'] = {}
            for anumber in dic['xf_article']:
                if str(anumber) in lmir_dic: 
                    score_vector = lmir_dic[str(anumber)].jelinek_mercer(raw2jieba(dic['reason']))
                    dic['fg_article_idx'][anumber] = score_vector #[i for i in range(len(score_vector)) if score_vector[i]==min(score_vector)]

            if len(dic['xf_article']) != 0:
                dic['ridx'] = fid + '_' + str(idx)
                for a in dic['xf_article']:
                    if a >= 102:
                        if a in adic: adic[a].append(idx)
                        else: adic[a] = [idx]
                idx += 1
                dics.append(dic)
    
# add pos case
    for idx,dic in enumerate(dics):
        candidates = []
        for a in dic['xf_article']:
            if a >= 102: 
                curr_list = adic[a]
                curr_list.remove(idx)
                candidates.extend(adic[a])
        if candidates:
            pos_case_id = random.choice(candidates)
            dic['pos_case'] = dics[pos_case_id]['fact']
            dic['pos_article'] = dics[pos_case_id]['xf_article']

            dic['pos_fg_article_idx'] = {}
            for anumber in dic['pos_article']:
                if str(anumber) in lmir_dic: 
                    score_vector = lmir_dic[str(anumber)].jelinek_mercer(raw2jieba(dics[pos_case_id]['reason']))
                    dic['pos_fg_article_idx'][anumber] = score_vector #[i for i in range(len(score_vector)) if score_vector[i]==min(score_vector)]

        else: 
            dic['pos_case'] = ''
            dic['pos_article'] = []
            dic['pos_fg_article_idx'] = {}
        
    return dics

files = os.listdir(DIR)
# files = ['crime_data_1.json']
for file_ in tqdm(files[:]):
    fid = file_.split('_')[-1].split('.')[0]
    w_file = 'data+a_%s.json'%(fid)
    lines = open(os.path.join(DIR, file_), 'r').readlines()

    filt_dics = filt_line(lines)
    filt_dics_2 = add_info(filt_dics, fid)

    with open(os.path.join(W_DIR, w_file), 'w') as f:
        for line in filt_dics_2:
            json.dump(line, f, ensure_ascii=False)
            
            f.write('\n')