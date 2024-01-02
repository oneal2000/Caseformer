from tqdm import tqdm
import json
import random
import os


BM25_PATH = '/home/user_name/legal/data/final_200.trec'
BERT_ENCODED_CORPUS = '/home/user_name/legal/project/pretrain_legal/data/encode_ultra/encoded_corpus.json'
ROBERTA_ENCODED_CORPUS = '/home/user_name/legal/project/pretrain_legal/data/encode_ultra/encoded_corpus_roberta.json'


input_encoded_corpus = BERT_ENCODED_CORPUS
output_filename = '1003_released_anyou_mid_hard.json'


OUTPUT_PATH = f'/home/user_name/legal/project/caseformer/process/sampling_strategy/bert/data/{output_filename}'
writer = open(OUTPUT_PATH, 'w')



print('reading crimes')
id_anyou = {}

with open('/home/user_name/legal/project/caseformer/gen_anyou/id_anyou.json') as f:
    lines = f.read().split('\n')
    for line in tqdm(lines):
        if not line:continue
        tmp = json.loads(line)
        id = tmp['id']
        anyou = tmp['anyou']
        id_anyou[id] = anyou


# read the whole corpus
id_embedding = {}
with open(input_encoded_corpus) as f:
    lines = f.read().split('\n')
    for line in tqdm(lines):
        if not line:continue
        tmp = json.loads(line)
        id = tmp['id']
        tokens = tmp['tokens'][0:256]
        id_embedding[id] = tokens
print(len(id_embedding))



print('reading law articles...')
qid_xf = json.loads(open('/home/user_name/legal/project/pretrain_legal/process/cat_512/calc_statistics/id_xf.json').read())




# read bm25 results
print('reading bm25 results')
qid_dids_new = {}
with open(BM25_PATH) as f:
    for i in tqdm(range(576409134)):
        line = f.readline()
        if not line:break
        idx,_,did,rank,score,_ = line.split(' ')
        qid = idx_qid[idx]
        if (qid not in qid_xf) or (qid not in id_embedding) or (qid not in id_anyou):
            continue
        if qid not in qid_dids_new:
            qid_dids_new[qid] = []
        if did in qid_xf and did in id_embedding and did in id_anyou:
            qid_dids_new[qid].append(did)




cnt = 0
cnt_error = 0
cnt_less_than_8 = 0
cnt_no_pos = 0
cnt_negs_total = 0
cnt_negs_fenmu = 0
cnt_pos_total = 0
cnt_pos_fenmu = 0

cnt_0_anyou = 0
for qid in tqdm(qid_dids_new):
    dids_new = qid_dids_new[qid]
    dids = dids_new 

    did_overlap = {}
    max_overlap = len(qid_xf[qid])
    query_anyous = id_anyou[qid]
    if len(query_anyous) == 0:
        cnt_0_anyou += 1
        continue
    pos_embeddings = []
    neg_embeddings = []

    for did in dids:

        this_anyous = id_anyou[did]
        if len(this_anyous) ==0:
            continue

        query_embedding = id_embedding[qid]
        doc_embedding = id_embedding[did]

        xf_query = qid_xf[qid]
        xf_doc = qid_xf[did]

        total_len = len(xf_query)
        overlap = 0

        for id in xf_query:
            if id in xf_doc:
                overlap += 1

        did_overlap[did] = overlap
        if overlap == max_overlap:
            add_flag = 0
            for did_anyou in this_anyous:
                if did_anyou in query_anyous:
                    add_flag = 1
                    continue
            if add_flag ==1:
                pos_embeddings.append(doc_embedding)
        else:
            neg_embeddings.append(doc_embedding)

    continue_flag = 0

    if len(neg_embeddings) < 16:
        cnt_less_than_8 += 1
        continue_flag = 1

    if len(pos_embeddings) == 0:
        cnt_no_pos += 1
        continue_flag = 1

    if continue_flag==1:
        continue



    cnt_negs_total += len(neg_embeddings)
    cnt_negs_fenmu += 1


    cnt_pos_total += len(pos_embeddings)
    cnt_pos_fenmu += 1

    the_pos_embedding = pos_embeddings

    out_negs_embeddings = neg_embeddings
    out_qry_embedding = id_embedding[qid]


    out_dict = {}
    out_dict['qry'] = {}
    out_dict['qry']['qid'] = '1'
    out_dict['qry']['query'] = out_qry_embedding

    pos_out = []
    for item in the_pos_embedding:
        pos_tmp = {}
        pos_tmp['pid'] = '1'
        pos_tmp['passage'] = item
        pos_out.append(pos_tmp)

    neg_out = []

    for item in out_negs_embeddings:
        neg_tmp = {}
        neg_tmp['pid'] = '1'
        neg_tmp['passage'] = item
        neg_out.append(neg_tmp)

    out_dict['neg'] = neg_out
    out_dict['pos'] = pos_out
    writer.write(json.dumps(out_dict, ensure_ascii=False) + '\n')
    cnt += 1


print(f'total training data: {cnt}')
print(f'Cases with non-standard formats: {cnt_0_anyou}')
print(f'average neg example: {cnt_negs_total / cnt_negs_fenmu}')
print(f'average pos example: {cnt_pos_total / cnt_pos_fenmu}')


'''
Statistics of previous run:
    total training data: 584404
    Cases with non-standard formats: 14129
    average neg example: 171.25986988453192
    average pos example: 22.45085420359888
'''














