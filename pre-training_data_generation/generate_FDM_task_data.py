import json
import os
from collections import defaultdict
from tqdm import tqdm

'''
read qid_xf
'''
print('reading id_xf...')
qid_xf = json.loads(open('/home/user_name/legal/project/pretrain_legal/process/cat_512/calc_statistics/id_xf.json').read())


'''
read qid_anyou
'''
id_anyou = {}
with open('/home/user_name/legal/project/caseformer/gen_anyou/id_anyou.json') as f:
    lines = f.read().split('\n')
    for line in tqdm(lines):
        if not line: continue
        tmp = json.loads(line)
        id = str(tmp['id'])
        anyou = tmp['anyou']
        id_anyou[id] = anyou



'''
json file format
{
    'qid':'',
    'pos':[],
    'neg':[]
}
'''


def calc_overlap(src,dest):
    length = len(src)
    cnt = 0
    for i in dest:
        if i in src:cnt += 1
    return cnt/length



'''
read bm25 results by os.walk
'''
cnt_output = 0
cnt_0anyou = 0
total_pos = 0
cnt_pos = 0
cnt_total = 0
writer = open('/home/user_name/legal/project/caseformer/IPF/contrastive_learning/data/pos_neg.json2','w')

# qid_dids_scores = defaultdict(list)
file = '/home/user_name/legal/project/caseformer/IPF/contrastive_learning/verify/narrow_mistake/final'
for root, dirs, files in tqdm(os.walk(file)):
    for f in tqdm(files):
        path = os.path.join(root, f)
        with open(path,'r',encoding='utf-8') as myFile:
            qid_dids_scores = defaultdict(list)
            while True:
                line = myFile.readline()
                if not line:break
                qid,did,score = line.split('\t')
                score = float(score)
                qid_dids_scores[qid].append((did,score))
            print(len(qid_dids_scores))



            for qid in qid_dids_scores:
                cnt_total += 1
                q_xfs = qid_xf[qid]
                q_anyous = id_anyou[qid]

                if len(q_anyous) == 0 :
                    cnt_0anyou += 1
                    continue

                sorted_list = sorted(qid_dids_scores[qid], key=lambda t: t[1],reverse=True)
                negs_ds = sorted_list[-16:]
                poss_ds = sorted_list[0:32]

                negs = []
                poss = []

                for pos in poss_ds:
                    poss.append(pos[0])

                for neg in negs_ds:
                    negs.append(neg[0])


                output_poss = []

                for pos in poss:

                    pos_xfs = qid_xf[pos]
                    pos_anyous = id_anyou[pos]

                    if len(pos_anyous) == 0:continue
                    

                    if calc_overlap(q_anyous,pos_anyous) == 1 and calc_overlap(q_xfs,pos_xfs) == 1:
                        output_poss.append(pos)
                    if len(output_poss) > 3:break
                
                if len(output_poss)!=0:
                    cnt_output += 1
                    tmp = {}
                    tmp['qid'] = qid
                    tmp['pos'] = output_poss
                    tmp['neg'] = negs

                    writer.write(json.dumps(tmp,ensure_ascii=False) + '\n')

                    total_pos += len(output_poss)
                    cnt_pos += 1

print(cnt_output)
print(total_pos/cnt_pos)
print(cnt_total)
print(cnt_0anyou)         


                







'''
for each query, writer train_file
'''

