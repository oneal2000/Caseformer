import os
os.environ["CUDA_VISIBLE_DEVICES"] = '9'
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.device_count())
print(torch.cuda.current_device())
from tqdm import tqdm
import json
import random
import os
from collections import defaultdict
import numpy as np


print('reading anyou')
id_anyou = {}
anyoulist_ids = defaultdict(list)
with open('/home/user_name/legal/project/caseformer/gen_anyou/id_anyou.json') as f:
    lines = f.read().split('\n')
    for line in tqdm(lines):
        if not line: continue
        tmp = json.loads(line)
        id = tmp['id']
        anyous = tmp['anyou']
        id_anyou[id] = anyous
        anyous.sort()
        list_key = '-'.join(anyous)
        anyoulist_ids[list_key].append(id)

print(f'len anyou:{len(id_anyou)}')

# read the xf to generate pos and neg examples
print('reading id_xf...')
qid_xf = json.loads(open('/home/user_name/legal/project/pretrain_legal/process/cat_512/calc_statistics/id_xf.json').read())




xf_IPF = {}
with open('/home/user_name/legal/project/caseformer/1208_new/data/IPF.csv') as f:
    lines = f.read().split('\n')
    for line in tqdm(lines):
        if not line:continue
        xf,s_IPF = line.split(',')
        xf_IPF[int(xf)] = float(s_IPF)


writer = open('/home/user_name/legal/project/caseformer/IPF/data/qid_idx.csv','w')

all_vec=torch.zeros([4367284,507],dtype=torch.float16,device = device)

index = 0
for qid in tqdm(qid_xf):
    xfs = qid_xf[qid]
    # vec = np.zeros(507)
    for xf in xfs:
        if xf not in xf_IPF:continue
        all_vec[index][xf] = xf_IPF[xf]
    writer.write(f'{qid},{index}\n')
    index += 1


'''
4367284   507
'''


dim0, dim1 = all_vec.shape




all_re = all_vec.permute(1,0)


print(all_vec.size())
print(all_re.size())

writer_res = open('/home/user_name/legal/project/caseformer/IPF/data/res.json','w')

len = int(dim0/50)
start_idx = 0

for i in tqdm(range(len)):
    vec = all_vec[start_idx:start_idx + 50]
    vec.to(device=device)
    res = torch.mm(vec,all_re)
    
    output = res.tolist()

    writer.write(str(i) + ',' + json.dumps(output) + '\n')

    if i%100==0:torch.cuda.empty_cache()
    
torch.ran

torch.save(res,'/home/user_name/legal/project/caseformer/IPF/data/res.pt')








