import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from scipy.stats import pearsonr
import numpy as np
import sys
from contextlib import nullcontext
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
)

from dense.arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
from dense.data import InferenceDataset, InferenceCollator
from dense.modeling import DenseOutput, DenseModel

logger = logging.getLogger(__name__)



def main():
    scores_label = []
    model_score = []
    with open('/home/user_name/legal/project/caseformer/india/dataset/test/similarity_scores_all.csv') as f:
        lines = f.read().split('\n')
        for line in tqdm(lines):
            if not line:continue
            id1,id2,score = line.split(',')
            score = float(score)
            scores_label.append(score)
    

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    # if training_args.local_rank > 0 or training_args.n_gpu > 1:
    #     raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    model = DenseModel.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )


    text_max_length = 1536

    # writer = open(data_args.inference_result_path,'w')



    inf_dataset = InferenceDataset(data_args.inference_in_path, tokenizer, max_len=text_max_length)
    inf_loader = DataLoader(
        inf_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=InferenceCollator(
            tokenizer,
            max_length=text_max_length,
            padding='max_length'
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    
    
    model = model.to(training_args.device)
    model.eval()

    a = open(data_args.inference_result_path,'w')
    a.close()

    for (qids, dids, encoded_qry, encoded_doc) in tqdm(inf_loader):
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            with torch.no_grad():

                # 把纯文本编码成长度为4096的token
                # 之后放到GPU里面
                for k, v in encoded_qry.items():
                    encoded_qry[k] = v.to(training_args.device)

                for k, v in encoded_doc.items():
                    encoded_doc[k] = v.to(training_args.device)


                
                # 把qry和psg批量传到inference函数中
                # 得到qid-did-score


                '''
                重新写一个函数   model.mean_inf
                
                这个函数跟inference一样, 只是把query 和 psg 的 embedding 取平均后再点积

                '''


                res_qids,res_pids,res_scores = model.inference(
                    qids,
                    dids,
                    encoded_qry,
                    encoded_doc,
                )
                # print(res_scores)

                


                # output
                for i in range(len(res_qids)):
                    qid = res_qids[i]
                    did = res_pids[i]
                    score = res_scores[i]
                    model_score.append(score.item())
                    # print(qid,did,score.item())
                    # with open(data_args.inference_result_path,'a') as writer:
                    #     writer.write(f'{qid}\t{did}\t{score}\n')

    print(model_score)
    print(scores_label)

    x = np.array(model_score[0:90])
    y = np.array(scores_label[0:90])

    
    x2 = np.array(model_score[90:])
    y2 = np.array(scores_label[90:])

    

    pc = pearsonr(x,y)

    # print(pc[0])

    pc1 = pearsonr(x2,y2)

    print("相关系数：",pc[0],pc1[0])
    print("显著性水平：",pc[1],pc1[1])          
    
    def Normalize(array):
        mx = np.nanmax(array)
        mn = np.nanmin(array)
        t = (array-mn)/(mx-mn)
        return t,mx,mn

    model_score,_,_ = Normalize(model_score)

    model_score /= 2

    x = np.array(model_score[0:90])
    y = np.array(scores_label)


    
    x2 = np.array(model_score[90:])




    def mse(actual: np.ndarray, predicted: np.ndarray):
        return np.mean(np.square(actual, predicted))

    print(mse(x,y[0:90]))
    print(mse(x2,y[90:]))






if __name__ == "__main__":
    main()
