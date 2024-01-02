import os
from itertools import repeat
from typing import Dict, List, Tuple, Optional, Any, Union

from transformers.trainer import Trainer
from scipy.stats import pearsonr
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from .loss import SimpleContrastiveLoss, DistributedContrastiveLoss
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

try:
    from grad_cache import GradCache
    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False


class DenseTrainer(Trainer):
    def __init__(self,inf_loader,data_args,training_args,board_writer, *args, **kwargs):
        super(DenseTrainer, self).__init__(*args, **kwargs)
        self._dist_loss_scale_factor = dist.get_world_size() if self.args.negatives_x_device else 1
        self.inf_loader = inf_loader
        self.data_args = data_args
        self.training_args = training_args
        self.board_writer = board_writer


        self.scores_label = []
        with open('/home/user_name/legal/project/caseformer/india/dataset/test/similarity_scores_all.csv') as f:
            lines = f.read().split('\n')
            for line in tqdm(lines):
                if not line:continue
                id1,id2,score = line.split(',')
                score = float(score)
                self.scores_label.append(score)



    
    def inf_india(self,output_dir):
        inf_path = self.data_args.inference_result_path + '/' + str(output_dir).split('/')[-1]
        writer = open(inf_path,'w')
        model_score = []
        for (qids, dids, encoded_qry, encoded_doc) in tqdm(self.inf_loader):
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    for k, v in encoded_qry.items():
                        encoded_qry[k] = v.to(self.training_args.device)

                    for k, v in encoded_doc.items():
                        encoded_doc[k] = v.to(self.training_args.device)
                    
                    res_qids,res_pids,res_scores = self.model.mean_inf(
                        qids,
                        dids,
                        encoded_qry,
                        encoded_doc,
                    )
                    for i in range(len(res_qids)):
                        qid = res_qids[i]
                        did = res_pids[i]
                        score = res_scores[i]
                        model_score.append(score.item())

        x = np.array(model_score[0:90])
        y = np.array(self.scores_label[0:90])

        
        x2 = np.array(model_score[90:])
        y2 = np.array(self.scores_label[90:])

        

        pc = pearsonr(x,y)

        pc1 = pearsonr(x2,y2)

        print("相关系数：",pc[0],pc1[0])
        print("显著性水平：",pc[1],pc1[1])          
        
        def Normalize(array):
            mx = np.nanmax(array)
            mn = np.nanmin(array)
            t = (array-mn)/(mx-mn)
            return t,mx,mn

        model_score,_,_ = Normalize(model_score)

        x = np.array(model_score[0:90])
        y = np.array(self.scores_label)


        
        x2 = np.array(model_score[90:])




        def mse(actual: np.ndarray, predicted: np.ndarray):
            return np.mean(np.square(actual, predicted))

        print(mse(x,y[0:90]))
        print(mse(x2,y[90:]))



    def eval(self,output_dir):

        inf_path = self.data_args.inference_result_path + '/' + str(output_dir).split('/')[-1]
        writer = open(inf_path,'w')


        for (qids, dids, encoded_qry, encoded_doc) in tqdm(self.inf_loader):
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    for k, v in encoded_qry.items():
                        encoded_qry[k] = v.to(self.training_args.device)

                    for k, v in encoded_doc.items():
                        encoded_doc[k] = v.to(self.training_args.device)
                    
                    res_qids,res_pids,res_scores = self.model.inference(
                        qids,
                        dids,
                        encoded_qry,
                        encoded_doc,
                    )
                    # print(res_scores)

                    qry_did_score = {}

                    already = set()

                    for i in range(len(res_qids)):
                        qid = res_qids[i]
                        did = res_pids[i]
                        score = res_scores[i]
                        if qid not in qry_did_score:
                            qry_did_score[qid] = []
                        if (qid,did) not in already:
                            qry_did_score[qid].append((did,score))
                            already.add((qid,did))
                    

                    from operator import itemgetter, attrgetter
                    for qid in qry_did_score:
                        qry_did_score[qid].sort(key=itemgetter(1), reverse=True)
                    
                    
                        # print(qid,did,score.item())
                    already2 = set()
                    for qid in qry_did_score:
                        candidata = qry_did_score[qid]
                        for i in range(0, len(candidata)):
                            did = candidata[i][0]
                            if (qid,did) not in already2:                              
                                score = candidata[i][1]
                                already2.add((qid,did))
                                writer.write(f'{qid} Q0 {did} {i + 1} {score} user_name\n')
        writer.close()

        def execCmd(cmd):
            r = os.popen(cmd)
            text = r.read()
            r.close()
            return text

        # qrel_path = '/home/user_name/legal/project/sigir/dense/qrel_dense_test.trec'
        qrel_path = '/home/user_name/legal/project/pretrain_legal/data/to_be_inf/qrel.trec'
        # qrel_path = '/home/zjt/datasets/v2-msmarco-doc/docv2_dev_qrels.tsv'

        cmd = f'/home/ysh/trec/trec_eval {qrel_path} ' + inf_path + ' -m all_trec'
        a = execCmd(cmd)
        # print(a)
        lines = a.split('\n')
        try:
            ndcg_30 = float(lines[62].split('\t')[2])
            ndcg_20 = float(lines[61].split('\t')[2])
            ndcg_15 = float(lines[60].split('\t')[2])
            ndcg_10 = float(lines[59].split('\t')[2])
            ndcg_5 = float(lines[58].split('\t')[2])
            print(lines[62])
            self.board_writer.add_scalar('ndcg_30', ndcg_30, self.state.global_step)
            self.board_writer.add_scalar('loss', float(self._total_loss_scalar / self.state.global_step), self.state.global_step)
            self.board_writer.add_scalar('ndcg_5', ndcg_5, self.state.global_step)
            self.board_writer.add_scalar('ndcg_10', ndcg_10, self.state.global_step)
            self.board_writer.add_scalar('ndcg_15', ndcg_15, self.state.global_step)
            self.board_writer.add_scalar('ndcg_20', ndcg_20, self.state.global_step)
        except:
            print(lines)
            print(cmd)




    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)
        self.inf_india(output_dir)

    def _prepare_inputs(
            self,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        prepared = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                prepared.append(x.to(self.args.device))
            else:
                prepared.append(super()._prepare_inputs(x))
        return prepared

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs):
        query, passage = inputs
        return model(query=query, passage=passage).loss

    def training_step(self, *args):
        return super(DenseTrainer, self).training_step(*args) / self._dist_loss_scale_factor


def split_dense_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in keys]
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    return [{arg_key: c} for c in chunked_arg_val]


def get_dense_rep(x):
    if x.q_reps is None:
        return x.p_reps
    else:
        return x.q_reps


class GCTrainer(DenseTrainer):
    def __init__(self, *args, **kwargs):
        logger.info('Initializing Gradient Cache Trainer')
        if not _grad_cache_available:
            raise ValueError(
                'Grad Cache package not available. You can obtain it from https://github.com/luyug/GradCache.')
        super(GCTrainer, self).__init__(*args, **kwargs)

        loss_fn_cls = DistributedContrastiveLoss if self.args.negatives_x_device else SimpleContrastiveLoss
        loss_fn = loss_fn_cls(self.model.data_args.train_n_passages)

        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_dense_inputs,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler
        )

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        queries, passages = self._prepare_inputs(inputs)
        queries, passages = {'query': queries}, {'passage': passages}

        _distributed = self.args.local_rank > -1
        self.gc.models = [model, model]
        loss = self.gc(queries, passages, no_sync_except_last=_distributed)

        return loss / self._dist_loss_scale_factor
