import logging
import os
import sys
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from dense.data import InferenceDataset, InferenceCollator


from dense.arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
from dense.data import TrainDataset, QPCollator
from dense.modeling import DenseModel
from dense.trainer import DenseTrainer as Trainer, GCTrainer

logger = logging.getLogger(__name__)


def main():
    # handle the hyper-paremeters
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments
    
    
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)


    print('-----------------------------------------1----------------------------')


    # initialize config, tokenizer and model
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

    if not os.path.isdir(data_args.inference_result_path):
        os.makedirs(data_args.inference_result_path)
    if not os.path.isdir(data_args.tensorboard_path):
        os.makedirs(data_args.tensorboard_path)
    board_writer = SummaryWriter(data_args.tensorboard_path)
    print('-----------------------------------------loading inf dataset----------------------------')

    mlen = 1536


    # load inf_dataset
    inf_dataset = InferenceDataset(data_args.inference_in_path, tokenizer, max_len=mlen)
    inf_loader_ = DataLoader(
        inf_dataset,
        # batch_size=training_args.per_device_eval_batch_size,
        batch_size=1,
        collate_fn=InferenceCollator(
            tokenizer,
            max_length=mlen,
            padding='max_length'
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    

    print('train path:',data_args.train_path)



    # load train_dataset
    train_dataset = TrainDataset(
        data_args, data_args.train_path, tokenizer,
    )


    # initialize trainer
    trainer_cls = GCTrainer if training_args.grad_cache else Trainer
    trainer = trainer_cls(
        inf_loader=inf_loader_,
        data_args=data_args,
        training_args=training_args,
        board_writer=board_writer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=QPCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
    )
    train_dataset.trainer = trainer

    # start to train

    trainer.train(
        model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
    )
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
