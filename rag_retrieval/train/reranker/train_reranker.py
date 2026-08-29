import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from accelerate.utils import set_seed, ProjectConfiguration
from model_bert import CrossEncoder
from model_llm import LLMDecoder
from transformers import get_cosine_schedule_with_warmup, get_wsd_schedule
from data import PointwiseRankerDataset, GroupedRankerDataset
from torch.utils.data import DataLoader
from trainer import Trainer
from accelerate import Accelerator


def create_adamw_optimizer(
    model,
    lr,
    weight_decay = 1e-2,
    no_decay_keywords = ('bias', 'LayerNorm', 'layernorm'),
):
    parameters = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in parameters if not any(nd in n for nd in no_decay_keywords)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in parameters if any(nd in n for nd in no_decay_keywords)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer


def parse_args():
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--model_name_or_path", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument(
        "--model_type",
        type=str,
        help="choose from [bert_encoder,llm_decoder]",
    )
    parser.add_argument("--train_dataset", help="training file")
    parser.add_argument("--train_dataset_type", help="the type of training file", default="pointwise")
    parser.add_argument("--train_group_size", type=int, default=8)
    parser.add_argument("--train_label_key", help="label key of training", default="label")
    
    parser.add_argument("--val_dataset", help="validation file", default=None)
    parser.add_argument("--val_dataset_type", help="the type of validation file", default="pointwise")
    parser.add_argument("--val_label_key", help="label key of validation", default="label")
    parser.add_argument("--shuffle_rate", type=float, default=0.0)
    parser.add_argument("--output_dir", help="output dir", default="./output")
    parser.add_argument("--save_on_epoch_end", type=int, default=0)
    parser.add_argument("--num_max_checkpoints", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--stable_proportion", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument(
        "--loss_type",
        type=str,
        default="pointwise_bce",
        help="chose from [pointwise_bce, pointwise_mse, pairwise_ranknet, listwise_ce]",
    )
    parser.add_argument(
        "--log_with", type=str, default="wandb", help="wandb, tensorboard"
    )
    # args.mixed_precision 会覆盖 deepspeed config 文件中的 mixed_precision 配置，除非 args.mixed_precision = None
    parser.add_argument("--mixed_precision", type=str, default=None)
    # deepspeed config 文件中的 gradient_accumulation_steps 配置会覆盖 args.gradient_accumulation_steps
    # 所以删除 deepspeed config 文件中的 gradient_accumulation_steps 配置
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument('--gradient_checkpointing', action='store_true', help='if use gradient_checkpointing')
    parser.add_argument("--num_labels", type=int, default=1, help="mlp dim")
    parser.add_argument("--query_format", type=str, default="{}")
    parser.add_argument("--document_format", type=str, default="{}")
    parser.add_argument("--seq", type=str, default="")
    parser.add_argument("--special_token", type=str, default="")
    parser.add_argument("--max_label", type=int, default=1)
    parser.add_argument("--min_label", type=int, default=0)

    args = parser.parse_args()

    # 加载 YAML 配置文件
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 使用 YAML 配置文件中的参数覆盖命令行参数
    for key, value in config.items():
        setattr(args, key, value)

    return args


def main():

    args = parse_args()

    set_seed(args.seed)

    project_config = ProjectConfiguration(
        project_dir=str(args.output_dir) + "/runs",
        automatic_checkpoint_naming=True,
        total_limit=args.num_max_checkpoints,
        logging_dir=str(args.output_dir),
    )

    accelerator = Accelerator(
        project_config=project_config,
        log_with=args.log_with,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    accelerator.init_trackers("ranker", config=vars(args))
    accelerator.print(f"Train Args from User Input: {vars(args)}")

    if args.model_type == "bert_encoder":
        model = CrossEncoder.from_pretrained(
            model_name_or_path=args.model_name_or_path,
            loss_type=args.loss_type,
            num_labels=args.num_labels,
            query_format=args.query_format,
            document_format=args.document_format
        )
    elif args.model_type == "llm_decoder":
        model = LLMDecoder.from_pretrained(
            model_name_or_path=args.model_name_or_path,
            loss_type=args.loss_type,
            num_labels=args.num_labels,
            query_format=args.query_format,
            document_format=args.document_format,
            seq=args.seq,
            special_token=args.special_token,
        )
    else:
        raise ValueError("Model type not currently supported")
    
    if "pointwise" == args.train_dataset_type:
        train_dataset = PointwiseRankerDataset(
                data_path=args.train_dataset,
                label_key=args.train_label_key,
                target_model=model,
                max_len=args.max_len,
                max_label=args.max_label,
                min_label=args.min_label,
                shuffle_rate=args.shuffle_rate,
                tag="training",
            )
        model.train_group_size = 1
    elif "grouped" == args.train_dataset_type:
        train_dataset = GroupedRankerDataset(
                data_path=args.train_dataset,
                label_key=args.train_label_key,
                target_model=model,
                max_len=args.max_len,
                shuffle_rate=args.shuffle_rate,
                train_group_size=args.train_group_size,
                tag="training",
            )
        model.train_group_size = args.train_group_size
    else:
        raise ValueError(f"Train dataset type {args.train_dataset_type} not currently supported")
    
    if args.val_dataset:
        if "pointwise" == args.val_dataset_type:
            val_dataset = PointwiseRankerDataset(
                data_path=args.val_dataset,
                label_key=args.val_label_key,
                target_model=model,
                max_len=args.max_len,
                max_label=args.max_label,
                min_label=args.min_label,
                shuffle_rate=args.shuffle_rate,
                tag="validation",
            )
        elif "grouped" == args.val_dataset_type:
            val_dataset = GroupedRankerDataset(
                data_path=args.val_dataset,
                label_key=args.val_label_key,
                target_model=model,
                max_len=args.max_len,
                shuffle_rate=args.shuffle_rate,
                train_group_size=args.train_group_size,
                tag="validation",
            )

    if args.gradient_checkpointing:
        model.model.gradient_checkpointing_enable() # explicitly pass {"use_reentrant": False} is recommeded
        model.model.enable_input_require_grads()
    num_workers = 10
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    val_dataloader = None
    if args.val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=val_dataset.collate_fn,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    optimizer = create_adamw_optimizer(
        model, lr=float(args.lr)
    )
    assert 0 <= args.warmup_proportion <= 1
    assert 0 <= args.stable_proportion <= 1
    assert args.warmup_proportion + args.stable_proportion <= 1
    total_steps = (
        len(train_dataloader) * args.epochs
    ) // accelerator.gradient_state.num_steps
    num_warmup_steps = int(args.warmup_proportion * total_steps)
    num_stable_steps = int(args.stable_proportion * total_steps)
    
    # lr_scheduler = get_cosine_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=total_steps,
    # )
    lr_scheduler = get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=total_steps - num_warmup_steps - num_stable_steps
    )


    model, optimizer, lr_scheduler, train_dataloader, val_dataloader = (
        accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader, val_dataloader
        )
    )

    accelerator.wait_for_everyone()

    trainer = Trainer(
        model=model,
        tokenizer=model.tokenizer,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=val_dataloader,
        accelerator=accelerator,
        epochs=args.epochs,
        lr_scheduler=lr_scheduler,
        log_interval=args.log_interval * accelerator.gradient_state.num_steps,
        save_on_epoch_end=args.save_on_epoch_end,
    )

    accelerator.print(f"Start training for {args.epochs} epochs ...")
    trainer.train()
    accelerator.print("Training finished!")

if __name__ == "__main__":
    main()
