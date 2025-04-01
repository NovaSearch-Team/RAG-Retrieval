import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from accelerate.utils import set_seed, ProjectConfiguration
from model import Embedding
from model_distill import DistillEmbedding
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from data import EmbeddingDataset, EmbeddingDistillDataset
from torch.utils.data import DataLoader
from trainer import Trainer
from accelerate import Accelerator


def create_adamw_optimizer(
        model,
        lr,
        weight_decay=1e-2,
        no_decay_keywords=('bias', 'LayerNorm', 'layernorm'),
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
    parser.add_argument("--model_name_or_path", default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument(
            "--train_type",
            type=str,
            default="train",
            help="chose from [train, distill]")

    parser.add_argument("--train_dataset", help='trainset')
    parser.add_argument("--train_dataset_vec", help='distillion trainset embedding')
    parser.add_argument("--val_dataset", help="validation file", default=None)
    parser.add_argument('--shuffle', action='store_true', help='if shuffle')

    parser.add_argument('--neg_nums', type=int, default=15)
    parser.add_argument('--query_max_len', type=int, default=128)
    parser.add_argument('--passage_max_len', type=int, default=512)
    parser.add_argument('--teatch_emebedding_dim', type=int)

    parser.add_argument('--output_dir', help='output dir')
    parser.add_argument('--save_on_epoch_end', type=int, default=1, help='if save_on_epoch_end')
    parser.add_argument('--num_max_checkpoints', type=int, default=5)


    parser.add_argument('--epochs', type=int, default=2, help='epoch nums')
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument("--warmup_proportion", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument('--mixed_precision', default='fp16', help='')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    parser.add_argument("--log_with", type=str, default='wandb', help='wandb,tensorboard')
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)

    parser.add_argument('--use_mrl', action='store_true', help='if use mrl loss')
    parser.add_argument('--mrl_dims', type=str, help='list of mrl dims', default='128, 256, 512, 768, 1024, 1280, 1536, 1792')

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    for key, value in config.items():
        setattr(args, key, value)

    return args


def main():
    args = parse_args()

    set_seed(args.seed)

    project_config = ProjectConfiguration(
        project_dir=str(args.output_dir) + '/runs',
        automatic_checkpoint_naming=True,
        total_limit=args.num_max_checkpoints,
        logging_dir=str(args.output_dir),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.log_with,
        mixed_precision=args.mixed_precision,
        project_config=project_config
    )

    accelerator.init_trackers('embedding', config=vars(args))
    accelerator.print(f"Train Args from User Input: {vars(args)}")

    if args.use_mrl:
        mrl_dims = list(map(int, args.mrl_dims.split(",")))
    else:
        mrl_dims = []
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.train_type=="train":
        model = Embedding.from_pretrained(
            model_name_or_path=args.model_name_or_path,
            temperature=args.temperature,
            use_mrl=args.use_mrl,
            mrl_dims=mrl_dims
        )
        train_datast = EmbeddingDataset(
            train_data_path=args.train_dataset,
            tokenizer=tokenizer,
            neg_nums=args.neg_nums,
            query_max_len=args.query_max_len,
            passage_max_len=args.passage_max_len,
        )
        if args.val_dataset:
            val_dataset = EmbeddingDataset(
                train_data_path=args.val_dataset,
                tokenizer=tokenizer,
                neg_nums=args.neg_nums,
                query_max_len=args.query_max_len,
                passage_max_len=args.passage_max_len,
            )
    elif args.train_type=="distill":
        model = DistillEmbedding.from_pretrained(
            model_name_or_path=args.model_name_or_path,
            mrl_dims=mrl_dims,
            teatch_emebedding_dim=args.teatch_emebedding_dim
        )
        train_datast = EmbeddingDistillDataset(
            train_data_path=args.train_dataset,
            train_dataset_vec_path=args.train_dataset_vec,
            tokenizer=tokenizer,
            query_max_len=args.query_max_len,
            teatch_emebedding_dim=args.teatch_emebedding_dim
        )

    num_workers = 0
    train_dataloader = DataLoader(
        train_datast,
        batch_size=args.batch_size,
        collate_fn=train_datast.collate_fn,
        shuffle=args.shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = None
    if args.val_dataset:
        val_dataloader = DataLoader(
            val_datast,
            batch_size=args.batch_size,
            collate_fn=train_datast.collate_fn,
            shuffle=args.shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
    accelerator.print(f'train_dataloader total is : {len(train_dataloader)}')
    accelerator.print(f'train_dataloader data_type is : {train_datast.data_type}')

    optimizer = create_adamw_optimizer(
            model, lr=float(args.lr)
    )
    assert 0 <= args.warmup_proportion < 1
    total_steps = (
        len(train_dataloader) * args.epochs
    ) // accelerator.gradient_state.num_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_proportion * total_steps),
        num_training_steps=total_steps,
    )
    accelerator.print(lr_scheduler.lr_lambdas)

    model, optimizer, lr_scheduler, train_dataloader, validation_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, train_dataloader, validation_dataloader)

    accelerator.wait_for_everyone()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=val_dataloader,
        accelerator=accelerator,
        epochs=args.epochs,
        lr_scheduler=lr_scheduler,
        log_interval=args.log_interval * accelerator.gradient_state.num_steps,
        eval_steps=args.eval_steps * accelerator.gradient_state.num_steps if args.eval_steps else args.eval_steps,
        save_steps=args.save_steps * accelerator.gradient_state.num_steps if args.save_steps else args.save_steps,
        save_on_epoch_end=args.save_on_epoch_end,
        tokenizer=tokenizer,
    )

    accelerator.print(f'Start training for {args.epochs} epochs')
    trainer.train()

    accelerator.print('Training finished')
    accelerator.print('Saving model')
    save_dir = args.output_dir + '/model'

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_dir, safe_serialization=False)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
