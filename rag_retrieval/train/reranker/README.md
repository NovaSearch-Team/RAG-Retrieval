[English | [中文](README_zh.md)]
# Environment Setup

```bash
conda create -n rag-retrieval python=3.8 && conda activate rag-retrieval
# To avoid compatibility issues between the automatically installed torch and your local CUDA, it is recommended to manually install a torch version compatible with your local CUDA before proceeding to the next step.
pip install -r requirements.txt 
```

| Requirement | Recommend |
| --------------- | ---------------- |
| accelerate | 1.0.1 |
| deepspeed | 0.15.4 |
| transformers | 4.44.2 |

# Fine-tuning the Model

After installing the dependencies, we'll use specific examples to demonstrate how to fine-tune an open-source ranking model (BAAI/bge-reranker-v2-m3) using your own data. Alternatively, you can train a ranking model from scratch using BERT-based models (hfl/chinese-roberta-wwm-ext) or LLM-based models (Qwen/Qwen2.5-1.5B). Additionally, we support distilling the ranking capabilities of LLM-based models into smaller BERT models.

## Data Loading

We provide two dataset loading methods to support different types of loss functions:

### Pointwise Data Loading

The standard format for pointwise datasets is as follows. See an example in [pointwise_reranker_train_data.jsonl](../../../example_data/pointwise_reranker_train_data.jsonl):
```
{"query": str, "content": str, "label": xx}
```
- `content` is the actual document content corresponding to the `query`.
- `label` represents the relevance label assigned through either human annotation (multi-level relevance labels: 0/1/2/...) or teacher model scoring (a continuous score between 0 and 1), which serves as the supervision signal for model fine-tuning.

This configuration supports Mean Squared Error (`MSE`) loss and Binary Cross Entropy (`BCE`) loss, with the optimization objective being the absolute relevance judgment between query and content. When relevance is represented as multi-level labels, the dataset will automatically scale these labels into the 0-1 score range using the `max_label` and `min_label` settings. For example, if the dataset contains three levels of labels (0, 1, 2), they will be scaled to `{ label 0: 0, label 1: 0.5, label 2: 1 }`. During inference, the model’s predicted score is the raw logit output, which can be normalized to the 0-1 range using a sigmoid function. Users can utilize LLMs to obtain relevance labels for distillation. Example code for LLM-based relevance annotation can be found in the directory [examples/distill_llm_to_bert_reranker](../../../examples/distill_llm_to_bert_reranker).

Complete configuration:
```
train_dataset: "../../../example_data/pointwise_reranker_train_data.jsonl"
train_dataset_type: "pointwise"
max_label: 2
min_label: 0
max_len: 512
shuffle_rate: 0.0
val_dataset: "../../../example_data/pointwise_reranker_eval_data.jsonl"
val_dataset_type: "pointwise"
loss_type: "pointwise_bce"  # "pointwise_bce" or "pointwise_mse"
```

### Grouped Data Loading

The standard format for grouped datasets is as follows. See examples in [grouped_reranker_train_data_pointwise_label.jsonl](../../../example_data/grouped_reranker_train_data_pointwise_label.jsonl) & [grouped_reranker_train_data_listwise_label.jsonl](../../../example_data/grouped_reranker_train_data_listwise_label.jsonl):
```
{"query": str, "hits": [{"content": xxx, "label": xxx}, ...]}
```
- `hits` contains all document samples under a given `query`, where `content` is the actual document content.
- `label` represents the relevance label assigned through either human annotation (multi-level relevance labels: 0/1/2/...) or teacher model scoring (not necessarily limited to a continuous score between 0 and 1, but possibly a list-based relative ranking), serving as the supervision signal for model fine-tuning.

This configuration supports `Pairwise RankNet Loss` and `Listwise Cross Entropy Loss`, with the optimization objective being the relative relevance judgment among multiple documents for a given query. The parameter `train_group_size` specifies the number of documents to consider simultaneously for each query. If the number of available documents is insufficient, samples will be duplicated to meet the `train_group_size` requirement.

Complete configuration:
```
train_dataset: "../../../example_data/grouped_reranker_train_data_listwise_label.jsonl"
train_dataset_type: "grouped"
train_group_size: 10
shuffle_rate: 0.0
max_len: 512
val_dataset: "../../../example_data/grouped_reranker_eval_data.jsonl"
val_dataset_type: "grouped"
loss_type: "pairwise_ranknet"  # "pairwise_ranknet" or "listwise_ce"
```

## Training

Training BERT-based Models with FSDP (DDP)
```bash
CUDA_VISIBLE_DEVICES="0,1" nohup accelerate launch \
--config_file ../../../config/xlmroberta_default_config.yaml \
train_reranker.py \
--config config/training_bert.yaml \
> ./logs/training_bert.log &
```

Training LLM-based Models with DeepSpeed (Only applicable to zero 1-2; zero 3 is not currently compatible due to a bug when saving the model)
```bash
CUDA_VISIBLE_DEVICES="0,1" nohup accelerate launch \
--config_file ../../../config/deepspeed/deepspeed_zero1.yaml \
train_reranker.py \
--config config/training_llm.yaml \
> ./logs/training_llm_deepspeed1.log &
```

## Parameter Explanation

### Multi-GPU Training Configuration Files
- For BERT-based models, FSDP is used by default to support multi-GPU training. Here are examples of configuration files:
  - [default_fsdp](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/default_fsdp.yaml): Use this configuration file if you want to train a ranking model from scratch based on hfl/chinese-roberta-wwm-ext.
  - [xlmroberta_default_config](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/xlmroberta_default_config.yaml): Use this configuration file if you want to fine-tune models such as BAAI/bge-reranker-base, maidalun1020/bce-reranker-base_v1, or BAAI/bge-reranker-v2-m3, as they are all trained on the multilingual XLMRoberta.
- For LLM-based models, DeepSpeed is recommended to support multi-GPU training. Currently, only the training phases of zero1 and zero2 are supported. Here are examples of configuration files:
  - [deepspeed_zero1](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/deepspeed/deepspeed_zero1.yaml)
  - [deepspeed_zero2](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/deepspeed/deepspeed_zero2.yaml)
- Modifying Multi-GPU Training Configuration Files:
  - Change `CUDA_VISIBLE_DEVICES="0"` in the command to the GPUs you want to use.
  - Modify the `num_processes` parameter in the above-mentioned configuration files to the number of GPUs you want to use.

### Model-related Parameters
- `model_name_or_path`: The name of an open-source reranker model or the local server location where it is downloaded. For example: BAAI/bge-reranker-base, maidalun1020/bce-reranker-base_v1. You can also train from scratch, such as using BERT: hfl/chinese-roberta-wwm-ext or LLM: Qwen/Qwen2.5-1.5B.
- `model_type`: Currently supports bert_encoder or llm_decoder models.
- `max_len`: The maximum input length supported by the data.

### Dataset-related Parameters
- `train_dataset`: The training dataset. See the above for the specific format.
- `val_dataset`: The validation dataset, with the same format as the training dataset (set to `None` if not available).
- `max_label`: The maximum label in the pointwise dataset, defaulting to 1.
- `min_label`: The minimum label in the pointwise dataset, defaulting to 0.

### Training-related Parameters
- `output_dir`: The directory where checkpoints and the final model are saved during training.
- `loss_type`: Choose from `point_ce` (Cross Entropy Loss) and `point_mse` (Mean Squared Error Loss).
- `epoch`: The number of epochs to train the model on the training dataset.
- `lr`: The learning rate, typically between 1e-5 and 5e-5.
- `batch_size`: The number of query-doc pairs in each batch.
- `seed`: Set a unified seed for reproducibility of experimental results.
- `warmup_proportion`: The proportion of learning rate warm-up steps to the total number of model update steps. If set to 0, no learning rate warm-up will be performed, and the learning rate will directly decay cosine-wise from the set `lr`.
- `stable_proportion`: The proportion of steps during which the learning rate remains stable to the total number of model update steps, defaulting to 0.
- `gradient_accumulation_steps`: The number of gradient accumulation steps. The actual batch size of the model is equal to `batch_size` * `gradient_accumulation_steps` * `num_of_GPUs`.
- `mixed_precision`: Whether to perform mixed-precision training to reduce GPU memory requirements. Mixed-precision training optimizes GPU memory usage by using low precision for computation and high precision for parameter updates. Additionally, bf16 (Brain Floating Point 16) can effectively reduce abnormal loss scaling situations, but this type is only supported by some hardware.
- `save_on_epoch_end`: Whether to save the model after each epoch.
- `num_max_checkpoints`: Controls the maximum number of checkpoints saved during a single training session.
- `log_interval`: Record the loss every `x` parameter updates of the model.
- `log_with`: The visualization tool, choose from `wandb` and `tensorboard`.

### Model Parameter-related Parameters
- `num_labels`: The number of logits output by the model, which is the number of classification categories of the model, usually set to 1 by default.
- When using an LLM for discriminative ranking scoring, you need to manually construct the input format, which introduces the following parameters:
  - `query_format`, e.g., "query: {}"
  - `document_format`, e.g., "document: {}"
  - `seq`: Separates the query and document parts, e.g., " "
  - `special_token`: Indicates the end of the document content and guides the model to start scoring. Theoretically, it can be any token, e.g., "\</s>"
  - The overall format is: "query: xxx document: xxx\</s>"

# Loading the Model for Prediction

You can easily load a saved model for prediction.

### Cross-Encoder Model (BERT-like)
```python
ckpt_path = "./bge-reranker-m3-base"
reranker = CrossEncoder.from_pretrained(
    model_name_or_path=ckpt_path,
    num_labels=1,  # binary classification
)
reranker.model.to("cuda:0")
reranker.eval()

input_lst = [
    ["I like China", "I like China"],
    ["I like the United States", "I don't like the United States at all"]
]

res = reranker.compute_score(input_lst)

print(torch.sigmoid(res[0]))
print(torch.sigmoid(res[1]))
```

### LLM-Decoder Model (Based on MLP for Scalar Mapping)

> To meet the special requirements of using an LLM like "Qwen/Qwen2.5-1.5B" for discriminative ranking, a specific format has been designed. The actual effect is: "query: {xxx} document: {xxx}\</s>". Experiments have shown that the introduction of \</s> significantly improves the ranking performance of the LLM [from https://arxiv.org/abs/2411.04539 section 4.3].

```python
ckpt_path = "./Qwen2-1.5B-Instruct"
reranker = LLMDecoder.from_pretrained(
    model_name_or_path=ckpt_path,
    num_labels=1,  # binary classification
    query_format="query: {}",
    document_format="document: {}",
    seq="\n",
    special_token="\nrelevance",
)
reranker.model.to("cuda:0")
reranker.eval()

input_lst = [
    ["I like China", "I like China"],
    ["I like the United States", "I don't like the United States at all"],
]

res = reranker.compute_score(input_lst)

print(torch.sigmoid(res[0]))
print(torch.sigmoid(res[1]))
```