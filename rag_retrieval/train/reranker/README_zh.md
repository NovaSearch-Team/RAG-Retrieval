
# 安装环境

```bash
conda create -n rag-retrieval python=3.8 && conda activate rag-retrieval
#为了避免自动安装的torch与本地的cuda不兼容，建议进行下一步之前先手动安装本地cuda版本兼容的torch。
pip install -r requirements.txt 
```
    
               

# 微调模型

在安装好依赖后，我们通过具体的示例来展示如何利用我们自己的数据来微调开源的排序模型 (BAAI/bge-reranker-v2-m3)，或者从 BERT 类模型 (hfl/chinese-roberta-wwm-ext) 以及 LLM 类模型 (Qwen/Qwen2.5-1.5B) 从零开始训练排序模型。与此同时，我们也支持将 LLM 类模型的排序能力蒸馏到较小的 BERT 模型中去。


## 数据加载

我们提供两种数据集加载方式，用于支持不同类型的损失函数：

### 单点数据加载

单点数据的格式：都是一个query，一个doc，和其对应的label。示例见 [pointwise_reranker_train_data.jsonl](../../../example_data/pointwise_reranker_train_data.jsonl)
```
{"query": str, "content": str, "label": int|float}
```
- `content` 是 query 所对应的文档实际内容。
- `label` 是模型微调的监督信号，有两种类型：
  - 连续型：0-1 之间的连续值分数
  - 离散型：多级相关性标签（0/1/2/...），数据加载模块会将其均匀放缩成 0-1 区间的连续值。

> 当相关性是多级标签时，通过设定 `max_label` 和 `min_label` ，数据集内部会自动将多级标签均匀放缩到 0-1 分数区间中。
> 例如数据集中存在三级标签（0，1，2），经过放缩后，得到：{ label 0: 0，label 1: 0.5，label 2: 1 }。

支持两种损失函数：二分类交叉熵损失（BCE）（二分类场景和soft label下的交叉熵loss均可）和均方误差损失（MSE）。


### 分组数据加载

分组数据的格式：都是一个query，一组doc以及对应的label。示例见 [grouped_reranker_train_data_listwise_label.jsonl](../../../example_data/grouped_reranker_train_data_listwise_label.jsonl)

```
{"query": str, "hits": [{"content": str, "label": int|float}, ...]}
```
- `hits` 为 query 对应的所有文档样本，content 是文档的实际内容。
- `label` 是模型微调的监督信号，可以有两种类型：
  - 连续型：0-1 之间的连续值分数
  - 离散型：多级相关性标签（0/1/2/...）

**成对排名损失（Pairwise RankNet Loss）：**

在该loss下，rag-retrieval会自动根据query和doc对应的label，自动组成pair计算loss，并使用两者的差值进行加权。

```math
\mathcal{L}_\mathrm{RankNet}= \sum_{i=1}^M\sum_{j=1}^M \mathbb{1}_{r_{i} < r_{j} } \ |r_j-r_i|\ \log(1 + \exp(s_i-s_j))
```
  
**列表排名损失（Listwise Cross Entropy Loss）：**

在该loss下，对于数据集有严格要求，必须要求所有doc的label只有一个为1，而其他doc的label均为0，此时列表排名损失函数为标准的 listwise loss：

```math
\mathcal{L}_\text{Listwise CE} \Rightarrow \mathcal{L}_\text{listwise}=-\sum_{i=1}^M\mathbb{1}_{r_i=1}\log(\frac{\exp(s_i)}{\sum_j\exp(s_j)})
```

其中，$`\mathbf{1}_{r_i=1}`$ 是示性函数，其含义为：当 $`r_i=1`$ 时，$`\mathbb{1}_{r_i=1}=1`$；当 $`r_i\neq 1`$ 时，$`\mathbb{1}_{r_i=1}=0`$ 。 

## 训练

BERT 类模型训练, fsdp(ddp)
```bash
CUDA_VISIBLE_DEVICES="0,1" nohup accelerate launch \
--config_file ../../../config/xlmroberta_default_config.yaml \
train_reranker.py \
--config config/training_bert.yaml \
> ./logs/training_bert.log &
```

LLM 类模型训练, deepspeed（仅适用于zero 1-2, zero 3 暂不适配【保存模型的时候有 bug】）
```bash
CUDA_VISIBLE_DEVICES="0,1" nohup accelerate launch \
--config_file ../../../config/deepspeed/deepspeed_zero1.yaml \
train_reranker.py \
--config config/training_llm.yaml \
> ./logs/training_llm_deepspeed1.log &
```

## **参数解释**

多卡训练config_file:

- 对于 BERT 类模型，默认使用fsdp来支持多卡训练模型，以下是配置文件的示例。
  - [default_fsdp](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/default_fsdp.yaml)： 如果要在 hfl/chinese-roberta-wwm-ext 的基础上从零开始训练的排序，采用该配置文件
  -  [xlmroberta_default_config](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/xlmroberta_default_config.yaml)： 如果要在 BAAI/bge-reranker-base、maidalun1020/bce-reranker-base_v1、BAAI/bge-reranker-v2-m3 的基础上进行微调，采用该配置文件，因为其都是在多语言的 XLMRoberta 的基础上训练而来

- 对于 LLM 类模型，建议使用 deepspeed 来支持多卡训练模型，目前只支持 zero1 和 zero2 的训练阶段，以下是配置文件的示例
  - [deepspeed_zero1](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/deepspeed/deepspeed_zero1.yaml)
  - [deepspeed_zero2](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/deepspeed/deepspeed_zero2.yaml)

- 多卡训练配置文件修改:
  - 修改命令中的 CUDA_VISIBLE_DEVICES="0" 为你想要设置的多卡
  - 修改上述提到的配置文件的 num_processes 为你想要跑的卡的数量


模型方面：
- `model_name_or_path`：开源的reranker模型的名称或下载下来的本地服务器位置。例如：BAAI/bge-reranker-base, maidalun1020/bce-reranker-base_v1，也可以从零开始训练，例如BERT: hfl/chinese-roberta-wwm-ext 和LLM: Qwen/Qwen2.5-1.5B）
- `model_type`：当前支持 bert_encoder或llm_decoder类模型
- `max_len`：数据支持的最大输入长度

数据集方面：
- `train_dataset`：训练数据集，具体格式见上文
- `val_dataset`：验证数据集，格式同训练集(如果没有，设置为 None 即可)
- `max_label`：单点数据集中的最大 label，默认为 1
- `min_label`：单点数据集中的最小 label，默认为 0

训练方面：
- `output_dir`：训练过程中保存的 checkpoint 和最终模型的目录
- `loss_type`：从 point_ce（交叉熵损失）和 point_mse（均方损失） 中选择
- `epoch`：模型在训练数据集上训练的轮数
- `lr`：学习率，一般1e-5到5e-5之间
- `batch_size`：每个 batch 中 query-doc pair 对的数量
- `seed`：设置统一种子，用于实验结果的复现
- `warmup_proportion`：学习率预热步数占模型更新总步数的比例，如果设置为 0，那么不进行学习率预热，直接从设置的 `lr` 进行余弦衰退
- `stable_proportion`：学习率稳定不变的步数占模型更新总步数的比例，默认是 0
- `gradient_accumulation_steps`：梯度累积步数，模型实际的 batch_size 大小等于 `batch_size` * `gradient_accumulation_steps` * `num_of_GPUs`
- `mixed_precision`：是否进行混合精度的训练，以降低显存的需求。混合精度训练通过在计算使用低精度，更新参数用高精度，来优化显存占用。并且 bf16（Brain Floating Point 16）可以有效降低 loss scaling 的异常情况，但该类型仅被部分硬件支持
- `save_on_epoch_end`：是否在每一个 epoch 结束后都保存模型
- `num_max_checkpoints`：控制单次训练下保存的最多 checkpoints 数目
- `log_interval`：模型每更新 x 次参数记录一次 loss
- `log_with`：可视化工具，从 wandb 和 tensorboard 中选择

模型参数：
- `num_labels`：模型输出 logit 的数目，即为模型分类类别的个数，一般默认设置为 1
- 对于 LLM 用于判别式排序打分时，需要人工构造输入格式，由此引入下列参数
  - `query_format`, e.g. "query: {}"
  - `document_format`, e.g. "document: {}" 
  - `seq`：分隔 query 和 document 部分, e.g. " "
  - `special_token`：预示着 document 内容的结束，引导模型开始打分，理论上可以是任何 token, e.g. "\</s>" 
  - 整体的格式为："query: xxx document: xxx\</s>" 


# 加载模型进行预测

对于保存的模型，你可以很容易加载模型来进行预测。

Cross-Encoder 模型（BERT-like）
```python
ckpt_path = "./bge-reranker-m3-base"
reranker = CrossEncoder.from_pretrained(
    model_name_or_path=ckpt_path,
    num_labels=1,  # binary classification
)
reranker.model.to("cuda:0")
reranker.eval()

input_lst = [
    ["我喜欢中国", "我喜欢中国"],
    ["我喜欢美国", "我一点都不喜欢美国"]
]

res = reranker.compute_score(input_lst)

print(torch.sigmoid(res[0]))
print(torch.sigmoid(res[1]))
```

LLM-Decoder 模型 （基于 MLP 进行标量映射）

> 为了满足 LLM 如 "Qwen/Qwen2.5-1.5B" 用于判别式排序的特殊情况，设计了相关格式，实际效果为："query: {xxx} document: {xxx}\</s>"，实验显示 \</s> 的引入对 LLM 排序性能提升较大 [源于 https://arxiv.org/abs/2411.04539 section 4.3]。

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
    ["我喜欢中国", "我喜欢中国"],
    ["我喜欢美国", "我一点都不喜欢美国"],
]

res = reranker.compute_score(input_lst)

print(torch.sigmoid(res[0]))
print(torch.sigmoid(res[1]))
```

