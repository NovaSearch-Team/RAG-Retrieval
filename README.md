<h1 align="center">RAG-Retrieval</h1>
<p align="center">
    <a href="https://pypi.org/project/rag-retrieval/#description">
            <img alt="Build" src="https://img.shields.io/pypi/v/rag-retrieval?color=brightgreen">
    </a>
<!--     <a href="https://www.pepy.tech/projects/rag-retrieval">
            <img alt="Build" src="https://static.pepy.tech/personalized-badge/rag-retrieval?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads">
    </a> -->
    <a href="https://github.com/NLPJCL/RAG-Retrieval">
            <img alt="Build" src="https://img.shields.io/badge/Contribution-Welcome-blue">
    </a>
    <a href="https://github.com/NLPJCL/RAG-Retrieval/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
</p>

[English](./README.md) | [中文](./README_zh.md)

The RAG-Retrieval offers end-to-end code for training, inference, and distillation of the RAG retrieval model.
- For training, **RAG-Retrieval supports fine-tuning of any open-source RAG retrieval models**, including embedding models (figure a,bert-based, llm-based), late interactive models (figure d,colbert), and reranker models (figure c,bert-based, llm-based).
- For inference, RAG-Retrieval focuses reranker and has developed a lightweight Python library [rag-retrieval](https://pypi.org/project/rag-retrieval/), **which provides a unified way to call any different RAG ranking models.**
- For distillation, **Distillation of support embedding models and reranker models**, support distill from a larger model to a smaller model (0.5b llm or bert-base).

![ColBERT](pictures/models.png)


# Communication between communities

[Join our WeChat group chat](https://www.notion.so/RAG-Retrieval-Roadmap-c817257e3e8a484b8850cac40a3fcf88)

# News

- 🔥 **22/05/2025**: RAG-Retrieval released Myopic Trap, an empirical study of positional bias across the full IR pipeline. We systematically evaluate a range of SOTA retrieval models—including BM25, dense embeddings, ColBERT-style models, and rerankers—on two carefully designed position-aware benchmarks: SQuAD-PosQ and FineWeb-PosQ. [Learn more](./examples/MyopicTrap/)

- **29/12/2024**: RAG-Retrieval released the core training code (stage3) of Stella and Jasper embedding model [Jasper and Stella: distillation of SOTA embedding models](https://arxiv.org/abs/2412.19048).

- **21/10/2024**: RAG-Retrieval released two different methods for Reranker tasks based on LLM, as well as a method for distilling them into BERT. [Best Practices for LLM in Reranker Tasks? A Simple Experiment Report (with code)](https://zhuanlan.zhihu.com/p/987727357)

- **05/06/2024**: Implementation of MRL loss for the Embedding model in RAG-Retrieval. [RAG-Retrieval: Making MRL Loss a Standard for Training Vector (Embedding) Models](https://zhuanlan.zhihu.com/p/701884479)

- **02/06/2024**: RAG-Retrieval implements LLM preference-based supervised fine-tuning of the RAG retriever. [RAG-Retrieval Implements LLM Preference-Based Supervised Fine-Tuning of the RAG Retriever](https://zhuanlan.zhihu.com/p/701215443)

- **05/05/2024**: Released a lightweight Python library for RAG-Retrieval. [RAG-Retrieval: Your RAG Application Deserves a better infer framework](https://zhuanlan.zhihu.com/p/692404995)

- **18/03/2024**: Released RAG-Retrieval [Introduction to RAG-Retrieval on Zhihu](https://zhuanlan.zhihu.com/p/683483778)



# Features

- **Simple yet Elegant**: Rejects complex, with a simple and understandable code structure for easy modifications.
- **Supports end-to-end fine-tuning of RAG retrieval models**: Embedding (bert-based, llm-based), late interaction models (colbert), and reranker models (bert-based, llm-based).
- **Supports fine-tuning of any open-source RAG retrieval models**: Compatible with most open-source embedding and reranker models, such as: bge (bge-embedding, bge-m3, bge-reranker), bce (bce-embedding, bce-reranker), gte (gte-embedding, gte-multilingual-reranker-base).
- **Supports distillation of larger models into smaller models**: Enables the distillation of larger LLM-based reranker or embedding models into smaller ones (e.g., a 0.5B-parameter LLM or BERT-base).
- **Advanced Algorithms**: For embedding models, supports the [MRL algorithm](https://arxiv.org/abs/2205.13147) to reduce the dimensionality of output vectors and [Stella distillation method](https://arxiv.org/abs/2412.19048).
- **Multi-gpu training strategy**: Includes deepspeed, fsdp.


# Quick Start

## Installation
For training (all):
```bash
conda create -n rag-retrieval python=3.8 && conda activate rag-retrieval
# To avoid incompatibility between the automatically installed torch and the local cuda, it is recommended to manually install the compatible version of torch before proceeding to the next step.
pip install -r requirements.txt 
```
For prediction (reranker):
```bash
# To avoid incompatibility between the automatically installed torch and the local cuda, it is recommended to manually install the compatible version of torch before proceeding to the next step.
pip install rag-retrieval
```

## Training

For different model types, please go into different subdirectories. For example:
For [embedding](https://github.com/NLPJCL/RAG-Retrieval/tree/master/rag_retrieval/train/embedding), and similarly for others. Detailed procedures can be found in the README file in each subdirectories.
```bash
cd ./rag_retrieval/train/embedding
bash train_embedding.sh
```

## inference

RAG-Retrieval has developed a lightweight Python library, [rag-retrieval](https://pypi.org/project/rag-retrieval/), which provides a unified interface for calling various RAG reranker models with the following features:

- Supports multiple ranking models: Compatible with common open-source ranking models (Cross Encoder Reranker, Decoder-Only LLM Reranker).

- Long document friendly: Supports two different handling logics for long documents (maximum length truncation and splitting to take the maximum score).

- Easy to Extend: If there is a new ranking model, users only need to inherit from BaseReranker and implement the rank and compute_score functions.

**For detailed usage and considerations of the rag-retrieval package, please refer to the [Tutorial](https://github.com/NLPJCL/RAG-Retrieval/blob/master/examples/Reranker_Tutorial.md)**



# Experimental Results


## Results of the reranker model on the MTEB Reranking task


|      **Model**       |  **Model Size(GB)**  |**T2Reranking** | **MMarcoReranking** | **CMedQAv1** | **CMedQAv2** | **Avg** |
|:-----------:|:----------:|:----------:|:-------------:|:--------------:|:---------------:| :---------------:|
|   bge-reranker-base   |  1.11 | 67.28    |      35.46     |      81.27      |       84.10      | 67.03
| bce-reranker-base_v1 |   1.11 |70.25    |      34.13     |      79.64      |       81.31      | 66.33
| rag-retrieval-reranker |  0.41 | 67.33    |      31.57     |      83.54     |       86.03     | 67.12

Among them, rag-retrieval-reranker is the result of training on the hfl/chinese-roberta-wwm-ext model using the RAG-Retrieval code, and the training data uses the training data of the bge-rerank model.

## Results of the Colbert model in the MTEB Reranking task

|      **Model**  | **Model Size(GB)**  | **Dim**  | **T2Reranking** | **MMarcoReranking** | **CMedQAv1** | **CMedQAv2** | **Avg** |
|:-----------: |:----------:|:----------:|:----------:|:-------------:|:--------------:|:---------------:| :---------------:|
|   bge-m3-colbert   | 2.24 | 1024 | 66.82 | 26.71    |      75.88     |      76.83      |      61.56      
| rag-retrieval-colbert | 0.41 |  1024|  66.85    |      31.46     |      81.05     |       84.22     | 65.90

Among them, rag-retrieval-colbert is the result of training on the hfl/chinese-roberta-wwm-ext model using the RAG-Retrieval code, and the training data uses the training data of the bge-rerank model.

## Fine-tune the open source BGE series models with domain data

|      **Model**  | **T2ranking**  | |
|:-----------: |:----------:|:----------:|
|   bge-v1.5-embedding   | 66.49|  | 
|   bge-v1.5-embedding **finetune**    | 67.15 | **+0.66** | 
|   bge-m3-colbert   | 66.82|  | 
|   bge-m3-colbert **finetune**    | 67.22 | **+0.40** | 
|   bge-reranker-base   | 67.28|  | 
|   bge-reranker-base  **finetune**    | 67.57 | **+0.29** | 

The number with finetune at the end means that we used RAG-Retrieval to fine-tune the corresponding open source model, and the training data used the training set of T2-Reranking.

It is worth noting that the training set of the three open source models of bge already includes T2-Reranking, and the data is relatively general, so the performance improvement of fine-tuning using this data is not significant. However, if the open source model is fine-tuned using a vertical field data set, the performance improvement will be greater.


# Citation
If you find this repository helpful, please cite our work:
```bib
@misc{zhang2025jasperstelladistillationsota,
      title={Jasper and Stella: distillation of SOTA embedding models}, 
      author={Dun Zhang and Jiacheng Li and Ziyang Zeng and Fulong Wang},
      year={2025},
      eprint={2412.19048},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2412.19048}, 
}
```

# Acknowledge

During the development process, we borrowed or based on the implementation of the following projects. We sincerely appreciate the efforts of these teams for their contributions to open-source research and development.

- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- [uniem](https://github.com/wangyuxinwhy/uniem)
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [rerankers](https://github.com/AnswerDotAI/rerankers)


# Star History

[![Star History Chart](https://star-history.dera.page/svg?repos=NovaSearch-Team/RAG-Retrieval&type=Date)](https://star-history.dera.page/#NovaSearch-Team/RAG-Retrieval&Date)

# License
RAG-Retrieval is licensed under the [MIT License](https://github.com/NLPJCL/RAG-Retrieval/blob/master/LICENSE). 



