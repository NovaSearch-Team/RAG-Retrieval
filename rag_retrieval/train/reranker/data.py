import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
import json
from collections import defaultdict
from utils import map_label_to_continuous, visualize_label_distribution, shuffle_text
import random


class PointwiseRankerDataset(Dataset):
    def __init__(self, data_path=None, label_key="label", target_model=None, max_len=512, max_label=1, min_label=0, shuffle_rate=0.0, tag="train"):
        assert data_path is not None and target_model is not None
        self.model = target_model
        self.max_len = max_len
        self.label_key = label_key
        assert max_label > min_label and min_label >= 0
        self.max_label = max_label
        self.min_label = min_label
        self.map_func = lambda x: map_label_to_continuous(
            x, self.min_label, self.max_label)
        assert 0 <= shuffle_rate <= 1, "shuffle rate must be between 0 and 1"
        self.shuffle_rate = shuffle_rate  # The probability of shuffling the text
        self.tag = tag
        self.data = self.read_data(data_path)

    def read_data(self, data_path):
        # standard input data type:
        # {"query": str(required), "content": str(required), "**label": int|float}

        data = []
        label_distribution = defaultdict(int)
        with open(data_path) as f:
            for line in tqdm.tqdm(f):
                data_dic = json.loads(line.strip())
                query = data_dic["query"].strip()
                text = data_dic["content"].strip()
                if self.shuffle_rate > 0:
                    text = shuffle_text(text, self.shuffle_rate)
                label = self.map_func(data_dic.get(self.label_key, 0))
                label_distribution[f"{label:.2f}"] += 1
                data.append([query, text, label])

        # Only visualize the label distribution on the main process of distributed mode or in the single process mode
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"----- {self.tag} data -----")
            visualize_label_distribution(label_distribution)

        # standard output data type: [query, doc, label]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        all_batch_pairs = []
        all_labels = []

        for item in batch:
            all_batch_pairs.append([item[0], item[1]])
            all_labels.append(item[2])

        # 模型的 preprocess 方法实际将 (query, doc) pair 转换为模型的输入 input_ids 形式
        tokens = self.model.preprocess(
            all_batch_pairs, self.max_len)  # max_len 实际的作用由模型本身界定
        label_batch = torch.tensor(all_labels, dtype=torch.float16)

        return tokens, label_batch


class GroupedRankerDataset(Dataset):
    def __init__(self, data_path=None, label_key=None, target_model=None, max_len=512, shuffle_rate=0.0, train_group_size=8, tag="train"):
        assert data_path is not None and target_model is not None and label_key is not None
        self.model = target_model
        self.max_len = max_len
        assert 0 <= shuffle_rate <= 1, "shuffle rate must be between 0 and 1"
        self.shuffle_rate = shuffle_rate  # The probability of shuffling the text
        self.tag = tag
        assert train_group_size >= 2
        self.train_group_size = train_group_size
        self.label_key = label_key
        print(f"Using label_key: {self.label_key}")
        if tag == "train":
            print(f"Using train_group_size: {self.train_group_size}")
        self.data = self.read_data(data_path)

    def read_data(self, data_path):
        # standard input data type:
        # {"query": str, hits: List[{"content": str, "**label": int|float}]}

        data = []
        label_distribution = defaultdict(int)
        with open(data_path) as f:
            for line in tqdm.tqdm(f):
                data_dic = json.loads(line.strip())
                assert "query" in data_dic and "hits" in data_dic
                query = data_dic["query"].strip()
                if len(data_dic["hits"]) < self.train_group_size:
                    print(
                        f'Skip query: {query}({len(data_dic["hits"])}) with less than {self.train_group_size} hits')
                    continue
                random.shuffle(data_dic["hits"])
                for i in range(0, len(data_dic["hits"]), self.train_group_size):
                    group_docs = []
                    group_labels = []
                    for hit in data_dic["hits"][i:i+self.train_group_size]:
                        text = hit["content"].strip()
                        if self.shuffle_rate > 0:
                            text = shuffle_text(text, self.shuffle_rate)
                        label = hit.get(self.label_key, 0)
                        label_distribution[f"{label:.2f}"] += 1
                        group_docs.append(text)
                        group_labels.append(label)

                    if len(group_docs) < self.train_group_size:
                        print("debug", len(group_docs),
                              self.train_group_size, len(data_dic["hits"]))
                        if len(data_dic["hits"]) - len(group_docs) > self.train_group_size - len(group_docs):
                            candidate_range = len(
                                data_dic["hits"]) - len(group_docs)
                        else:
                            candidate_range = len(
                                data_dic["hits"]) - len(group_docs) + self.train_group_size - len(group_docs)
                        additional_idx = random.sample(
                            range(candidate_range), self.train_group_size - len(group_docs))
                        for add_idx in additional_idx:
                            text = data_dic["hits"][add_idx]["content"].strip()
                            if self.shuffle_rate > 0:
                                text = shuffle_text(text, self.shuffle_rate)
                            label = data_dic["hits"][add_idx].get(
                                self.label_key, 0)
                            label_distribution[f"{label:.2f}"] += 1
                            group_docs.append(text)
                            group_labels.append(label)

                    assert len(group_docs) == len(
                        group_labels) == self.train_group_size
                    if len(set(group_labels)) == 1:
                        # Skip the group with the same label
                        continue

                    data.append([query, group_docs, group_labels])

        # Only visualize the label distribution on the main process of distributed mode or in the single process mode
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"----- {self.tag} data -----")
            from collections import Counter
            label_distribution = Counter(label_distribution)
            print(f"Label Distribution: {label_distribution}")

        # standard output data type: [query, doc_list, label_list]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        all_batch_pairs = []
        all_labels = []

        for item in batch:
            for doc, score in zip(item[1], item[2]):
                all_batch_pairs.append([item[0], doc])
                all_labels.append(score)

        # 模型的 preprocess 方法实际将 (query, doc) pair 转换为模型的输入 input_ids 形式
        # [batch_size * train_group_size, max_len]
        tokens = self.model.preprocess(
            all_batch_pairs, self.max_len)  # max_len 实际的作用由模型本身界定
        label_batch = torch.tensor(all_labels, dtype=torch.float16)

        return tokens, label_batch


def test_PointwiseRankerDataset():
    from model_llm import LLMDecoder

    data_path = "../../../example_data/pointwise_reranker_eval_data.jsonl"
    ckpt_path = "Qwen/Qwen2.5-1.5B"
    reranker = LLMDecoder.from_pretrained(
        model_name_or_path=ckpt_path,
        num_labels=1,  # binary classification
        query_format="query: {}",
        document_format="document: {}",
        seq="\n",
        special_token="\nrelevance"
    )
    print("Testing PointwiseRankerDataset ...")
    dataset = PointwiseRankerDataset(
        data_path=data_path, label_key="label", target_model=reranker, max_len=512, max_label=2, min_label=0)

    dataloader = DataLoader(dataset, batch_size=32,
                            collate_fn=dataset.collate_fn)

    print(f"len(dataloader): {len(dataloader)}")

    for batch in tqdm.tqdm(dataloader):
        print(batch)
        print(batch[0]["input_ids"].shape)
        print(reranker.tokenizer.batch_decode(batch[0]["input_ids"])[0])
        break


def test_GroupedRankerDataset():
    from model_llm import LLMDecoder

    data_path = "../../../example_data/grouped_reranker_train_data.jsonl"
    ckpt_path = "Qwen/Qwen2.5-1.5B-Instruct"
    reranker = LLMDecoder.from_pretrained(
        model_name_or_path=ckpt_path,
        num_labels=1,  # binary classification
        query_format="query: {}",
        document_format="document: {}",
        seq="\n",
        special_token="\nrelevance"
    )
    print("Testing GroupedRankerDataset ...")
    dataset = GroupedRankerDataset(data_path=data_path, label_key="listwise_score",
                                   target_model=reranker, max_len=512, train_group_size=10)

    dataloader = DataLoader(dataset, batch_size=10,
                            collate_fn=dataset.collate_fn)
    # 模型实际的输入是 [batch_size * train_group_size, max_len]

    print(f"len(dataloader): {len(dataloader)}")

    for batch in tqdm.tqdm(dataloader):
        print(batch)
        print(batch[0]["input_ids"].shape)
        print(reranker.tokenizer.batch_decode(batch[0]["input_ids"])[0])
        break


if __name__ == "__main__":
    # test_PointwiseRankerDataset()
    test_GroupedRankerDataset()
