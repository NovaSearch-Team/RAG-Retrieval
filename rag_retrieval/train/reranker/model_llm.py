import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tqdm
import ranking_loss


class LLMDecoder(nn.Module):
    def __init__(
        self,
        hf_model=None,
        tokenizer=None,
        loss_type="pointwise_bce",
        query_format="{}",
        document_format="{}",
        seq="",
        special_token="",
    ):
        super().__init__()

        self.model = hf_model
        self.tokenizer = tokenizer
        self.loss_type = loss_type
        self.train_group_size = None  # to be set in train_reranker.py
        self.query_format = query_format
        self.document_format = document_format
        self.seq = seq
        self.special_token = special_token

        self.special_token_ids = self.tokenizer.encode(
            self.special_token, add_special_tokens=False
        )
        # if `seq` token is "", the `sep_token_ids` will be [].
        self.sep_token_ids = self.tokenizer.encode(self.seq, add_special_tokens=False)

    def forward(self, batch, labels=None):
        output = self.model(**batch) # odict_keys(['logits', 'past_key_values'])

        if labels is not None:
            logits = output.logits.squeeze()
            if self.loss_type == "pointwise_mse":
                loss = ranking_loss.pointwise_mse(logits, labels)
            elif self.loss_type == "pointwise_bce":
                loss = ranking_loss.pointwise_bce(logits, labels)
            elif self.loss_type == "pairwise_ranknet":
                loss = ranking_loss.pairwise_ranknet(
                    logits, labels, self.train_group_size
                )
            elif self.loss_type == "listwise_ce":
                loss = ranking_loss.listwise_ce(logits, labels, self.train_group_size)
            output["loss"] = loss

        return output

    @torch.no_grad()
    def compute_score(
        self, sentences_pairs, batch_size=256, max_length=512, normalize=False
    ):
        """
        sentences_pairs=[[query,title],[query1,title1],...]
        """

        all_logits = []
        for start_index in tqdm.tqdm(range(0, len(sentences_pairs), batch_size)):
            sentences_batch = sentences_pairs[start_index : start_index + batch_size]
            batch_data = self.preprocess(sentences_batch, max_length).to(
                self.model.device
            )
            output = self.forward(batch_data)
            logits = output.logits.detach().cpu()
            all_logits.extend(logits)

        if normalize:
            all_logits = torch.sigmoid(torch.tensor(all_logits)).detach().cpu().tolist()

        return all_logits

    def preprocess(self, sentences_pairs, max_len):
        batch_input_ids = []

        for query, document in sentences_pairs:
            query_ids = self.tokenizer.encode(
                self.query_format.format(query.strip()),
                add_special_tokens=False,
            )
            document_max_len = (
                max_len
                - len(query_ids)
                - len(self.special_token_ids)
                - len(self.sep_token_ids)
            )
            document_valid_ids = self.tokenizer.encode(
                self.document_format.format(document.strip()),
                max_length=document_max_len,
                truncation=True,
                add_special_tokens=False,
            )
            batch_input_ids.append(
                query_ids
                + self.sep_token_ids
                + document_valid_ids
                + self.special_token_ids
            )
        assert len(batch_input_ids) == len(sentences_pairs)

        # padding and add special tokens of model
        tokens = self.tokenizer.prepare_for_model(
            batch_input_ids,
            add_special_tokens=True,
            padding="longest",
            truncation=False,
            return_tensors="pt",
        )
        return tokens

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        loss_type="pointwise_bce",
        num_labels=1,
        query_format="query: {}",
        document_format="document: {}",
        seq=" ",
        special_token="</s>",
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True, trust_remote_code=True
        )
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

        # if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if hf_model.config.pad_token_id is None:
            hf_model.config.pad_token_id = tokenizer.pad_token_id

        # make sure tokenizer is right pad in our logic
        # https://github.com/huggingface/transformers/blob/5d7739f15a6e50de416977fe2cc9cb516d67edda/src/transformers/models/qwen2/modeling_qwen2.py#L1201
        tokenizer.padding_side = "right"

        reranker = cls(
            hf_model,
            tokenizer,
            loss_type,
            query_format,
            document_format,
            seq,
            special_token,
        )
        return reranker

    def save_pretrained(self, save_dir, safe_serialization=False):
        # 模型的参数无论原本是分布在多张卡还是单张卡上，保存后的权重都在 CPU 上，避免了跨设备加载的潜在问题。
        def _trans_state_dict(state_dict):
            state_dict = type(state_dict)(
                {k: v.clone().cpu() for k, v in state_dict.items()}
            )
            return state_dict

        self.model.save_pretrained(
            save_dir,
            state_dict=_trans_state_dict(self.model.state_dict()),
            safe_serialization=safe_serialization,
        )


def test_LLMDecoder():
    ckpt_path = "./Qwen2-1.5B-Instruct"
    reranker = LLMDecoder.from_pretrained(
        model_name_or_path=ckpt_path,
        num_labels=1,  # binary classification
        query_format="query: {}",
        document_format="document: {}",
        seq=" ",
        special_token="</s>",
    )
    reranker.model.to("cuda:0")
    reranker.eval()

    input_lst = [
        ["我喜欢中国", "我喜欢中国"],
        ["我喜欢美国", "我一点都不喜欢美国"],
        [
            "泰山要多长时间爬上去",
            "爬上泰山需要1-8个小时，具体的时间需要看个人的身体素质。专业登山运动员可能只需要1个多小时就可以登顶，有些身体素质比较低的，爬的慢的就需要5个多小时了。",
        ],
    ]

    res = reranker.compute_score(input_lst)

    print(torch.sigmoid(res[0]))
    print(torch.sigmoid(res[1]))
    print(torch.sigmoid(res[2]))


if __name__ == "__main__":
    test_LLMDecoder()
