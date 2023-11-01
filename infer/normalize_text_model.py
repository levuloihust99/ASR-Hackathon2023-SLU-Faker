import re
import os
import sys
import json
import unicodedata
from typing import List, Dict, Any, Tuple
import torch
import onnxruntime
from transformers import BertTokenizer


class ITN:
    def __init__(
        self,
        model_root: str,
        gpu_id: int = None
    ):
        self.check_model_root(model_root)
        self.load_config(model_root)
        self.load_tokenizer(model_root)
        self.load_ort_session(model_root, gpu_id)

        self.maxlen = self.config.get("maxlen", 256)

        self.id2label = self.config["id2label"]
        self.id2label_task_punc = self.config["id2label_task_punc"]
        self.id2label_task_symbol = self.config["id2label_task_symbol"]
        self.id2label_task_case = self.config["id2label_task_case"]
        self.id2label_task_eop = self.config["id2label_task_eop"]

        self.id2label = {int(k): v for k, v in self.id2label.items()}
        self.id2label_task_punc = {int(k): v for k, v in self.id2label_task_punc.items()}
        self.id2label_task_symbol = {int(k): v for k, v in self.id2label_task_symbol.items()}
        self.id2label_task_case = {int(k): v for k, v in self.id2label_task_case.items()}
        self.id2label_task_eop = {int(k): v for k, v in self.id2label_task_eop.items()}

        self.threshold = 0.15

        self.del_label = "<DEL>"
        self.upper_label = "<UPPER>"
        self.title_label = "<TITLE>"
        self.keep_label = "<KEEP>"
        self.space_label = "<SPACE>"
        self.eop_label = "<EOP>"
        self.subword_label = "<SUBWORD>"
        self.start_symbol_label = "START_SYMBOL_"
        self.end_symbol_label = "END_SYMBOL_"

    @staticmethod
    def check_model_root(model_root):
        if not os.path.exists(model_root):
            sys.exit(f"{model_root} not found")

    def load_config(self, model_root):
        config_path = os.path.join(model_root, "config.json")
        if not os.path.exists(config_path):
            sys.exit("config.json not found")
        self.config: Dict[str, Any] = json.load(open(config_path, "r", encoding="utf-8"))

    def load_tokenizer(self, model_root):
        vocab_path = os.path.join(model_root, "vocab.txt")
        if not os.path.exists(vocab_path):
            sys.exit("vocab.txt not found")
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path, strip_accents=False)

    def load_ort_session(self, model_root, gpu_id: int = None):
        model_path = os.path.join(model_root, "model.onnx")
        if not os.path.exists(model_path):
            sys.exit("model.onnx not found")

        providers = ["CPUExecutionProvider"]
        if isinstance(gpu_id, int):
            providers.append(("CUDAExecutionProvider", {"device_id": gpu_id}))

        self.ort_session = onnxruntime.InferenceSession(model_path, providers=providers)

    def normalize(self, text: str, threshold: float = None):
        if threshold is not None:
            self.threshold = threshold

        text: str = self.clean(text)
        labels: List[str] = self.classify(text)
        res: str = ""
        for e in labels:
            if e.startswith("##"):
                e = e.lstrip("##")
                res += e
            else:
                res += f" {e}"
        res: str = self.post_process_output(res)
        return res

    @staticmethod
    def clean(s: str) -> str:
        s = s.replace("/", "").replace(".", ",").replace(" ?", "?")
        return " ".join(s.split())

    def classify(self, text: str):
        text = re.compile(r"\b(https|http)\b").sub(lambda x: " ".join(list(x[0])), text)
        e = self.encode_text(text)
        tokens = e["tokens"]
        ids = e["ids"]
        attn_mask = e["attn_mask"]

        with torch.no_grad():
            ort_inputs = {
                self.ort_session.get_inputs()[0].name: ids,
                self.ort_session.get_inputs()[1].name: attn_mask
            }
            ort_outputs = self.ort_session.run(None, ort_inputs)
            logits, logits_task_punc, logits_task_symbol, _, _ = ort_outputs

            ids = ids[0]
            labels = logits[0].argmax(-1)
            labels_task_punc = logits_task_punc[0].argmax(-1)
            labels_task_symbol = logits_task_symbol[0]

            return self.post_process_all_labels(tokens, labels, labels_task_punc, labels_task_symbol)

    def post_process_output(
        self,
        outputs: str
    ):
        res: str = " ".join(outputs.split(" ")).replace("\n ", "\n")
        # res: str = res.replace('"', "") if res.count('"') == 1 else res
        res: str = re.compile(r"(?<=\. ).").sub(lambda x: x[0].upper(), res)
        res: str = re.compile(r"(?<=\n).").sub(lambda x: x[0].upper(), res)
        res: str = re.compile(r"( ){2,}").sub(" ", res)
        return res.replace("\n", "\n ").strip()

    def post_process_all_labels(
        self,
        tokens: List[str],
        labels: List[int],
        labels_punc: List[int],
        labels_symbol: List[List[float]]
    ) -> Tuple[List[str]]:
        labels_str: List[str] = []

        for (
            token,
            label,
            label_punc,
            label_symbol
        ) in zip(tokens, labels, labels_punc, labels_symbol):
            # print(token, self.id2label[label], self.id2label_task_case[label_case])
            if (
                token == self.tokenizer.cls_token or
                token == self.tokenizer.sep_token
            ):
                continue
            elif token == self.tokenizer.pad_token:
                break

            label: str = self.id2label[label]
            if label == self.keep_label:
                label = token
            elif label == self.subword_label:
                label = f"##{token}" if not token.startswith("##") else token
            elif label == self.del_label:
                label = ""

            label_symbol: List[str] = [self.id2label_task_symbol[i] for i, score in enumerate(label_symbol) if score > self.threshold]
            label_start_symbol: str = ""
            label_end_symbol: str = ""
            for e in label_symbol:
                if e.startswith(self.start_symbol_label):
                    e = e.lstrip(self.start_symbol_label).replace(self.space_label, " ")
                    label_start_symbol += e
                elif e.startswith(self.end_symbol_label):
                    e = e.lstrip(self.end_symbol_label).replace(self.space_label, " ")
                    label_end_symbol += e

            if label_start_symbol:
                if label and not label.startswith("##"): # not a subword
                    label = f"{label_start_symbol}{label}"
                elif labels_str and not labels_str[-1].startswith("##"): # not a subword
                    labels_str[-1] = f"{label_start_symbol}{labels_str[-1]}"
            if label_end_symbol:
                if label:
                    label = f"{label}{label_end_symbol}"
                elif labels_str:
                    labels_str[-1] = f"{labels_str[-1]}{label_end_symbol}"

            label_punc: str = self.id2label_task_punc[label_punc]
            if label_punc != "O":
                if label:
                    label = f"{label}{label_punc}"
                elif labels_str:
                    labels_str[-1] = f"{labels_str[-1]}{label_punc}"

            # label_case: str = self.id2label_task_case[label_case]
            # if label_case == self.upper_label:
            #     if label:
            #         label = label.upper()
            #     elif labels_str:
            #         labels_str[-1] = labels_str[-1].upper()
            # elif label_case == self.title_label:
            #     if label:
            #         label = label.title()
            #     elif labels_str:
            #         labels_str[-1] = labels_str[-1].title()

            # label_eop: str = self.id2label_task_eop[label_eop]
            # if label_eop == self.eop_label:
            #     if label:
            #         label = f"{label}\n"
            #     elif labels_str:
            #         labels_str[-1] = f"{labels_str[-1]}\n"

            if label:
                labels_str.append(label)

        return labels_str

    def encode_text(self, text: str):
        text = self.normalize_text(text)

        tokens: List[str] = []
        ids: List[int] = []
        for i, word in enumerate(text.split()):
            itokens = self.tokenizer.tokenize(word)
            if itokens == [self.tokenizer.unk_token]:
                tokens.append(word)
                ids.append(self.tokenizer.unk_token_id)
            else:
                tokens.extend(itokens)
                ids.extend(self.tokenizer.convert_tokens_to_ids(itokens))

        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        ids = [self.tokenizer.cls_token_id] + ids + [self.tokenizer.sep_token_id]

        if len(ids) > self.maxlen:
            ids = ids[:self.maxlen]

        token_type_ids = [0] * len(ids)
        attn_mask = [1] * len(ids)

        # Padding
        pad_len = self.maxlen - len(ids)      
        ids.extend([self.tokenizer.pad_token_id]*pad_len)
        token_type_ids.extend([0]*pad_len)
        attn_mask.extend([0]*pad_len)

        return {
            "tokens": tokens,
            "ids": [ids],
            "token_type_ids": [token_type_ids],
            "attn_mask": [attn_mask]
        }

    @staticmethod
    def normalize_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        return " ".join(text.split())

if __name__ == "__main__":
    itn = ITN(model_root="model", gpu_id=0)

    # s = "bà tám đang tám chuyện với tám người hàng xóm"
    # s = "kính thưa các vị đại biểu kính thưa các em học sinh thân mến hôm nay trong tiết trời mùa thu của ngày tựu trường tôi rất vinh dự khi được đứng tại đây cùng các em đón lễ khai giảng đầu tiên của mình"
    s = "kính thưa quốc hội kính thưa chủ toạ quốc hội kính thưa chủ tịch nước"
    print(s)
    print(itn.normalize(s))