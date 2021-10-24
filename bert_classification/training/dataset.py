import torch
import transformers
import os


class Dataset:
    def __init__(self, texts, target1, target2, target3, train_args):
        print(f'Train Argument from dataset.py - {train_args}')
        self.texts = texts
        self.target1 = target1
        self.target2 = target2
        self.target3 = target3
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            os.environ.get("BASE_MODEL_PATH"), do_lower_case=True)
        self.max_len = train_args.get('MAX_LEN')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        texts = str(self.texts[item])
        texts = " ".join(texts.split())
        inputs = self.tokenizer.encode_plus(
            texts,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target1": torch.tensor(self.target1[item], dtype=torch.long),
            "target2": torch.tensor(self.target2[item], dtype=torch.long),
            "target3": torch.tensor(self.target3[item], dtype=torch.long)
        }
