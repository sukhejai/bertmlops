import torch
import transformers
import torch.nn as nn
import os


def loss_fn(output, target):
    loss = nn.CrossEntropyLoss()(output, target)
    return loss


class EntityModel(nn.Module):
    def __init__(self, num_target1, num_target2, num_target3, train_args):
        print(f'Train Argument from model.py - {train_args}')
        super(EntityModel, self).__init__()
        self.num_target1 = num_target1
        self.num_target2 = num_target2
        self.num_target3 = num_target3
        self.bert = transformers.BertModel.from_pretrained(
            os.environ.get("BASE_MODEL_PATH"), return_dict=False)
        list_dropout = [train_args.get('DROP_OUT_1'), train_args.get(
            'DROP_OUT_2'), train_args.get('DROP_OUT_3')]
        targets = torch.as_tensor(list_dropout)
        # print(targets)
        # print(type(targets))
        self.bert_drop_1 = nn.Dropout(targets[0])
        self.bert_drop_2 = nn.Dropout(targets[1])
        self.bert_drop_3 = nn.Dropout(targets[2])

        self.out_target1 = nn.Linear(768, self.num_target1)
        self.out_target2 = nn.Linear(768, self.num_target2)
        self.out_target3 = nn.Linear(768, self.num_target3)

    def forward(self, ids, mask, token_type_ids, target1, target2, target3):
        o1, o2 = self.bert(ids, attention_mask=mask,
                           token_type_ids=token_type_ids, return_dict=False)

        bo_target1 = self.bert_drop_1(o2)
        bo_target2 = self.bert_drop_2(o2)
        bo_target3 = self.bert_drop_3(o2)

        output_target1 = self.out_target1(bo_target1)
        output_target2 = self.out_target2(bo_target2)
        output_target3 = self.out_target3(bo_target3)

        loss_target1 = loss_fn(output_target1, target1)
        loss_target2 = loss_fn(output_target2, target2)
        loss_target3 = loss_fn(output_target3, target3)

        loss = (loss_target1 + loss_target2 + loss_target3) / 3

        return output_target1, output_target2, output_target3, loss
