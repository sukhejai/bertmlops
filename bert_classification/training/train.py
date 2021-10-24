# For Model Training
import os
import pandas as pd
import numpy as np

import joblib
import torch
import json

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import dataset
import engine
from model import EntityModel

import warnings

warnings.filterwarnings('ignore')


def process_data(data_path):
    df = pd.read_csv(data_path)
    print(df.head())
    col1 = 'prob'
    col2 = 'cons'
    col3 = 'ipc'
    text_col = 'text'
    enc_target1 = preprocessing.LabelEncoder()
    enc_target2 = preprocessing.LabelEncoder()
    enc_target3 = preprocessing.LabelEncoder()

    target1 = enc_target1.fit_transform(df[col1])
    target2 = enc_target2.fit_transform(df[col2])
    target3 = enc_target3.fit_transform(df[col3])

    sentences = df[text_col].values
    return sentences, target1, target2, target3, enc_target1, enc_target2, enc_target3  # NOQA: E501


# Split the dataframe into test and train data
def split_data(sentences, target1, target2, target3):
    (train_sentences,
        test_sentences,
        train_target1,
        test_target1,
        train_target2,
        test_target2,
        train_target3,
        test_target3
     ) = model_selection.train_test_split(sentences, target1, target2, target3, random_state=21, test_size=0.25)  # NOQA: E501
    return (train_sentences, test_sentences, train_target1, test_target1, train_target2, test_target2, train_target3, test_target3)  # NOQA: E501


# Train the model, return the model
def train_model(data, train_args):
    print(f'Train Arguments - {train_args}')
    sentences, target1, target2, target3, enc_target1, enc_target2, enc_target3 = process_data(data)  # NOQA: E501
    meta_data = {"enc_target1": enc_target1,
                 "enc_target2": enc_target2,
                 "enc_target3": enc_target3}
    joblib.dump(meta_data, os.environ.get("META_PATH"))
    print(f'META FILE CREATED AND UPLOADED ON - {os.environ.get("META_PATH")}')

    # finding out no.of classes for each target
    num_target1 = len(list(enc_target1.classes_))
    num_target2 = len(list(enc_target2.classes_))
    num_target3 = len(list(enc_target3.classes_))

    print(f'Classes in target1 prob - {enc_target1.classes_}. No. of classes in target 1 - {num_target1}')  # NOQA: E501
    print(f'Classes in target2 cons - {enc_target2.classes_}. No. of classes in target 2 - {num_target2}')  # NOQA: E501
    print(f'Classes in target3 ipc - {enc_target3.classes_}. No. of classes in target 3 - {num_target3}')  # NOQA: E501

    # Spliting data into train and test
    (train_sentences, test_sentences,
     train_target1, test_target1,
     train_target2, test_target2,
     train_target3, test_target3) = split_data(sentences,
                                               target1,
                                               target2,
                                               target3)

    train_dataset = dataset.Dataset(texts=train_sentences,
                                    target1=train_target1,
                                    target2=train_target2,
                                    target3=train_target3,
                                    train_args=train_args)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=train_args.get(
                                                        'TRAIN_BATCH_SIZE'),
                                                    num_workers=2)

    valid_dataset = dataset.Dataset(texts=test_sentences, target1=test_target1,
                                    target2=test_target2, target3=test_target3,
                                    train_args=train_args)

    valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                    batch_size=train_args.get(
                                                        'VALID_BATCH_SIZE'),
                                                    num_workers=1)

    device = torch.device(os.environ.get("DEVICE"))

    model = EntityModel(num_target1=num_target1,
                        num_target2=num_target2,
                        num_target3=num_target3,
                        train_args=train_args)

    model.to(device)

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)  # NOQA: E501
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)  # NOQA: E501
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_sentences) / train_args.get('TRAIN_BATCH_SIZE') * train_args.get('EPOCHS'))  # NOQA: E501

    print(f'No. of Train Steps - {num_train_steps}')

    optimizer = AdamW(optimizer_parameters, lr=train_args.get('LEARNING_RATE'))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    best_loss = np.inf
    for epoch in range(train_args.get('EPOCHS')):
        train_loss = engine.train_fn(
            train_data_loader, model, optimizer, device, scheduler)
        test_loss = engine.eval_fn(valid_data_loader, model, device)
        print(f"Epoch = {epoch} Train Loss = {train_loss} Valid Loss = {test_loss}")  # NOQA: E501
        if test_loss < best_loss:
            reg_model = torch.save(model.state_dict(), os.environ.get("MODEL_PATH"))  # NOQA: E501
            best_loss = test_loss
    return reg_model, valid_data_loader, device  # NOQA: E501


# Evaluate the metrics for the model
def get_model_metrics(reg_model, valid_data_loader, device):
    test_loss = engine.eval_fn(valid_data_loader, reg_model, device)
    metrics = {"device": device, "loss": test_loss}
    return metrics


def main():
    print("Running train.py")
    # Define training parameters
    source_dir = "bert_classification/util"
    arg_file = os.path.join(source_dir, 'parameters.json')
    with open(arg_file) as f:
        pars = json.load(f)
    try:
        train_args = pars["training"]
    except KeyError:
        print("Could not load training values from file")
        train_args = {}

    # Load the training data as dataframe
    data_dir = "data"
    data_file = os.path.join(data_dir, 'bert_classification_input.csv')
    data = pd.read_csv(data_file)

    # Train the model
    model, valid_data_loader, device = train_model(data, train_args)

    # Log the metrics for the model
    metrics = get_model_metrics(model, valid_data_loader, device)
    for (k, v) in metrics.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()
