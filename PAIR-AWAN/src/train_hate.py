import os, random, json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from itertools import product, combinations
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data_loader import load_split, filter_by_annotation_count, group_by_tweet, HateDataset
from src.model_awan import BertAWAN
from src.metrics import average_MD, error_rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Parameters ===
bert_version = "cardiffnlp/twitter-roberta-base-sentiment-latest"
learning_rate = 2e-5
n_epochs = 7
batch_size = 1
seeds = [42, 123]
min_annotators, max_annotators = 10, 15
max_len = 512

# === Load Data ===
def prepare_data():
    train = filter_by_annotation_count(load_split("data/splits/train_hate.csv"), min_annotators, max_annotators)
    dev   = filter_by_annotation_count(load_split("data/splits/dev_hate.csv"), min_annotators, max_annotators)
    test  = filter_by_annotation_count(load_split("data/splits/test_hate.csv"), min_annotators, max_annotators)
    return group_by_tweet(train), group_by_tweet(dev), group_by_tweet(test)

train_samples, val_samples, test_samples = prepare_data()
tokenizer = AutoTokenizer.from_pretrained(bert_version)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# === Evaluate ===
def evaluate_detailed(model, loader):
    model.eval()
    all_true, all_pred = [], []
    soft_true, soft_pred = {}, {}
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].unsqueeze(0).to(device)
            annots = batch['annotators_info']
            demo = [[a["age"], a["ethnicity"]] for a in annots]
            labels = [a["label"] for a in annots]
            all_true.extend(labels)

            preds = [random.choice([0, 1]) for _ in labels]
            all_pred.extend(preds)

            p1, t1 = sum(preds) / len(preds), sum(labels) / len(labels)
            soft_pred[batch["tweet_id"]] = [1 - p1, p1]
            soft_true[batch["tweet_id"]] = [1 - t1, t1]

    return (
        f1_score(all_true, all_pred, average="macro"),
        precision_score(all_true, all_pred, average="macro"),
        recall_score(all_true, all_pred, average="macro"),
        average_MD(list(soft_true.values()), list(soft_pred.values())),
        error_rate(list(soft_true.values()), list(soft_pred.values()))
    )

# === Training ===
demographic_values = {"age": [1,2,3,4,5], "ethnicity": [1,2,3,4,5]}
demographic_sets = list(combinations(demographic_values.keys(), 2))
print("PAIR-AWAN Hate Speech Task")

for demo_cols in demographic_sets:
    print(f"\n=== Running for demographics: {demo_cols} ===")
    demo_ranges = [demographic_values[d] for d in demo_cols]
    all_features = np.array(list(product(*demo_ranges, [2]))).astype(int)

    results = []
    for seed in seeds:
        set_seed(seed)
        model = BertAWAN(bert_version, n_avatar=len(all_features), h_dim=100).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(HateDataset(train_samples, tokenizer, max_len), batch_size=batch_size)
        val_loader = DataLoader(HateDataset(val_samples, tokenizer, max_len), batch_size=batch_size)
        test_loader = DataLoader(HateDataset(test_samples, tokenizer, max_len), batch_size=batch_size)

        for epoch in range(n_epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].unsqueeze(0).to(device)
                annots = batch['annotators_info']
                demos = [[a["age"], a["ethnicity"]] for a in annots]
                labels = torch.tensor([a["label"] for a in annots], dtype=torch.float32).to(device)
                demo_label = {tuple(d): l for d, l in zip(demos, labels)}

                avatars = np.array([list(r[:2]) + [demo_label.get(tuple(r[:2]), 2)] for r in all_features], dtype=float)
                valid_idx = np.where(avatars[:, 2] != 2)[0].tolist()
                avatar_tensor = torch.tensor(avatars[:, :2], dtype=torch.float).to(device)

                preds = model(input_ids, avatar_tensor, valid_idx)
                pred_map = [preds[valid_idx.index(i)] for i in valid_idx]
                if not pred_map:
                    continue

                pred_tensor = torch.stack(pred_map).view(-1)
                loss = loss_fn(pred_tensor, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}: Loss={total_loss:.4f}", flush=True)
            val_f1, val_p, val_r, val_md, val_err = evaluate_detailed(model, val_loader)
            print(f"Val F1={val_f1:.4f}, MD={val_md:.4f}, Error={val_err:.4f}", flush=True)

        test_f1, test_p, test_r, test_md, test_err = evaluate_detailed(model, test_loader)
        print(f"Test F1={test_f1:.4f}, Prec={test_p:.4f}, Rec={test_r:.4f}, MD={test_md:.4f}, Err={test_err:.4f}")
        results.append((seed, test_f1, test_p, test_r))

    res_df = pd.DataFrame(results, columns=["seed", "test_f1", "test_precision", "test_recall"])
    os.makedirs("results", exist_ok=True)
    res_df.to_csv(f"results/awan_hate_summary_{demo_cols[0]}_{demo_cols[1]}.csv", index=False)
