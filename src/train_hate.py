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
            annotators = batch['annotators_info']
            sample_id = batch['tweet_id']

            demographics = [[a["age"], a["ethnicity"], a["education"], a["party"]] for a in annotators]
            labels = [a["label"] for a in annotators]

            demog_label = {tuple(demo): label for demo, label in zip(demographics, labels)}
            annotator_ids = [a["annotator_idx"] for a in annotators]

            avatars = np.array([list(row[:2]) + [demog_label.get(tuple(row[:2]), 2)] for row in all_features],dtype=float)
            avatar_valid_indices = np.where(avatars[:, 2] != 2)[0].tolist()

            demog_to_valid_avatars = {tuple(avatars[i, :2]): i for i in avatar_valid_indices}
            annotator_to_valid_idx = {a["annotator_idx"]: demog_to_valid_avatars.get((a["age"], a["ethnicity"], a["education"], a["party"]))for a in annotators}
            avatar_tensor = torch.tensor(avatars[:, :2], dtype=torch.float, device=device)
            logits = model(input_ids, avatar_tensor, avatar_valid_indices)
            preds = torch.sigmoid(logits).cpu().detach().numpy()
            annotator_to_prediction = {ann_idx: preds[avatar_valid_indices.index(valid_idx)] for ann_idx, valid_idx in annotator_to_valid_idx.items() if valid_idx is not None}

            y_true, y_pred = [], []
            for a in annotators:
                ann_id = a["annotator_idx"]
                if ann_id in annotator_to_prediction:
                    y_true.append(float(a["label"]))
                    y_pred.append(float(annotator_to_prediction[ann_id] >= 0.5))
            if not y_true:
                continue

            p1 = sum(y_pred) / len(y_pred)
            t1 = sum(y_true) / len(y_true)
            soft_pred[sample_id] = [1 - p1, p1]
            soft_true[sample_id] = [1 - t1, t1]

            all_true_flat.extend(y_true)
            all_pred_flat.extend(y_pred)

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
            # Each batch corresponds to one tweet (with multiple annotator labels)
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].unsqueeze(0).to(device)
                annotators = batch['annotators_info']
                demographics = [[a["age"], a["ethnicity"]] for a in annotators]
                labels = [a["label"] for a in annotators]
                
                # Map demographic tuple -> label for valid avatars
                demog_label = {tuple(demo): label for demo, label in zip(demographics, labels)}
                
                annotator_ids = [a["annotator_idx"] for a in annotators]
                input_text = tokenizer.decode(input_ids[0].cpu().numpy(), skip_special_tokens=True)
                
                # Create demographic “avatars” (all possible demographic combinations).  2 = no annotation for that avatar
                avatars = np.array([list(row[:2]) + [demog_label.get(tuple(row[:2]), 2)] for row in all_features], dtype=float)
                
                # Select valid avatars (ones that have an actual label)
                avatar_valid_indices = np.where(avatars[:, 2] != 2)[0].tolist()
                
                 # Map demographic -> valid index
                demog_to_valid_avatars = {tuple(avatars[i, :2]): i for i in avatar_valid_indices} 

                # Match each annotator’s prediction to its corresponding demographic index
                annotator_to_valid_idx = {a["annotator_idx"]: demog_to_valid_avatars.get((a["age"], a["ethnicity"])) for a in annotators}
                
                avatar_tensor = torch.tensor(avatars[:, :2], dtype=torch.float).to(device)
                labels = torch.tensor(labels, dtype=torch.float32).to(device)
                preds = model(input_ids, avatar_tensor, avatar_valid_indices)
                annotator_to_prediction = {ann_idx: preds[avatar_valid_indices.index(valid_idx)] for ann_idx, valid_idx in annotator_to_valid_idx.items()}
                
                # Stack predictions for all annotators in this tweet
                annotator_preds = torch.stack(list(annotator_to_prediction.values()))  # shape [#annotators, 1]
                
                loss = loss_function(annotator_preds.view(-1), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}", flush=True)
            val_f1, val_p, val_r, val_md, val_err = evaluate_detailed(model, val_loader)
            print(f"Val F1={val_f1:.4f}, MD={val_md:.4f}, Error={val_err:.4f}", flush=True)

        test_f1, test_p, test_r, test_md, test_err = evaluate_detailed(model, test_loader)
        print(f"Test F1={test_f1:.4f}, Prec={test_p:.4f}, Rec={test_r:.4f}, MD={test_md:.4f}, Err={test_err:.4f}")
        results.append((seed, test_f1, test_p, test_r))

    res_df = pd.DataFrame(results, columns=["seed", "test_f1", "test_precision", "test_recall"])
    os.makedirs("results", exist_ok=True)
    res_df.to_csv(f"results/awan_hate_summary_{demo_cols[0]}_{demo_cols[1]}.csv", index=False)
