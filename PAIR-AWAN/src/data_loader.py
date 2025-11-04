import pandas as pd
from torch.utils.data import Dataset

def load_split(path):
    df = pd.read_csv(path)
    df = df[['tweet_id', 'tweet_hashed', 'annotator_id',
             'hate_speech', 'Ethnicity_Scalar', 'Age_Scalar',
             'Education_x', 'Party_x']]
    df = df.dropna(subset=['tweet_hashed', 'hate_speech'])
    df['hate_speech'] = df['hate_speech'].astype(int)
    df = df.rename(columns={
        "hate_speech": "hate",
        "tweet_hashed": "tweet",
        "Ethnicity_Scalar": "ethnicity",
        "Education_x": "education",
        "Party_x": "party",
        "Age_Scalar": "age"
    })
    return df


def filter_by_annotation_count(df, min_ann=10, max_ann=15):
    counts = df['tweet_id'].value_counts()
    valid_ids = counts[(counts >= min_ann) & (counts <= max_ann)].index
    return df[df['tweet_id'].isin(valid_ids)]


def group_by_tweet(df):
    grouped = []
    for tweet_id, g in df.groupby("tweet_id"):
        grouped.append({
            "tweet_id": tweet_id,
            "text": g["tweet"].iloc[0],
            "labels": g["hate"].tolist(),
            "annotators": g["annotator_id"].tolist(),
            "demographics": g[["age", "ethnicity", "education", "party"]].values.tolist()
        })
    return grouped


class HateDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        inputs = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        annotator_data = [
            {"age": d[0], "ethnicity": d[1], "education": d[2],
             "party": d[3], "label": l, "annotator_idx": a}
            for d, l, a in zip(item["demographics"], item["labels"], item["annotators"])
        ]
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "annotators_info": annotator_data,
            "tweet_id": item["tweet_id"]
        }
