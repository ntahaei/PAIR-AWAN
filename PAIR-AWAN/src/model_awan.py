import torch
import torch.nn as nn
from transformers import AutoModel

class AWAttention(nn.Module):
    def __init__(self, n_feat, h_dim, model_dim):
        super().__init__()
        self.query = nn.Linear(n_feat, h_dim)
        self.key = nn.Linear(model_dim, h_dim)

    def forward(self, A, X):
        Q = self.query(A)
        K = self.key(X)
        att_scores = torch.softmax(torch.matmul(Q, K.transpose(0, 1)), dim=-1)
        return torch.matmul(att_scores, X)


class BertAWAN(nn.Module):
    def __init__(self, bert_version, n_avatar, h_dim=128):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_version)
        hidden = self.bert.config.hidden_size
        self.awattention = AWAttention(n_feat=2, h_dim=h_dim, model_dim=hidden)
        self.classifiers = nn.ModuleList([nn.Linear(h_dim, 1) for _ in range(n_avatar)])
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden, h_dim)

    def forward(self, token_ids, feats, valid_indices):
        device = feats.device
        outputs = self.bert(token_ids, return_dict=True)['last_hidden_state'][0]
        outputs = self.awattention(feats, outputs)
        outputs = self.dropout1(outputs)
        outputs = self.fc1(outputs)
        outputs = self.dropout2(outputs)
        classified = [self.classifiers[i](outputs[i])
                      for i in range(len(self.classifiers)) if i in valid_indices]
        return torch.stack(classified)
