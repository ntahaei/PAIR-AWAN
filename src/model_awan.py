import torch
import torch.nn as nn
from transformers import AutoModel

class AWAttention(nn.Module):
        """
    Annotation-Wise Attention Mechanism.
    Inputs:
      - A: demographic feature matrix of shape [n_avatar, n_feat]
           (e.g., [age, ethnicity] pairs for each avatar)
      - X: contextual representation from BERT of shape [seq_len, model_dim]

    Process:
      - Projects A and X into a shared h_dim space via linear transformations.
      - Computes similarity (attention) between A and X.
      - Aggregates contextual information from X, weighted by attention scores.

    Output:
      - Contextually informed demographic representations [n_avatar, model_dim]
    """
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
        """
    AWAN was built on top of a pretrained BERT model.
    
    This model combines textual and demographic representations to simulate 
    per-avatar predictions. Each "avatar" corresponds to a possible 
    demographic combination (e.g., [age, ethnicity]) that gets its own classifier head.

    Architecture:
      - Base encoder: PLM model for tweet encoding
      - Attention block (AWAttention): aligns demographic embeddings with text
      - Multiple classifier heads: one linear layer per demographic avatar
        """
    
    def __init__(self, bert_version, n_avatar, h_dim=128):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_version)
        self.hidden_size = self.bert.config.hidden_size
        self.awattention = AWAttention(n_feat=2, h_dim=h_dim, model_dim= self.hidden_size)
        self.classifiers = nn.ModuleList([nn.Linear(h_dim, 1) for _ in range(n_avatar)])
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear( self.hidden_size, h_dim)

    def forward(self, token_ids, feats, valid_indices):
        outputs = self.bert(torch.tensor(token_ids, dtype=torch.long).to(device), return_dict=True)
        outputs = outputs['last_hidden_state'][0, :, :]
        outputs = self.awattention(feats, outputs)
        outputs = self.dropout1(outputs)
        outputs = self.fc1(outputs)
        outputs = self.dropout2(outputs)
        classified = [self.classifiers[i](outputs[i])
                      for i in range(len(self.classifiers)) if i in valid_indices]
        return torch.stack(classified)
