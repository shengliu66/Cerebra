import numpy as np

import torch
from torch import nn

class SelfAttentionAverage(nn.Module):
    def __init__(self, embed_dim=768, weight_dim=128, dropout=0.0, classifier_dropout=0.0):
        super(SelfAttentionAverage, self).__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads=1,
            kdim=weight_dim,
            vdim=weight_dim,
            batch_first=True,
            dropout=dropout
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, weight_dim),
            nn.ReLU(),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(weight_dim, 1)
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights with smaller values for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with smaller values for numerical stability."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Also initialize attention weights
        for name, param in self.attn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x, attn_mask=None, avgerage_attn_weights=False):

        attn_output, attn_weights = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            average_attn_weights=avgerage_attn_weights
        )

        pooled_output = torch.mean(attn_output, dim=1)
        pooled_output_norm = self.norm(pooled_output)

        log_reg = self.mlp(pooled_output_norm).squeeze(-1)  # Only squeeze last dimension to preserve batch dimension

        return log_reg, attn_weights.squeeze(dim=1), attn_output


class SentenceAttentionBERT(nn.Module):
    def __init__(self, sentence_embed_dim=768, weight_dim=768, dropout=0.1, classifier_dropout=0.1):
        super().__init__()
        
        # Don't cap weight_dim - instead use better initialization
        self.sentence_attention = SelfAttentionAverage(
            embed_dim=sentence_embed_dim,
            weight_dim=weight_dim,
            dropout=dropout,
            classifier_dropout=classifier_dropout
        )
        
        # Initialize all parameters with smaller variance
        self._init_weights()

    def _init_weights(self):
        """Initialize all weights with smaller values for Cox model stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # x: [B, S, 768] — a sequence of sentence embeddings
        logits, attn_weights, attn_output = self.sentence_attention(x)
        
        # Clamp output to prevent extreme values in Cox loss
        logits = torch.clamp(logits, min=-10, max=10)

        return logits, attn_weights.sum(dim=-2), attn_output