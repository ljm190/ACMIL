import numpy as np
from torch import nn
from carlos_aggregators.transformers import Transformer as transformer_module

from utils.utils import initialize_weights

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class MILTransformer(nn.Module):
    def __init__(self, input_size=384, hidden_size=256, num_classes=2, device='cpu', multitask=False):
        super(MILTransformer, self).__init__()
        self.multitask = multitask
        heads = 5
        dim = 32 * heads
        mlp_dim = dim
        self.attention = transformer_module(input_dim=input_size, dim=dim,
                                            mlp_dim=mlp_dim, heads=heads, depth=1)#, emb_dropout=0.2 ,dropout=0.2)
        self.device = device
        self.norm = nn.LayerNorm(dim)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size // 2, num_classes)
        )
        # Regression head for multitasking
        if self.multitask:
            self.regression_head = nn.Sequential(
                nn.Linear(dim, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(hidden_size // 2, 1),
                # As we have negative and positive values we do not apply a non-linearity
            )
        # Initialize weights
        self.apply(initialize_weights)
    
    def get_attn_weights(self):
        return self.attention.transformer.layers[1][0].fn.get_attention_map().squeeze()[:,0][0]

    def forward(self, patches):
        try:
            aggregated_features = self.attention(patches)
        except Exception as e:
            print(patches.shape)
            raise e
        
        aggregated_features = self.norm(aggregated_features)
        # Classification prediction
        classification_pred = self.classification_head(aggregated_features)
        if self.multitask:
            # Regression prediction for multitask
            regression_pred = self.regression_head(aggregated_features)
            return classification_pred, regression_pred

        return classification_pred
