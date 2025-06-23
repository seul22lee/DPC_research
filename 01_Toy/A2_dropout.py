import torch.nn as nn

def enable_dropout(model):
    """Keep Dropout layers active during inference (for MC Dropout)"""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
