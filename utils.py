import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Dict, Any


class BaseModel(nn.Module):
    """Base class for models to standardise signatures."""

    def forward(
        self, x: Tensor, is_training: bool = False
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        raise NotImplementedError

    def calculate_loss(
        self,
        outputs: Tuple[Tensor, Dict[str, Any]],
        targets: Tensor,
        criterion: nn.Module,
    ) -> Tensor:
        logits, _ = outputs
        return criterion(logits, targets)


class DimReduction(nn.Module):
    def __init__(self, num_channels: int, m_dim: int, num_res_blocks: int = 0) -> None:
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(num_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.num_res_blocks = num_res_blocks

        res_blocks_lst: List[ResidualBlock] = []
        for _ in range(num_res_blocks):
            res_blocks_lst.append(ResidualBlock(m_dim))
        self.res_blocks = nn.Sequential(*res_blocks_lst)

    def forward(self, x: Tensor) -> Tensor:

        x = self.fc1(x)
        x = self.relu1(x)

        if self.num_res_blocks > 0:
            x = self.res_blocks(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_channels: int = 512) -> None:
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(n_channels, n_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels, n_channels, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        tt = self.block(x)
        x = x + tt
        return x


class SingleLayer_Classifier(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, dropout: float = 0.0) -> None:
        super(SingleLayer_Classifier, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(n_channels, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        x = self.fc(x)
        return x
