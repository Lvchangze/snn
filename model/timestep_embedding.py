import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.init as init

class TimestepEmbedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, time_step: int, padding_idx: Optional[int] = None, 
        max_norm: Optional[float] = None, norm_type: float = 2, scale_grad_by_freq: bool = False, sparse: bool = False, 
        _weight: Optional[Tensor] = None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = Parameter(torch.empty((num_embeddings, time_step, embedding_dim)), **factory_kwargs)
            

    def reset_parameters(self):
        init.normal_(self.weight)
        self._fill_padding_idx_with_zero()
        
    def _fill_padding_idx_with_zero(self):
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {time_step}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)

    def forward():
        pass