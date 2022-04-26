from torch import Tensor
from colossalai.nn.layer.base_layer import ParallelLayer
from colossalai.utils import checkpoint
from colossalai.utils.activation_checkpoint import checkpoint

from titans.layer.mlp import MLP1D


class GenericTransformerLayer1D(ParallelLayer):

    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 act_func: str = 'gelu',
                 mlp_ratio: float = 4.0,
                 attention_dropout_prob: float = 0.,
                 hidden_dropout_prob: float = 0.,
                 dtype=None,
                 checkpoint: bool = False,
                 max_position_embeddings: int = 1024,
                 layer_norm_epsilon: float = 1e-5,
                 apply_post_layer_norm: bool = False,
                 attention=None,
                 layer_norm=None):
        super().__init__()
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.norm1 = layer_norm(hidden_size, eps=layer_norm_epsilon)
        self.apply_post_layer_norm = apply_post_layer_norm
        self.attention = attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout_prob=attention_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            dtype=dtype,
            max_position_embeddings=max_position_embeddings,
            checkpoint=False,
        )

        self.norm2 = layer_norm(hidden_size, eps=layer_norm_epsilon)
        self.mlp = GPTMLP1D(
            in_features=hidden_size,
            dropout_prob=hidden_dropout_prob,
            act_func=act_func,
            mlp_ratio=mlp_ratio,
            dtype=dtype,
            checkpoint=False,
        )

    def _forward(self, hidden_states, attention_mask) -> Tensor:
        if not self.apply_post_layer_norm:
            residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        if self.apply_post_layer_norm:
            residual = hidden_states
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = residual + attention_output

        if not self.apply_post_layer_norm:
            residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        if self.apply_post_layer_norm:
            residual = hidden_states
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        output = (hidden_states, attention_mask)
        return output

    def forward(self, hidden_states, attention_mask):
        if self.checkpoint:
            return checkpoint(self._forward, False, hidden_states, attention_mask)
        else:
            return self._forward(hidden_states, attention_mask)
