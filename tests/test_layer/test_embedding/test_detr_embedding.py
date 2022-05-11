import colossalai
import pytest
import torch

from titans.layer.embedding import PositionEmbeddingSine
from colossalai.global_variables import tensor_parallel_env as tp_env
from colossalai.testing import rerun_if_address_is_in_use
from tests.utils import run_with_parallel_config

BATCH_SIZE = 4
SEQ_LENGHT = 16
HIDDEN_SIZE = 32
VOCAB_SIZE = 50304


def run_detr_embed(data, hidden_size, vocab_size):

    #build model
    model = PositionEmbeddingSine().cuda()

    # forward
    out = model(data)

    # backward
    out.mean().backward()


def run_dist(rank, world_size, port, config):
    colossalai.launch(config=config, rank=rank, world_size=world_size, port=port, host='localhost')

    if tp_env.mode == 'sequence':
        tp_env.mode = None

    data = torch.rand(BATCH_SIZE, SEQ_LENGHT) * VOCAB_SIZE
    data = data.int().cuda()
    run_detr_embed(data, HIDDEN_SIZE, VOCAB_SIZE)


@pytest.mark.parametrize('parallel_config', [(4, '1d'), (4, '2d'), (4, '2.5d'), (8, '2.5d'), (8, '3d')])
@rerun_if_address_is_in_use()
def test_detr_embedding(parallel_config):
    run_with_parallel_config(*parallel_config, run_func=run_dist)
