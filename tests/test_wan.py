import pytest
import torch
from diffusers.models.transformers.transformer_wan_animate_2 import (
    WanAnimate2Transformer3DModel,
)
from diffusers.modular_pipelines.wan_animate_2.modular_pipeline import (
    WanAnimate2ModularPipeline,
)
from torch import nn

from haste.config import Haste
from haste.linear import CompressedLinear
from haste.wan import install


def transformer():

    return WanAnimate2Transformer3DModel(
        dim=32, ffn_dim=64, num_heads=4, num_layers=3, text_dim=32, freq_dim=16
    ).eval()


def test_public_wan_layout_and_restoration():
    model = transformer()
    originals = [block.ffn[0] for block in model.blocks]
    attention = [block.self_attn for block in model.blocks]
    x = torch.randn(1, 35, 32)
    with torch.inference_mode():
        baseline = model.blocks[0].ffn(x)
        with install(model, Haste(backend='reference', layers='early')) as patch:
            assert len(patch.entries) == 2
            assert isinstance(model.blocks[0].ffn[0], CompressedLinear)
            assert model.blocks[1].ffn[0] is originals[1]
            assert all(
                block.self_attn is attn for block, attn in zip(model.blocks, attention, strict=True)
            )
            model.blocks[0].ffn(x)
            patch.enable(False)
            torch.testing.assert_close(model.blocks[0].ffn(x), baseline, atol=0, rtol=0)
        assert all(
            block.ffn[0] is source for block, source in zip(model.blocks, originals, strict=True)
        )
        torch.testing.assert_close(model.blocks[0].ffn(x), baseline, atol=0, rtol=0)


def test_invalid_structure_fails_before_mutation():
    model = transformer()
    original = model.blocks[0].ffn[0]
    model.blocks[2].ffn[1] = nn.Identity()
    with pytest.raises(ValueError, match='Unsupported feedforward'):
        install(model, Haste(backend='reference'))
    assert model.blocks[0].ffn[0] is original


def test_context_restores_after_failure():
    model = transformer()
    original = model.blocks[0].ffn[0]
    with pytest.raises(RuntimeError, match='experiment failed'):
        with install(model, Haste(backend='reference')):
            raise RuntimeError('experiment failed')
    assert model.blocks[0].ffn[0] is original


def test_modular_pipeline_contract():

    pipe = WanAnimate2ModularPipeline()
    names = {param.name for param in pipe.blocks.inputs}
    assert {
        'image',
        'driving_video',
        'driving_video_fps',
        'prompt',
        'segment_frame_length',
        'num_inference_steps',
        'height',
        'width',
        'fps',
        'generator',
    } <= names
