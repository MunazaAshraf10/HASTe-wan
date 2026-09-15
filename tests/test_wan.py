import pytest
import torch
from diffusers.models.transformers.transformer_wan_animate_2 import (
    WanAnimate2Attention,
    WanAnimate2AttnProcessor,
    WanAnimate2KVLayerCache,
    WanAnimate2Transformer3DModel,
    rope_params,
)
from diffusers.modular_pipelines.wan_animate_2.modular_pipeline import (
    WanAnimate2ModularPipeline,
)
from torch import nn

from haste.config import Haste
from haste.wan import Processor, install


def transformer():

    return WanAnimate2Transformer3DModel(
        dim=32, ffn_dim=64, num_heads=4, num_layers=3, text_dim=32, freq_dim=16
    ).eval()


def test_public_wan_layout_and_restoration():
    model = transformer()
    originals = [block.self_attn.processor for block in model.blocks]
    attention = [block.self_attn for block in model.blocks]
    x = torch.randn(1, 35, 32)
    with torch.inference_mode():
        baseline = model.blocks[0].ffn(x)
        with install(model, Haste(engine='reference', layers='early')) as patch:
            assert len(patch.entries) == 1
            assert isinstance(model.blocks[0].self_attn.processor, Processor)
            assert model.blocks[1].self_attn.processor is originals[1]
            assert all(
                block.self_attn is attn for block, attn in zip(model.blocks, attention, strict=True)
            )
            model.blocks[0].ffn(x)
            patch.enable(False)
            torch.testing.assert_close(model.blocks[0].ffn(x), baseline, atol=0, rtol=0)
        assert all(
            block.self_attn.processor is source
            for block, source in zip(model.blocks, originals, strict=True)
        )
        torch.testing.assert_close(model.blocks[0].ffn(x), baseline, atol=0, rtol=0)


def test_invalid_structure_fails_before_mutation():
    model = transformer()
    original = model.blocks[0].self_attn.processor
    model.blocks[2].self_attn = nn.Identity()
    with pytest.raises(ValueError, match='Unsupported self-attention'):
        install(model, Haste(engine='reference'))
    assert model.blocks[0].self_attn.processor is original


def test_context_restores_after_failure():
    model = transformer()
    original = model.blocks[0].self_attn.processor
    with pytest.raises(RuntimeError, match='experiment failed'):
        with install(model, Haste(engine='reference')):
            raise RuntimeError('experiment failed')
    assert model.blocks[0].self_attn.processor is original


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


def attention_case(device: str, dtype: torch.dtype):
    attn = WanAnimate2Attention(64, 4).to(device=device, dtype=dtype).eval()
    cache = WanAnimate2KVLayerCache()
    freqs = torch.cat([rope_params(32, 8), rope_params(32, 4), rope_params(32, 4)], 1).to(device)
    grid = torch.tensor([[3, 2, 2]])
    reference = torch.tensor([[2, 2, 2]])
    assert isinstance(attn.processor, WanAnimate2AttnProcessor)
    source = torch.compile(attn.processor, fullgraph=False) if device == 'cuda' else attn.processor
    with torch.inference_mode():
        source(
            attn,
            torch.randn(1, 8, 64, device=device, dtype=dtype),
            freqs,
            reference,
            cache,
            'extract',
        )
    mask = transformer().create_mask(2, 4, device)
    args = dict(
        rotary_emb=freqs,
        grid_sizes=grid,
        kv_cache=cache,
        kv_cache_mode='cached',
        reference_rotary_emb=freqs,
        reference_grid_sizes=reference,
        attention_mask=mask,
        origin_latent_frames=2,
        origin_latent_hw=4,
    )
    x = torch.randn(1, 12, 64, device=device, dtype=dtype)
    return attn, source, x, args


@pytest.mark.parametrize('backend', ['xattention', 'svg2'])
def test_real_processor_preserves_dense_and_reference_state(backend):
    attn, source, x, args = attention_case('cpu', torch.float32)
    cfg = Haste(engine='reference', backend=backend, threshold=1, queries=3, keys=4, iterations=1)
    processor = Processor(attn.processor, cfg, 0)
    weights = {name: value.clone() for name, value in attn.state_dict().items()}
    key, value = args['kv_cache'].get()
    with torch.inference_mode():
        baseline = source(attn, x, **args)
        actual = processor(attn, x, **args)
        torch.testing.assert_close(actual, baseline, rtol=1e-5, atol=1e-6)
        processor.target = (1, 1.0)
        processor.reset()
        torch.testing.assert_close(processor(attn, x, **args), baseline, rtol=0, atol=0)
    assert args['kv_cache'].get()[0] is key and args['kv_cache'].get()[1] is value
    for name, weight in attn.state_dict().items():
        torch.testing.assert_close(weight, weights[name], rtol=0, atol=0)


def test_branch_and_segment_cache_lifecycle():
    model = transformer()
    with install(model, Haste(engine='reference')) as patch:
        processor = patch.entries[0].processor
        cond = processor.state('cond')
        uncond = processor.state('uncond')
        assert cond is not uncond
        patch.context(model, (), dict(kv_cache_mode='cached', is_uncondtion=True))
        assert processor.branch == 'uncond'
        patch.context(model, (), dict(kv_cache_mode='extract'))
        assert not processor.states
        processor.state('cond')
        patch.enable(False)
        assert not processor.states and not processor.enabled


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('backend', ['xattention', 'svg2'])
def test_real_cuda_processor_dense_restoration(backend):
    attn, source, x, args = attention_case('cuda', torch.bfloat16)
    processor = Processor(
        attn.processor, Haste(backend=backend, threshold=1, queries=3, keys=4, iterations=1), 0
    )
    with torch.inference_mode():
        baseline = source(attn, x, **args)
        actual = processor(attn, x, **args)
        torch.testing.assert_close(actual, baseline, rtol=0.025, atol=0.025)
        processor.enabled = False
        torch.testing.assert_close(processor(attn, x, **args), baseline, rtol=0, atol=0)
