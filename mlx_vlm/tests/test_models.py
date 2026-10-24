"""JSON model contracts, checkpoint loading, sanitization, and document layout."""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import logging
import math
import struct
import textwrap
import unittest
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from operator import attrgetter
from pathlib import Path
from types import SimpleNamespace
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

from mlx_vlm import embedding_loader
from mlx_vlm.models.base import InputEmbeddingsFeatures, create_attention_mask
from mlx_vlm.models.cache import KVCache, RotatingKVCache, make_prompt_cache
from mlx_vlm.models.lfm2_encoder import Model as Lfm2Encoder
from mlx_vlm.utils import (
    _drop_modules_without_weights,
    _load_safetensors,
    get_model_and_args,
    get_model_path,
    load,
    load_config,
    load_model,
)

# Shared model contracts


def capture_positions(
    language_model, hidden_size, inputs, *, idx, offset, fa_idx=None, **kwargs
):
    class Recorder:
        embed_tokens = NS(as_linear=lambda x: x)

        def __call__(self, inputs, position_ids=None, **kwargs):
            self.positions = position_ids
            return mx.zeros((*inputs.shape, hidden_size))

    recorder = Recorder()
    if fa_idx is not None:
        recorder.fa_idx = fa_idx
    language_model.model = recorder
    language_model.lm_head = lambda x: x
    language_model(inputs, cache=[NS(_idx=idx, offset=mx.array(offset))], **kwargs)
    return recorder.positions


class ModelChecks:
    """Reusable assertions; each JSON case constructs fresh configs and models."""

    def forward_cache(self, model, vocab_size, *, chunk_sizes=()):
        model.eval()
        mx.eval(model.parameters())
        ids = mx.array([[1, 5, 9, 13, 2, 7, 11, 3]])
        assert model(ids).logits.shape == (1, 8, vocab_size)
        if chunk_sizes:
            for dtype in (mx.float32, mx.float16):
                model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
                self.prefill_parity(model, ids, {}, chunk_sizes)
            return
        cache = model.language_model.make_cache()
        model(ids[:, :-1], cache=cache)
        assert model(ids[:, -1:], cache=cache).logits.shape == (1, 1, vocab_size)

    def masked_lm(self, model, config):
        mask = mx.array([[1, 1, 1, 0]])
        first = model(mx.array([[1, 2, 3, 0]]), attention_mask=mask).logits
        second = model(mx.array([[1, 2, 3, 9]]), attention_mask=mask).logits
        assert first.shape == (1, 4, config.vocab_size)
        assert mx.allclose(first[:, :3], second[:, :3])

    def token_embeddings(self, model, config):
        output = model(mx.array([[1, 2, 3]]))
        assert output.last_hidden_state.shape == (1, 3, config.hidden_size)
        assert output.text_embeds.shape == (1, 3, config.embedding_dim)
        assert mx.allclose(
            mx.linalg.norm(output.text_embeds, axis=-1), mx.array(1.0), atol=1e-5
        )

    def sentence_embeddings(self, model, config):
        mask = mx.array([[1, 1, 1, 0]])
        output = model(mx.array([[1, 2, 3, 0]]), attention_mask=mask)
        changed_padding = model(mx.array([[1, 2, 3, 9]]), attention_mask=mask)
        width = config.text_config.embedding_dim
        assert output.last_hidden_state.shape == (1, 4, width)
        assert output.text_embeds.shape == (1, width)
        assert output.text_embeds.dtype == mx.float32
        self.assert_close(output.text_embeds, changed_padding.text_embeds)
        self.assert_close(mx.linalg.norm(output.text_embeds, axis=-1), mx.ones((1,)))

    def assert_close(self, actual, expected, *, logits=False):
        assert actual.shape == expected.shape
        assert mx.all(mx.isfinite(actual)).item()
        tolerance = 5e-3 if actual.dtype == mx.float16 else 1e-4
        # One-token attention uses a different Metal kernel from full prefill.
        if logits:
            tolerance = max(tolerance, 2e-3)
        np.testing.assert_allclose(
            np.array(actual.astype(mx.float32)),
            np.array(expected.astype(mx.float32)),
            atol=tolerance,
            rtol=tolerance,
        )

    def prefill_parity(self, model, ids, media, chunk_sizes):
        """Compare full, cached and chunked prefill, then decode one new token."""
        language = model.language_model
        features = model.get_input_embeddings(ids, **media).to_dict()
        embeds = features.pop("inputs_embeds")
        reference = model(ids, cache=make_prompt_cache(language), **media).logits
        if ids.shape[0] > 1:
            solo = []
            for row in range(ids.shape[0]):
                row_media = {
                    key: value[row * count : (row + 1) * count]
                    for key, value in media.items()
                    for count in [value.shape[0] // ids.shape[0]]
                }
                solo.append(
                    model(
                        ids[row : row + 1],
                        cache=make_prompt_cache(language),
                        **row_media,
                    ).logits
                )
            self.assert_close(reference, mx.concatenate(solo), logits=True)
        token = mx.full((ids.shape[0], 1), 7, mx.int32)
        extended = model(
            mx.concatenate([ids, token], axis=1),
            cache=make_prompt_cache(language),
            **media,
        ).logits[:, -1:]
        for size in (ids.shape[1] - 1, *chunk_sizes):
            cache, chunks = make_prompt_cache(language), []
            for start in range(0, ids.shape[1], size):
                stop = min(start + size, ids.shape[1])
                chunks.append(
                    language(
                        ids[:, start:stop],
                        inputs_embeds=embeds[:, start:stop],
                        cache=cache,
                        **features,
                    ).logits
                )
            self.assert_close(mx.concatenate(chunks, axis=1), reference, logits=True)
            decode = {k: v for k, v in features.items() if k != "position_ids"}
            self.assert_close(
                language(token, cache=cache, **decode).logits, extended, logits=True
            )

    def multimodal(
        self,
        model,
        config,
        *,
        image_grid,
        video_grid,
        chunk_sizes,
        batch_sizes=(1, 2),
        layouts=(("image",), ("video",), ("image", "video"), ("video", "image")),
    ):
        """Image/video fusion and DeepStack contracts across compatible models."""
        core = getattr(model, "thinker", model)
        config = getattr(config, "thinker_config", config)
        vision, vc = core.vision_tower, config.vision_config
        layers, width = len(vc.deepstack_visual_indexes), vc.out_hidden_size
        assert layers > 0, "The multimodal case must enable DeepStack layers"
        patch_width = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2
        tokens = {"image": config.image_token_id, "video": config.video_token_id}
        grids = {"image": image_grid, "video": video_grid}
        model.eval()
        for dtype in (mx.float32, mx.float16):
            model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
            for layout in (tuple(layout) for layout in layouts):
                for batch in batch_sizes:
                    rows, media, encoded = [], {}, {}
                    for row in range(batch):
                        order = layout if row == 0 else layout[::-1]
                        ids = [1, 2]
                        for kind in order:
                            count = math.prod(grids[kind]) // vc.spatial_merge_size**2
                            ids += [config.vision_start_token_id] + [
                                tokens[kind]
                            ] * count
                            ids += [config.vision_end_token_id]
                        rows.append(ids + [3, 4, 5])
                    ids = mx.array(rows, mx.int32)
                    for kind in layout:
                        pixels = mx.random.normal(
                            (batch * math.prod(grids[kind]), patch_width)
                        ).astype(dtype)
                        grid = mx.array([grids[kind]] * batch)
                        media[
                            "pixel_values" if kind == "image" else "pixel_values_videos"
                        ] = pixels
                        media[kind + "_grid_thw"] = grid
                        encoded[kind] = vision(pixels, grid)
                    features = model.get_input_embeddings(ids, **media)
                    assert isinstance(features, InputEmbeddingsFeatures)
                    mask = (ids == tokens["image"]) | (ids == tokens["video"])
                    assert mx.array_equal(features.visual_pos_masks, mask).item()
                    residuals = features.deepstack_visual_embeds
                    if isinstance(residuals, (list, tuple)):
                        residuals = mx.stack(residuals)
                    assert residuals is not None and residuals.dtype == dtype
                    expected = mx.zeros((*ids.shape, layers, width), dtype)
                    expected_embeds = model.language_model.model.embed_tokens(ids)
                    for kind in layout:
                        positions = mx.array(
                            [
                                i
                                for i, t in enumerate(ids.flatten().tolist())
                                if t == tokens[kind]
                            ],
                            mx.uint32,
                        )
                        embeddings, layer_features = encoded[kind]
                        assert len(layer_features) == layers
                        flat = expected.reshape(-1, layers, width)
                        flat[positions] = mx.stack(list(layer_features), axis=1)
                        expected = flat.reshape(expected.shape)
                        flat = expected_embeds.reshape(-1, width)
                        flat[positions] = embeddings
                        expected_embeds = flat.reshape(expected_embeds.shape)
                    self.assert_close(features.inputs_embeds, expected_embeds)
                    if residuals.ndim == 4:
                        self.assert_close(residuals, expected)
                    else:
                        positions = mx.array(
                            [i for i, v in enumerate(mask.flatten().tolist()) if v],
                            mx.uint32,
                        )
                        self.assert_close(
                            residuals,
                            expected.reshape(-1, layers, width)[positions].transpose(
                                1, 0, 2
                            ),
                        )
                    for layer in range(layers):
                        value = (
                            residuals[:, :, layer]
                            if residuals.ndim == 4
                            else residuals[layer]
                        )
                        injected = model.language_model.model._deepstack_process(
                            mx.zeros_like(features.inputs_embeds), mask, value
                        )
                        self.assert_close(injected, expected[:, :, layer])
                    self.prefill_parity(model, ids, media, chunk_sizes)

    def input_embeddings(self, model, model_name):
        result = model.get_input_embeddings(input_ids=mx.array([[1, 2, 3, 4, 5]]))
        assert isinstance(result, InputEmbeddingsFeatures), model_name
        assert result.inputs_embeds is not None

    def request_positions(self, model):
        language = model.language_model
        positions, deltas = mx.array([[[9, 9, 9]]], mx.int32), mx.array(
            [[99]], mx.int32
        )
        language._position_ids, language._rope_deltas = positions, deltas
        result = model.get_input_embeddings(input_ids=mx.array([[1, 2, 3]], mx.int32))
        assert result.position_ids is not None and result.rope_deltas is not None
        assert result.position_ids.shape == (1, 3)
        assert result.position_ids.tolist() == [[0, 1, 2]]
        assert result.rope_deltas.tolist() == [[0]]
        assert mx.array_equal(language._position_ids, positions).item()
        assert mx.array_equal(language._rope_deltas, deltas).item()

    def chunked_positions(self, model):
        language = model.language_model
        width = language.args.hidden_size
        positions = mx.arange(15, dtype=mx.int32).reshape(3, 1, 5)
        actual = capture_positions(
            language,
            width,
            mx.array([[7, 8]], dtype=mx.int32),
            idx=2,
            offset=2,
            fa_idx=0,
            inputs_embeds=mx.zeros((1, 2, width), dtype=mx.float32),
            position_ids=positions,
        )
        assert actual.shape == (3, 1, 2)
        assert actual.tolist() == positions[:, :, 2:4].tolist()

    def language(self, model, config, *, num_layers=None):
        if num_layers is None:
            num_layers = config.num_hidden_layers
        assert model.model_type == config.model_type
        assert len(model.layers) == num_layers
        for dtype in (mx.float32, mx.float16):
            model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
            logits = model(mx.array([[0, 1]])).logits
            assert logits.shape == (1, 2, config.vocab_size) and logits.dtype == dtype
            logits = model(
                mx.argmax(logits[0, -1:, :], keepdims=True), cache=None
            ).logits
            assert logits.shape == (1, 1, config.vocab_size) and logits.dtype == dtype

    def language_hidden_states(self, model):
        ids = mx.array([[1, 2, 3]])
        embeds = model.get_input_embeddings(ids)
        captured = []

        def record_input(forward):
            def call(layer, hidden, *args, **kwargs):
                captured.append(hidden)
                return forward(layer, hidden, *args, **kwargs)

            return call

        reference_cache = model.make_cache()
        with ExitStack() as stack:
            for layer_type in {type(layer) for layer in model.layers}:
                stack.enter_context(
                    patch.object(
                        layer_type, "__call__", record_input(layer_type.__call__)
                    )
                )
            reference = model(ids, cache=reference_cache)
        assert reference.hidden_states is None
        next_ids = mx.array([[4]])
        next_logits = model(next_ids, cache=reference_cache).logits
        for inputs in (
            {"inputs": ids},
            {"input_ids": ids},
            {"inputs_embeds": embeds},
        ):
            cache = model.make_cache()
            output = model(**inputs, cache=cache, output_hidden_states=True)
            assert len(output.hidden_states) == len(model.layers) + 1
            for actual, expected in zip(output.hidden_states[:-1], captured):
                self.assert_close(actual, expected)
            self.assert_close(output.logits, reference.logits)
            self.assert_close(model.lm_head(output.hidden_states[-1]), output.logits)
            self.assert_close(model(next_ids, cache=cache).logits, next_logits)

    def projector(
        self, model, input_width, output_width, *, grid_hw=None, downsample_ratio=1
    ):
        batch = math.prod(grid_hw) if grid_hw else 1
        tokens = (
            math.prod(math.ceil(n / downsample_ratio) for n in grid_hw)
            if grid_hw
            else 1
        )
        kwargs = dict(zip(("n_h", "n_w"), grid_hw)) if grid_hw else {}
        for dtype in (mx.float32, mx.float16):
            model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
            inputs = mx.random.uniform(shape=(batch, input_width), dtype=dtype)
            output = model(mx.array(inputs), **kwargs)
            assert output.shape == (tokens, output_width) and output.dtype == dtype

    def vision(
        self,
        model,
        model_type,
        width,
        channels,
        image_size,
        vision_feature_layer=-2,
        channel_first=False,
        **kwargs,
    ):
        if model_type == "llama4_vision_model":
            width = kwargs.pop("projector_output_dim", width)
        batch = kwargs.pop("batch_size", 1)
        flat = (
            "qwen2_5_vl qwen3_5 qwen3_5_moe qwen4_exp "
            "glm4v_moe glm4v hunyuan_vl siglip2_vision_model mimovl"
        ).split()
        shape = (
            image_size
            if len(image_size) > 2 or model_type in flat
            else (
                (batch, channels, *image_size)
                if channel_first
                else (batch, *image_size, channels)
            )
        )
        parameters = inspect.signature(model.__call__).parameters
        for dtype in (mx.float32, mx.float16):
            model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
            if model_type is not None:
                assert model.model_type == model_type
            inputs = mx.random.uniform(shape=shape)
            if "image_masks" in parameters:
                inputs = inputs.transpose(0, 3, 1, 2)
                kwargs["image_masks"] = mx.ones((batch, channels, image_size[0]))
            options = (
                {"output_hidden_states": True}
                if "output_hidden_states" in parameters
                else {}
            )
            hidden = model(inputs.astype(dtype), **options, **kwargs)
            if vision_feature_layer is not None:
                hidden = hidden[vision_feature_layer]
            assert hidden.shape[1 if channel_first else -1] == width
            assert hidden.dtype == dtype

    def _assert_audio_features(self, features, shape, dtype):
        mx.eval(features)
        assert features.shape == shape
        assert features.dtype == dtype
        assert mx.all(mx.isfinite(features)).item()

    def audio(self, model, config, model_name, *, frames=32, lengths=None):
        if model_name not in {"inkling", "gemma3n", "gemma4", "gemma4_unified"}:
            raise ValueError(f"Unsupported audio model: {model_name}")
        audio_config = config.audio_config
        text_width = config.text_config.hidden_size
        lengths = [frames, frames // 2] if lengths is None else lengths
        assert lengths and all((0 <= n <= frames for n in lengths))
        batch = len(lengths)
        valid_mask = mx.arange(frames)[None, :] < mx.array(lengths)[:, None]
        components = [
            getattr(model, name, None) for name in ("audio_tower", "embed_audio")
        ]
        for dtype in (mx.float32, mx.float16):
            for component in components:
                if component is not None:
                    component.eval()
                    component.update(
                        tree_map(lambda p: p.astype(dtype), component.parameters())
                    )

            if model_name == "inkling":
                bins = audio_config.n_mel_bins
                inputs = mx.arange(batch * frames * bins, dtype=mx.int32)
                inputs = (
                    inputs.reshape(batch, frames, bins) % audio_config.mel_vocab_size
                )
                features = model.audio_tower(inputs)
                shape, output_dtype = (batch, frames, text_width), dtype
            elif model_name == "gemma4_unified":
                inputs = mx.random.normal(
                    (batch, frames, audio_config.output_proj_dims)
                ).astype(dtype)
                projected = model.get_audio_features(inputs)
                self._assert_audio_features(
                    projected, (batch * frames, text_width), dtype
                )
                features = model.get_audio_features(inputs, valid_mask)
                shape, output_dtype = (sum(lengths), text_width), dtype
            else:
                is_gemma3n = model_name == "gemma3n"
                # Gemma 4's subsampling projection requires 128 input mel bins.
                bins = audio_config.input_feat_size if is_gemma3n else 128
                inputs = mx.random.normal((batch, frames, bins)).astype(dtype)
                encoded, mask = model.audio_tower(inputs, ~valid_mask)
                stride = (
                    math.prod(s[0] for s in audio_config.sscp_conv_stride_size)
                    * max(1, audio_config.conf_reduction_factor)
                    if is_gemma3n
                    else 4
                )
                steps = (frames + stride - 1) // stride
                width = (
                    getattr(audio_config, "output_proj_dims", None)
                    or audio_config.hidden_size
                )
                # Both Gemma encoders currently promote float16 inputs to float32.
                output_dtype = mx.float32
                self._assert_audio_features(
                    encoded, (batch, steps, width), output_dtype
                )
                assert mask.shape == (batch, steps)
                assert mask.dtype == mx.bool_
                assert mx.array_equal(mask, ~valid_mask[:, ::stride]).item()
                assert mx.all(mx.where(mask[..., None], encoded == 0, True)).item()
                features = (
                    model.embed_audio(inputs_embeds=encoded)
                    if is_gemma3n
                    else model.embed_audio(encoded)
                )
                shape = (batch, steps, text_width)
            self._assert_audio_features(features, shape, output_dtype)

    def mrope_cache_index(self, language_model, hidden_size):
        """Use the Python cache index (10), avoiding a GPU sync on offset (3)."""
        language_model._rope_deltas = mx.array([[0]])
        language_model._position_ids = None
        positions = capture_positions(
            language_model, hidden_size, mx.array([[5]]), idx=10, offset=3
        )
        assert positions is not None
        assert tuple(positions.shape) in {(1, 1), (3, 1, 1)}
        assert positions.reshape(-1)[0].item() == 10

    def mrope_deltas(self, language_model, hidden_size):
        """The request's delta (5) must override stale model state (99)."""
        language_model._rope_deltas = mx.array([[99]])
        language_model._position_ids = None
        positions = capture_positions(
            language_model,
            hidden_size,
            mx.array([[7]]),
            idx=10,
            offset=3,
            fa_idx=0,
            rope_deltas=mx.array([[5]]),
        )
        assert positions is not None
        assert tuple(positions.shape) == (3, 1, 1)
        assert positions[0, 0, 0].item() == 15


CONFIG_TYPES = {
    "text_config": "TextConfig",
    "vision_config": "VisionConfig",
    "projector_config": "ProjectorConfig",
    "perceiver_config": "PerceiverConfig",
    "audio_config": "AudioConfig",
    "thinker_config": "ThinkerConfig",
    "talker_config": "TalkerConfig",
    "code_predictor_config": "CodePredictorConfig",
    "code2wav_config": "Code2WavConfig",
    "vit_config": "config.VitConfig",
    "adapter_config": "config.AdapterConfig",
}
DATA = json.loads(
    Path(__file__).with_name("model_cases.json").read_text(encoding="utf-8")
)
if DATA["version"] != 2:
    raise ValueError(f"Unsupported model_cases.json version: {DATA['version']}")

TINY_DEFAULTS = DATA["tiny_defaults"]
TINY_MODELS = DATA["shared_configs"]


def model_case(name):
    """Return fresh case data for shared model and checkpoint tests."""
    return copy.deepcopy(next(case for case in DATA["cases"] if case["id"] == name))


def build_config(module, values, config_type="ModelConfig"):
    """Construct the model family's config classes from ordinary nested data."""
    fields = copy.deepcopy(values)
    if (
        module.__name__
        in ("mlx_vlm.models.minicpmv4_7", "mlx_vlm.models.nemotron_h_nano_omni")
        and config_type == "ModelConfig"
    ):
        # Preserve checkpoint aliases and per-layer text-config dispatch.
        return module.ModelConfig.from_dict(fields)
    for name, value in fields.items():
        if name in CONFIG_TYPES and isinstance(value, dict):
            fields[name] = build_config(module, value, CONFIG_TYPES[name])
    return attrgetter(config_type)(module)(**fields)


def first_attribute(obj, *names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"{type(obj).__name__} has none of {names}")


def check_arguments(kind, case, model, config):
    """Keep shared component selection and dimension wiring in Python."""
    name = case["module"]
    # Phi3-V keeps language dimensions on the outer config.
    core_config = getattr(config, "thinker_config", config)
    text = (
        config if name == "phi3_v" else getattr(core_config, "text_config", core_config)
    )
    if kind == "forward_cache":
        return (model, text.vocab_size), case.get("forward_cache", {})
    if kind in {"masked_lm", "token_embeddings", "sentence_embeddings"}:
        return (model, config), {}
    if kind == "multimodal":
        return (model, config), case["multimodal"]
    if kind == "input_embeddings":
        return (model, name), {}
    if kind == "audio":
        return (model, config, name), case.get("audio", {})
    if kind in {"request_positions", "chunked_positions"}:
        return (model,), {}
    if kind == "language_hidden_states":
        return (model.language_model,), {}
    if kind in {"language", "mrope_cache_index", "mrope_deltas"}:
        if kind == "language":
            options = (
                {}
                if hasattr(text, "num_hidden_layers")
                else {"num_layers": text.n_layers}
            )
            return (model.language_model, text), options
        return (model.language_model, text.hidden_size), {}
    if kind == "projector":
        projector = attrgetter(case.get("projector_path", "multi_modal_projector"))(
            model
        )
        if name in ("deepseek_v4", "deepseek_v41"):
            return (
                projector,
                first_attribute(config, "vision_dim", "vision_hidden_size"),
                config.hidden_size,
            ), {
                "grid_hw": case["vision"]["grid_hw"],
                "downsample_ratio": config.vision_downsample_ratio,
            }
        return (projector, config.vision_config.hidden_size, text.hidden_size), {}
    if kind == "vision":
        vision = attrgetter(case.get("vision_path", "vision_tower"))(model)
        options = case.get("vision", {})
        if name in ("deepseek_v4", "deepseek_v41"):
            return (
                vision,
                None,
                first_attribute(config, "vision_dim", "vision_hidden_size"),
                3,
                tuple(options["input_shape"]),
            ), {
                "vision_feature_layer": options["feature_layer"],
                "n_h": options["grid_hw"][0],
                "n_w": options["grid_hw"][1],
            }
        vc = config.vision_config
        image_size = options.get("input_shape")
        if image_size is None:
            image_size = (vc.image_size, vc.image_size)
        width = first_attribute(
            vc, "out_hidden_size", "hidden_size", "d_model", "width", "text_hidden_size"
        )
        # Molmo's hidden_size describes its projector; use d_model for vision.
        if name == "molmo":
            width = vc.d_model
        channels = first_attribute(vc, "num_channels", "in_channels")
        kwargs = {}
        if "feature_layer" in options:
            kwargs["vision_feature_layer"] = options["feature_layer"]
        if "channel_first" in options:
            kwargs["channel_first"] = options["channel_first"]
        if "grid_thw" in options:
            kwargs["grid_thw"] = mx.array(
                options["grid_thw"],
                dtype=getattr(mx, options.get("grid_dtype", "int64")),
            )
        if vc.model_type == "llama4_vision_model":
            kwargs["projector_output_dim"] = vc.projector_output_dim
        # MiniMax's vision wrapper does not expose model_type.
        model_type = None if name == "minimax_m3_vl" else vc.model_type
        return (vision, model_type, width, channels, tuple(image_size)), kwargs
    raise ValueError(f"Unknown model check: {kind}")


@pytest.mark.parametrize("case", DATA["cases"], ids=lambda case: case["id"])
def test_model_contract(case):
    if "multimodal" in case["checks"]:
        mx.random.seed(17)
    module = importlib.import_module("mlx_vlm.models." + case["module"])
    config = build_config(module, case["config"])
    model = module.Model(config)
    checks = ModelChecks()
    for kind in case["checks"]:
        args, kwargs = check_arguments(kind, case, model, config)
        getattr(checks, kind)(*args, **kwargs)


@pytest.fixture
def nemotron_case(request):
    case = model_case(getattr(request, "param", "nemotron_super_vl"))
    module = importlib.import_module("mlx_vlm.models." + case["module"])
    return module, case["config"]


@pytest.mark.parametrize("nemotron_case", ["nemotron_bolt"], indirect=True)
def test_nemotron_puzzle_layer_configs(nemotron_case):
    arch, config = nemotron_case
    # The block list takes precedence, as in NVIDIA's Puzzle config.
    config["llm_config"]["layers_block_type"] = ["attention"]
    _, family = get_model_and_args({"model_type": "nemotron_h_puzzle"})
    assert family == "nemotron_h"
    model = arch.Model(build_config(arch, config))
    assert [layer.block_type for layer in model.language_model.layers] == [
        "M",
        "E",
        "*",
        "E",
    ]
    assert model.config.text_config.time_step_limit == (0, float("inf"))
    for idx, experts, width, topk in [(1, 2, 32, 1), (3, 3, 64, 2)]:
        moe = model.language_model.layers[idx].mixer
        assert moe.switch_mlp.fc1.weight.shape == (experts, width, 32)
        assert moe.switch_mlp.fc2.weight.shape == (experts, 32, width)
        assert moe.gate.weight.shape == (experts, 32)
        assert moe.gate.top_k == topk


@pytest.mark.parametrize("vision_norm,mtp_depth", [(True, 0), (False, 1), (None, 1)])
def test_nemotron_explicit_vision_norm_overrides_mtp(
    nemotron_case, vision_norm, mtp_depth
):
    arch, config = nemotron_case
    config["vision_final_layernorm"] = vision_norm
    config["llm_config"]["num_nextn_predict_layers"] = mtp_depth
    model = arch.Model(build_config(arch, config))
    expected = vision_norm if vision_norm is not None else mtp_depth > 0
    assert isinstance(model.vision_final_layernorm, nn.LayerNorm) == expected
    assert model.config.to_dict().get("vision_final_layernorm") == vision_norm


@pytest.mark.parametrize(
    "nemotron_case", ["nemotron_super_vl", "nemotron_bolt"], indirect=True
)
@pytest.mark.parametrize("text_key", ["llm_config", "text_config"])
def test_nemotron_checkpoint_loading(tmp_path, nemotron_case, text_key):
    arch, config = nemotron_case
    config[text_key] = config.pop("llm_config")
    _, family = get_model_and_args(config)
    assert family == arch.__name__.rsplit(".", 1)[-1]
    model = arch.Model(build_config(arch, config))
    expected = dict(tree_flatten(model.parameters()))
    weights = {}
    for key, value in expected.items():
        if ".switch_mlp." in key:
            prefix, projection = key.split(".switch_mlp.")
            projection = projection.replace("fc1", "up_proj").replace(
                "fc2", "down_proj"
            )
            for expert, weight in enumerate(value):
                weights[f"{prefix}.experts.{expert}.{projection}"] = weight
        else:
            if "conv1d.weight" in key:
                value = value.moveaxis(1, 2)
            key = key.replace("mlp1.layers.", "mlp1.")
            if key.startswith("vision_final_layernorm."):
                key = "vision_projector." + key
            weights[key] = value
    weights["vision_model.summary_idxs"] = mx.array([0, 1])
    weights["language_model.mtp.unused.weight"] = mx.ones((2, 2))
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    loaded = load_model(tmp_path)
    assert asdict(loaded.config) == asdict(model.config)
    actual = dict(tree_flatten(loaded.parameters()))
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert mx.array_equal(actual[key], value), key


@pytest.mark.parametrize(
    "rate,expected",
    [
        (0.0, [True] * 6),
        (0.5, [True, True, False, True, False, False]),
        (1.0, [True, True, False, False, False, False]),
    ],
)
def test_nemotron_video_retention_order_and_first_tubelet(
    nemotron_case, rate, expected
):
    arch, _ = nemotron_case
    vision = inspect.getmodule(arch.Model)
    # The second token changes most in frame 2; equal scores preserve order.
    features = mx.array(
        [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]]]
    )
    assert vision.compute_retention_mask(features, rate).tolist() == expected
    assert vision.compute_retention_mask(mx.zeros((1, 2, 4)), rate).tolist() == [
        True,
        True,
    ]


def test_nemotron_vision_normalization_and_video_pruning(nemotron_case):
    arch, config = nemotron_case
    vision = inspect.getmodule(arch.Model)
    model = arch.Model(build_config(arch, config))
    features = mx.random.normal((1, 4, 16))
    # The projector must normalize BEFORE spatially merging the four patches.
    expected = model.mlp1(model.vision_final_layernorm(features).reshape(1, 1, 64))
    np.testing.assert_allclose(
        model._project_vision_features(features, 2, 2), expected, rtol=1e-5, atol=1e-5
    )
    # Five rectangular frames pad to three tubelets, two tokens per tubelet.
    pixels = mx.random.normal((5, 3, 32, 64))
    ids = mx.array([[1, 18, 18, 2, 18, 18, 3, 18, 18, 4]])
    full = model.extract_video_feature(pixels)
    assert full.shape == (3, 2, 32)
    keep = np.flatnonzero(vision.compute_retention_mask(full, 0.7))
    actual = model.get_input_embeddings(ids, pixel_values_videos=pixels).inputs_embeds
    assert actual.shape == (1, 6, 32)
    unpruned = model.get_input_embeddings(
        ids, pixel_values_videos=pixels, video_pruning_rate=0
    ).inputs_embeds
    assert unpruned.shape == (1, 10, 32)
    assert model.video_pruning_rate == 0.7
    with pytest.raises(ValueError, match="video_pruning_rate"):
        model.get_input_embeddings(
            ids, pixel_values_videos=pixels, video_pruning_rate=-1
        )
    np.testing.assert_allclose(actual[0, 1:3], full.reshape(-1, 32)[mx.array(keep)])
    expected_text = model.language_model.get_input_embeddings(mx.array([[1, 2, 3, 4]]))
    np.testing.assert_allclose(actual[:, [0, 3, 4, 5]], expected_text)
    images = [mx.zeros((3, 32, 32)), mx.zeros((3, 32, 64))]
    assert model.get_input_embeddings(
        mx.array([[1, 18, 18, 18, 2]]), images
    ).inputs_embeds.shape == (1, 5, 32)


def test_lfm2_encoder_sanitize_and_dispatch():
    case = next(case for case in DATA["cases"] if case["module"] == "lfm2_encoder")
    module = importlib.import_module("mlx_vlm.models.lfm2_encoder")
    model = module.Model(build_config(module, case["config"]))
    weights = {
        "lfm2.embed_tokens.weight": mx.zeros((32, 16)),
        "lfm2.layers.0.conv.conv.weight": mx.zeros((16, 1, 3)),
        "lm_head.weight": mx.zeros((32, 16)),
    }
    sanitized = model.sanitize(weights)
    assert "model.embed_tokens.weight" in sanitized
    assert sanitized["model.layers.0.conv.conv.weight"].shape == (16, 3, 1)
    assert "lm_head.weight" not in sanitized

    resolved, model_type = get_model_and_args(
        {
            "model_type": "lfm2",
            "architectures": ["Lfm2BidirectionalForMaskedLM"],
        }
    )
    assert resolved.Model is Lfm2Encoder
    assert model_type == "lfm2_encoder"


def test_lfm2_colbert_sanitize_and_loader(tmp_path, monkeypatch):
    case = next(case for case in DATA["cases"] if case["module"] == "lfm2_colbert")
    module = importlib.import_module("mlx_vlm.models.lfm2_colbert")
    model = module.Model(build_config(module, case["config"]))
    weights = {
        "embed_tokens.weight": mx.zeros((32, 16)),
        "layers.0.conv.conv.weight": mx.zeros((16, 1, 3)),
        "1_Dense.linear.weight": mx.zeros((8, 16)),
    }
    sanitized = model.sanitize(weights)
    assert "model.embed_tokens.weight" in sanitized
    assert sanitized["model.layers.0.conv.conv.weight"].shape == (16, 3, 1)
    assert "projection.weight" in sanitized

    dense_dir = tmp_path / "1_Dense"
    dense_dir.mkdir()
    (dense_dir / "config.json").write_text(json.dumps({"out_features": 128}))
    captured = {}

    def fake_load(model_path, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(embedding_loader, "load_encoder_model", fake_load)
    embedding_loader.load_embedding_model(tmp_path)
    assert captured["model_remapping"]["lfm2"] == "lfm2_colbert"
    assert captured["config_overrides"]["embedding_dim"] == 128


@pytest.fixture
def embedding_gemma2_case():
    case = next(c for c in DATA["cases"] if c["module"] == "embedding_gemma2")
    module = importlib.import_module("mlx_vlm.models." + case["module"])
    return module, build_config(module, case["config"])


@pytest.mark.parametrize("left_padding", [False, True])
def test_embedding_gemma2_padded_batch_matches_individual(
    embedding_gemma2_case, left_padding
):
    module, config = embedding_gemma2_case
    mx.random.seed(42)
    model = module.Model(config)
    ids = [1, 2, 3]
    solo = model(mx.array([ids])).text_embeds
    padded = [0, 0, *ids] if left_padding else [*ids, 0, 0]
    valid = [0, 0, 1, 1, 1] if left_padding else [1, 1, 1, 0, 0]
    output = model(
        mx.array([padded, [4, 5, 6, 7, 8]]),
        attention_mask=mx.array([valid, [1, 1, 1, 1, 1]]),
    )
    np.testing.assert_allclose(output.text_embeds[:1], solo, atol=1e-5, rtol=1e-5)


def test_embedding_gemma2_attention_is_bidirectional(embedding_gemma2_case):
    module, config = embedding_gemma2_case
    model = module.Model(config)
    first = model(mx.array([[1, 2, 3]])).last_hidden_state
    second = model(mx.array([[1, 2, 4]])).last_hidden_state
    assert not mx.allclose(first[:, 0], second[:, 0])


def test_embedding_gemma2_sliding_attention_includes_window_boundary(
    embedding_gemma2_case,
):
    module, config = embedding_gemma2_case
    mx.random.seed(42)
    model = module.Model(config)
    # Isolate the first (sliding) layer so the full layer cannot mix distant tokens.
    model.language_model.layers = model.language_model.layers[:1]
    baseline = model(mx.array([[1, 2, 3, 4, 5]])).last_hidden_state[:, 0]
    boundary = model(mx.array([[1, 2, 3, 6, 5]])).last_hidden_state[:, 0]
    outside = model(mx.array([[1, 2, 3, 4, 6]])).last_hidden_state[:, 0]
    assert not mx.allclose(baseline, boundary)
    np.testing.assert_allclose(baseline, outside, atol=1e-6)


def test_embedding_gemma2_explicit_position_ids(embedding_gemma2_case):
    module, config = embedding_gemma2_case
    mx.random.seed(42)
    model = module.Model(config)
    ids = mx.array([[1, 2, 3]])
    default = model(ids).last_hidden_state
    explicit = model(ids, position_ids=mx.array([[0, 1, 2]])).last_hidden_state
    spaced = model(ids, position_ids=mx.array([[0, 2, 4]])).last_hidden_state
    np.testing.assert_array_equal(default, explicit)
    assert not mx.allclose(default, spaced)


def test_embedding_gemma2_full_attention_config_defaults_and_index_normalization(
    embedding_gemma2_case,
):
    module, config = embedding_gemma2_case
    defaults = module.TextConfig(
        num_hidden_layers=7,
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        hidden_size_per_layer_input=8,
    )
    model = module.Model(module.ModelConfig(text_config=defaults))
    assert defaults.layer_types[-1] == "full_attention"
    assert model.layers[5].self_attn.head_dim == 512
    assert model.layers[5].self_attn.num_kv_heads == 1
    config.text_config.per_layer_config = {1: {"head_dim": 16}}
    config.text_config.__post_init__()
    assert module.Model(config).layers[1].self_attn.head_dim == 16


def test_embedding_gemma2_media_scatter_preserves_batch_order_and_validates_counts(
    embedding_gemma2_case,
):
    module, _ = embedding_gemma2_case
    ids = mx.array([[1, 60, 60], [60, 2, 0]])
    embeddings = mx.zeros((2, 3, 4))
    features = mx.arange(12).reshape(3, 4)
    output = module.Model._scatter(embeddings, ids, 60, features)
    np.testing.assert_array_equal(output[0, 1:], features[:2])
    np.testing.assert_array_equal(output[1, 0], features[2])
    np.testing.assert_array_equal(output[1, 1:], mx.zeros((2, 4)))
    with pytest.raises(ValueError, match="token count"):
        module.Model._scatter(embeddings, ids, 60, features[:2])


@pytest.mark.parametrize("modality", ["image", "video", "audio"])
def test_embedding_gemma2_disabled_towers_reject_media(embedding_gemma2_case, modality):
    module, config = embedding_gemma2_case
    model = module.Model(config)
    with pytest.raises(ValueError, match="require.*config"):
        getattr(model, f"get_{modality}_features")(mx.zeros((1, 2, 3)), None)


def test_embedding_gemma2_sanitize_and_converted_checkpoint_roundtrip(
    embedding_gemma2_case, tmp_path
):
    module, config = embedding_gemma2_case

    config.audio_config = module.AudioConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        subsampling_conv_channels=(4, 2),
        output_proj_dims=8,
    )
    model = module.Model(config)
    weights = dict(tree_flatten(model.parameters()))
    original = {}
    for key, value in weights.items():
        if key.endswith("conv.weight"):
            value = value.transpose(0, 3, 1, 2)
        elif key.endswith("depthwise_conv1d.weight"):
            value = value.transpose(0, 2, 1)
        original["model." + key] = value
    sanitized = model.sanitize(original)
    twice = model.sanitize(sanitized)
    assert weights.keys() == sanitized.keys() == twice.keys()
    for key in weights:
        np.testing.assert_array_equal(sanitized[key], weights[key])
        np.testing.assert_array_equal(twice[key], weights[key])

    (tmp_path / "config.json").write_text(json.dumps(asdict(config)))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), sanitized)
    loaded = embedding_loader.load_embedding_model(tmp_path)
    ids = mx.array([[1, 2, 3]])
    np.testing.assert_allclose(
        loaded(ids).text_embeds, model(ids).text_embeds, atol=1e-6
    )

    config.audio_config = None
    reduced = embedding_loader.load_embedding_model(tmp_path, config=asdict(config))
    assert reduced.audio_tower is None
    np.testing.assert_allclose(
        reduced(ids).text_embeds, model(ids).text_embeds, atol=1e-6
    )


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_embedding_gemma2_quantized_checkpoint_roundtrip(
    embedding_gemma2_case, tmp_path, bits
):
    from mlx_vlm.quant_utils import quantize_model

    module, config = embedding_gemma2_case
    config.text_config.hidden_size = 64
    config.text_config.intermediate_size = 128
    config.text_config.hidden_size_per_layer_input = 64
    config.text_config.head_dim = 32
    config.text_config.per_layer_config = {"01": {"head_dim": 64}}
    model, saved_config = quantize_model(module.Model(config), asdict(config), 64, bits)
    assert model.language_model.embed_tokens.bits == bits
    (tmp_path / "config.json").write_text(json.dumps(saved_config))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    reloaded = embedding_loader.load_embedding_model(tmp_path)
    ids = mx.array([[1, 2, 3], [4, 5, 6]])
    expected = model(ids).text_embeds
    actual = reloaded(ids).text_embeds
    assert reloaded.language_model.embed_tokens.bits == bits
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(mx.linalg.norm(actual, axis=-1), 1, atol=1e-6)


@pytest.fixture(params=[c["id"] for c in DATA["cases"] if c["module"] == "minicpmv4_7"])
def minicpmv47_case(request):
    case = next(c for c in DATA["cases"] if c["id"] == request.param)
    module = importlib.import_module("mlx_vlm.models." + case["module"])
    return module, build_config(module, case["config"])


@pytest.mark.parametrize("mode,count", [("16x", 4), ("4x", 16)])
def test_minicpmv47_vision_and_chunked_prefill(minicpmv47_case, mode, count):
    module, config = minicpmv47_case

    model = module.Model(config)
    ids = mx.array([[1, 50] + [48] * count + [51, 2]])
    media = {
        "pixel_values": [[mx.random.normal((3, 2, 128))]],
        "tgt_sizes": [np.array([[8, 8]])],
        "downsample_mode": mode,
    }
    features = model.get_input_embeddings(ids, **media)
    assert features.inputs_embeds.shape == (*ids.shape, 16)
    assert features.position_ids.shape == (3, 1, ids.shape[-1])
    assert features.rope_deltas.item() < 0
    ModelChecks().prefill_parity(model, ids, media, chunk_sizes=(3,))

    # The outer model must preserve the canvas delta when decoding without images.
    cache = model.language_model.make_cache()
    model(ids, cache=cache, **media)
    next_logits = model(mx.array([[3]]), cache=cache).logits
    extended = mx.concatenate([ids, mx.array([[3]])], axis=1)
    expected = model(extended, **media).logits[:, -1:]
    assert mx.allclose(next_logits, expected, atol=2e-5)


def test_minicpmv47_canvas_and_padding(minicpmv47_case):
    module, config = minicpmv47_case
    rope_module = importlib.import_module(module.__name__ + ".rope")

    rope = rope_module.CanvasRoPE(config)
    ids = np.array([[0, 0, 1, 50, 48, 48, 48, 48, 51, 2]])
    positions, deltas = rope.get_rope_index(
        ids,
        (ids == 48).astype(np.int32),
        np.array([[8, 8]]),
        attention_mask=ids != 0,
    )
    expected = [
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 4],
        [0, 0, 0, 0, 1, 1, 2, 2, 3, 4],
        [0, 0, 0, 0, 1, 2, 1, 2, 3, 4],
    ]
    np.testing.assert_array_equal(positions[:, 0], expected)
    assert deltas.item() == -3


def test_minicpmv47_grid_validation(minicpmv47_case):
    module, config = minicpmv47_case

    model = module.Model(config)
    ids = mx.array([[50, 48, 51]])
    with pytest.raises(ValueError, match="crop grids"):
        model.get_input_embeddings(ids)
    with pytest.raises(ValueError, match="token count"):
        model.get_input_embeddings(ids, tgt_sizes=[[[8, 8]]])
    with pytest.raises(ValueError, match="crop grids"):
        model.language_model.get_rope_index(ids)


def test_minicpmv47_slices_share_the_thumbnail_canvas(minicpmv47_case):
    module, config = minicpmv47_case
    rope_module = importlib.import_module(module.__name__ + ".rope")

    ids = np.array([[1, 50, 48, 51, 52, 48, 53, 52, 48, 53, 2]])
    positions, _ = rope_module.CanvasRoPE(config).get_rope_index(
        ids, (ids == 48).astype(np.int32), np.array([[4, 4]] * 3)
    )
    expected = [
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
        [0, 0, 1, 2, 1, 1, 1, 1, 1, 1, 4],
        [0, 0, 1, 3, 1, 1, 1, 2, 2, 2, 4],
    ]
    np.testing.assert_array_equal(positions[:, 0], expected)


@pytest.mark.parametrize("minicpmv47_case", ["minicpmv4_7"], indirect=True)
def test_minicpmv47_sanitize_is_idempotent(minicpmv47_case):
    module, config = minicpmv47_case

    model = module.Model(config)
    weights = {
        "model.language_model.layers.0.input_layernorm.weight": mx.zeros((16,)),
        "model.language_model.layers.0.linear_attn.conv1d.weight": mx.ones((80, 1, 3)),
        "model.vision_tower.embeddings.patch_embedding.weight": mx.zeros((16, 3, 2, 2)),
    }
    first = model.sanitize(weights)
    second = model.sanitize(first)
    assert first.keys() == second.keys()
    for key in first:
        assert mx.array_equal(first[key], second[key])
    assert mx.array_equal(
        first["language_model.model.layers.0.input_layernorm.weight"],
        mx.ones((16,)),
    )
    assert first["vision_tower.embeddings.patch_embedding.weight"].shape == (
        16,
        2,
        2,
        3,
    )


@pytest.mark.parametrize("minicpmv47_case", ["minicpmv4_7_moe"], indirect=True)
def test_minicpmv47_native_expert_checkpoint_and_reload(minicpmv47_case, tmp_path):
    module, config = minicpmv47_case
    qwen = importlib.import_module("mlx_vlm.models.qwen3_5.qwen3_5")
    moe = importlib.import_module("mlx_vlm.models.qwen3_5_moe.language")

    model = module.Model(config)
    assert isinstance(model.layers[0].mlp, moe.Qwen3_5MoeSparseMoeBlock)
    expected = dict(tree_flatten(model.parameters()))
    raw = {}
    for key, value in expected.items():
        if ".switch_mlp." in key:
            continue
        if key.startswith("language_model.model."):
            key = key.replace("language_model.model.", "model.language_model.", 1)
            if key.endswith(qwen.NORM_WEIGHT_SUFFIXES):
                value = value - 1
            if "conv1d.weight" in key:
                value = value.swapaxes(1, 2)
        elif key.startswith("language_model.lm_head."):
            key = key.replace("language_model.", "", 1)
        else:
            key = "model." + key.replace("vit_merger.", "vision_tower.vit_merger.", 1)
            if key.endswith("patch_embedding.weight"):
                value = value.transpose(0, 3, 1, 2)
        raw[key] = value
    for index, layer in enumerate(model.layers):
        prefix = f"model.language_model.layers.{index}.mlp.experts"
        experts = layer.mlp.switch_mlp
        raw[f"{prefix}.gate_up_proj"] = mx.concatenate(
            [experts.gate_proj.weight, experts.up_proj.weight], axis=-2
        )
        raw[f"{prefix}.down_proj"] = experts.down_proj.weight

    sanitized = model.sanitize(raw)
    assert sanitized.keys() == expected.keys()
    for key, value in expected.items():
        assert mx.array_equal(sanitized[key], value), key
    reloaded = model.sanitize(sanitized)
    for key, value in sanitized.items():
        assert mx.array_equal(reloaded[key], value), key
    model.load_weights(list(reloaded.items()), strict=True)
    mx.eval(model(mx.array([[1, 2, 3]])).logits)
    (tmp_path / "config.json").write_text(json.dumps(asdict(model.config)))
    for weights in (raw, sanitized):
        mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
        loaded = load_model(tmp_path)
        assert isinstance(loaded.layers[0].mlp, moe.Qwen3_5MoeSparseMoeBlock)
        for key, value in tree_flatten(loaded.parameters()):
            assert mx.array_equal(value, expected[key]), key


@pytest.mark.parametrize("minicpmv47_case", ["minicpmv4_7_moe"], indirect=True)
def test_minicpmv47_router_quantization_and_state_dtype(minicpmv47_case):
    module, config = minicpmv47_case

    model = module.Model(config)
    prefix = "language_model.model.layers.0"
    assert model.quant_predicate(f"{prefix}.mlp.gate", None) == {
        "group_size": 64,
        "bits": 8,
    }
    assert model.quant_predicate(f"{prefix}.mlp.switch_mlp.up_proj", None)
    assert not model.quant_predicate(f"{prefix}.linear_attn.in_proj_qkv", None)
    assert not model.cast_predicate(f"{prefix}.linear_attn.A_log")
    assert model.cast_predicate(f"{prefix}.mlp.switch_mlp.up_proj.weight")


@pytest.mark.parametrize("name", DATA["dense"])
def test_dense_model(name):
    module = importlib.import_module("mlx_vlm.models." + name)
    config = DATA["dense"][name]
    model = module.Model(module.ModelConfig.from_dict(copy.deepcopy(config)))
    ModelChecks().forward_cache(model, config["vocab_size"])


def tiny_config(family, profile=None, **overrides):
    """Build a fresh tiny config, optionally selecting a named test profile."""
    case = TINY_MODELS[family]
    fields = TINY_DEFAULTS | case["config"] | case.get("profiles", {}).get(profile, {})
    module = importlib.import_module("mlx_vlm.models." + case["module"])
    return build_config(module, fields | overrides, case["config_type"])


def test_gemma3_can_preserve_caller_supplied_embedding_scale():
    case = next(case for case in DATA["cases"] if case["module"] == "gemma3")
    module = importlib.import_module(f"mlx_vlm.models.{case['module']}.language")
    config = build_config(
        module,
        case["config"]["text_config"]
        | TINY_DEFAULTS
        | dict(intermediate_size=32, head_dim=8, sliding_window_pattern=1),
        "TextConfig",
    )
    model = module.Gemma3Model(config, scale_inputs_embeds=False)
    capture = MagicMock(side_effect=lambda inputs, *_: inputs)
    model.layers, model.norm = [capture], nn.Identity()
    expected = mx.ones((1, 1, config.hidden_size))
    output = model(None, inputs_embeds=mx.array(expected))

    assert mx.array_equal(capture.call_args.args[0], expected)
    assert mx.array_equal(output, expected)


def rotating_language_model(case):
    """Reuse model contracts and tiny defaults for cache-specific model checks."""
    name = case["module"]
    module = importlib.import_module(f"mlx_vlm.models.{name}")
    language = importlib.import_module(f"mlx_vlm.models.{name}.language")
    outer, text = None, {}
    if "case" in case:
        outer = build_config(module, model_case(case["case"])["config"])
        outer = getattr(outer, "thinker_config", outer)
        text = asdict(outer.text_config)
    defaults = {} if name == "florence2" else TINY_DEFAULTS
    config = build_config(
        language,
        text | defaults | case["config"],
        case.get("config_type", "TextConfig"),
    )
    if name == "florence2":
        return language.Florence2Decoder(config), config
    if name in ("qwen2", "qwen3", "mllama", "moondream3", "minimax_m3_vl", "gemma3n"):
        return language.LanguageModel(config), config
    if outer is None:
        outer = module.ModelConfig(
            text_config=config, vision_config=module.VisionConfig()
        )
    else:
        outer.text_config = config
    model = (
        language.LanguageModel(outer)
        if name == "hunyuan_vl"
        else language.LanguageModel(config, outer)
    )
    return model, config


@pytest.mark.parametrize(
    "case",
    [c for c in DATA["rotating_cache"] if "chunk_parity" in c["checks"]],
    ids=lambda c: c["module"],
)
@pytest.mark.parametrize("keep", [0, 2])
def test_rotating_cache_random_model_chunk_parity(case, keep):
    mx.random.seed(17)
    model, config = rotating_language_model(case)
    model.eval()
    mx.eval(model.parameters())
    ids = (mx.arange(41)[None] * 7 + 3) % config.vocab_size

    def forward(schedule):
        for attr in ("_rope_deltas", "_position_ids"):
            if hasattr(model, attr):
                setattr(model, attr, None)
        caches = [RotatingKVCache(8, keep=keep) for _ in model.layers]
        result, start = [], 0
        for n in schedule:
            logits = model(ids[:, start : start + n], cache=caches).logits
            mx.eval(logits, [c.state for c in caches])
            result.append(logits)
            start += n
        assert start == ids.shape[1]
        return np.array(mx.concatenate(result, axis=1))

    expected = forward([1] * 41)
    # Include a chunk larger than the cache and multi-token continuation after
    # single-token decoding has wrapped the storage cursor.
    for schedule in ([3] * 13 + [1, 1], [17, 7, 8, 8, 1], [1] * 19 + [5, 8, 8, 1]):
        np.testing.assert_allclose(forward(schedule), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "case",
    [c for c in DATA["rotating_cache"] if "prompt_metadata" in c["checks"]],
    ids=lambda c: c["module"],
)
def test_rotated_model_slices_full_prompt_metadata(case):
    name = case["module"]
    mx.random.seed(19)
    model, config = rotating_language_model(case)
    model.eval()
    ids = (mx.arange(24)[None] * 7 + 3) % config.vocab_size
    positions = mx.broadcast_to(mx.arange(24)[None, None, :], (3, 1, 24))
    positions = positions + mx.array([3, 7, 11])[:, None, None]
    if name == "falcon_ocr":
        positions = positions[0]
    metadata = {"position_ids": positions}
    if name in ("qwen3_vl", "qwen3_vl_moe", "qwen3_omni_moe"):
        metadata["visual_pos_masks"] = mx.ones((1, 24), mx.bool_)
        metadata["deepstack_visual_embeds"] = (
            mx.random.normal((1, 24, 2, config.hidden_size))
            if name == "qwen3_omni_moe"
            else [mx.random.normal((24, config.hidden_size)) for _ in range(2)]
        )

    def forward(full_metadata):
        for attr in ("_rope_deltas", "_position_ids"):
            if hasattr(model, attr):
                setattr(model, attr, None)
        cache = [RotatingKVCache(8, keep=2) for _ in model.layers]
        for start in range(21):
            out = model(ids[:, start : start + 1], cache=cache)
            mx.eval(out.logits)
        assert cache[0].offset == 21 and cache[0]._idx != 21
        kwargs = dict(metadata)
        if not full_metadata:
            kwargs["position_ids"] = positions[..., 21:24]
            if "visual_pos_masks" in kwargs:
                kwargs["visual_pos_masks"] = kwargs["visual_pos_masks"][:, 21:24]
                embeds = kwargs["deepstack_visual_embeds"]
                kwargs["deepstack_visual_embeds"] = (
                    [e[21:24] for e in embeds]
                    if isinstance(embeds, list)
                    else embeds[:, 21:24]
                )
        if name == "falcon_ocr":
            # Falcon's full-prompt metadata is passed with precomputed embeddings.
            kwargs["inputs_embeds"] = model.model.embed_tokens(ids[:, 21:24])
            if not full_metadata:
                # This wrapper accepts full position IDs only; compare the
                # internally selected positions against a direct model call.
                mask = create_attention_mask(kwargs["inputs_embeds"], cache)
                return np.array(
                    model.lm_head(
                        model.model(
                            ids[:, 21:24],
                            inputs_embeds=kwargs["inputs_embeds"],
                            position_ids=positions[..., 21:24],
                            mask=mask,
                            cache=cache,
                        )
                    )
                )
        return np.array(model(ids[:, 21:24], cache=cache, **kwargs).logits)

    np.testing.assert_allclose(forward(True), forward(False), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "case",
    [c for c in DATA["rotating_cache"] if "per_layer_inputs" in c["checks"]],
    ids=lambda c: c["module"],
)
def test_rotated_per_layer_inputs(case):
    mx.random.seed(23)
    model, config = rotating_language_model(case)
    # The constructor initializes this gate to zero, which would hide a bad
    # per-layer-input slice in an otherwise randomly initialized model.
    for layer in model.layers:
        layer.altup.correct_output_scale = mx.random.uniform(
            low=0.5, high=1.5, shape=(config.hidden_size,)
        )
    model.eval()
    ids = (mx.arange(41)[None] * 7 + 3) % config.vocab_size
    per_layer_inputs = model.model.get_per_layer_inputs(ids)

    def forward(step, full_metadata):
        cache, outputs = model.make_cache(), []
        for start in range(0, 41, step):
            metadata = (
                per_layer_inputs
                if full_metadata
                else per_layer_inputs[:, start : start + step]
            )
            out = model(
                ids[:, start : start + step], cache=cache, per_layer_inputs=metadata
            )
            mx.eval(out.logits)
            outputs.append(out.logits)
        return np.array(mx.concatenate(outputs, axis=1))

    expected = forward(1, False)
    np.testing.assert_allclose(forward(3, True), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "case",
    [c for c in DATA["rotating_cache"] if "decoder_positions" in c["checks"]],
    ids=lambda c: c["module"],
)
def test_rotated_decoder_positions_and_mask(case):
    mx.random.seed(29)
    decoder, config = rotating_language_model(case)
    decoder.eval()
    encoder_states = mx.random.normal((1, 4, config.d_model))
    embeddings = mx.random.normal((1, 41, config.d_model))

    def forward(step):
        cache = [[RotatingKVCache(8, keep=2), KVCache()] for _ in decoder.layers]
        outputs = []
        for start in range(0, 41, step):
            out = decoder(
                inputs_embeds=embeddings[:, start : start + step],
                encoder_hidden_states=encoder_states,
                cache=cache,
            )
            mx.eval(out)
            outputs.append(out)
        return np.array(mx.concatenate(outputs, axis=1))

    np.testing.assert_allclose(forward(3), forward(1), atol=1e-4, rtol=1e-4)


# Document layout


class TestPPDocLayoutV3(unittest.TestCase):
    def test_pp_doclayout_v3_config_routes(self):
        from mlx_vlm.models import pp_doclayout_v3
        from mlx_vlm.utils import get_model_and_args

        model_class, model_type = get_model_and_args(
            config={"model_type": "pp_doclayout_v3"}
        )
        assert model_class is pp_doclayout_v3
        assert model_type == "pp_doclayout_v3"
        assert pp_doclayout_v3.ModelConfig().num_labels == 25

        cfg = pp_doclayout_v3.ModelConfig.from_dict(
            {
                "model_type": "pp_doclayout_v3",
                "num_labels": 37,
                "id2label": {"0": "Question", "1": "Paragraph"},
                "backbone_config": {"model_type": "hgnet_v2"},
            }
        )
        assert cfg.id2label == {0: "Question", 1: "Paragraph"}
        assert cfg.num_queries == 300
        assert cfg.decoder_layers == 6

    def test_pp_doclayout_v3_decode_order(self):
        import mlx.core as mx

        from mlx_vlm.models.pp_doclayout_v3.decoder import decode_order

        # Chain 0 -> 1 -> 2 with strong pairwise scores.
        scores = mx.array([[-1e4, 5.0, 5.0], [-1e4, -1e4, 5.0], [-1e4, -1e4, -1e4]])
        assert decode_order(scores).tolist() == [0, 1, 2]
        # Reference formula cross-check on random input.
        rng_scores = mx.random.normal((7, 7))
        mx.eval(rng_scores)
        got = decode_order(rng_scores).tolist()
        import numpy as np

        arr = np.array(rng_scores.tolist())
        s = 1.0 / (1.0 + np.exp(-arr))
        votes = np.triu(s, 1).sum(0) + np.tril(1.0 - s.T, -1).sum(0)
        assert got == np.argsort(votes).tolist()

    def test_pp_doclayout_v3_mask_to_box(self):
        import mlx.core as mx

        from mlx_vlm.models.pp_doclayout_v3.decoder import mask_to_box_coordinate

        mask = mx.zeros((1, 2, 8, 10))
        mask[0, 0, 2:5, 3:7] = 1.0
        mx.eval(mask)
        boxes = mask_to_box_coordinate(mask)
        mx.eval(boxes)
        # x in [3,7), y in [2,5) over W=10,H=8 -> cxcywh
        self.assertAlmostEqual(float(boxes[0, 0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(boxes[0, 0, 1]), 0.4375, places=5)
        self.assertAlmostEqual(float(boxes[0, 0, 2]), 0.4, places=5)
        self.assertAlmostEqual(float(boxes[0, 0, 3]), 0.375, places=5)
        # Empty mask -> zeros.
        assert float(boxes[0, 1].sum()) == 0.0

    def test_pp_doclayout_v3_bilinear_upsample(self):
        import mlx.core as mx

        from mlx_vlm.models.pp_doclayout_v3.encoder import upsample_bilinear2x

        x = mx.array([[[[0.0], [1.0]], [[2.0], [3.0]]]])
        mx.eval(x)
        y = upsample_bilinear2x(x)
        mx.eval(y)
        # align_corners=False exact values: edges replicate, interior lerps.
        assert tuple(y.shape) == (1, 4, 4, 1)
        self.assertAlmostEqual(float(y[0, 0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(y[0, 0, 1, 0]), 0.25, places=5)
        self.assertAlmostEqual(float(y[0, 1, 0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(y[0, 1, 1, 0]), 0.75, places=5)
        self.assertAlmostEqual(float(y[0, 3, 3, 0]), 3.0, places=5)

    def test_pp_doclayout_conversion_without_torch(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import mlx.core as mx

        from mlx_vlm.models.pp_doclayout_v3.convert import convert

        for dtype in (mx.float32, mx.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "source"
                src.mkdir()
                config = {"model_type": "pp_doclayout_v3"}
                (src / "config.json").write_text(json.dumps(config))
                weight = mx.arange(48).reshape(2, 3, 2, 4).astype(dtype)
                mx.save_safetensors(
                    str(src / "model.safetensors"),
                    {
                        "model.backbone.model.embedder.stem1.convolution.weight": weight,
                        "model.denoising_class_embed.weight": mx.ones((2, 2)),
                        "model.backbone.model.embedder.stem1.normalization.num_batches_tracked": mx.array(
                            0
                        ),
                    },
                )
                with (
                    patch.dict(
                        "sys.modules", {"torch": None, "safetensors.torch": None}
                    ),
                    patch("mlx_vlm.models.pp_doclayout_v3.convert._verify") as verify,
                ):
                    out = convert(str(src), str(Path(tmp) / "converted"))
                    again = convert(
                        str(out), str(Path(tmp) / "reconverted"), "bfloat16"
                    )
                assert verify.call_count == 2
                key = "backbone.embedder.stem1.conv.weight"
                expected = weight.transpose(0, 2, 3, 1)
                converted = mx.load(str(out / "model.safetensors"))
                reconverted = mx.load(str(again / "model.safetensors"))
                assert set(converted) == {key}
                assert set(reconverted) == {key}
                assert converted[key].dtype == mx.float32
                assert reconverted[key].dtype == mx.bfloat16
                assert converted[key].tolist() == expected.tolist()
                assert reconverted[key].tolist() == expected.tolist()
                assert json.loads((out / "config.json").read_text()) == config


# Loading and utility contracts


def test_load_config_applies_generation_config_sampling_defaults(tmp_path):
    generation_config = {
        "eos_token_id": [2, 3],
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_new_tokens": 4096,
    }
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "demo", "eos_token_id": 1}), encoding="utf-8"
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps(generation_config), encoding="utf-8"
    )

    config = load_config(tmp_path)

    assert config["generation_config"] == generation_config
    assert config["eos_token_id"] == [2, 3]
    assert config["do_sample"] is True
    assert config["temperature"] == 1.0
    assert config["top_p"] == 0.95
    assert config["top_k"] == 64
    assert "max_new_tokens" not in config


def test_get_model_path_downloads_jsonl_tokenizers(monkeypatch, tmp_path):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("mlx_vlm.utils.snapshot_download", fake_snapshot_download)

    assert get_model_path("org/model") == tmp_path
    assert "*.jsonl" in captured["allow_patterns"]


def test_load_passes_revision():
    model_mock = MagicMock()
    model_mock.config = MagicMock(eos_token_id=None)
    processor_mock = MagicMock()

    with (
        patch("mlx_vlm.utils.get_model_path") as mock_get_model_path,
        patch("mlx_vlm.utils.load_model", return_value=model_mock),
        patch("mlx_vlm.utils.load_processor", return_value=processor_mock),
        patch("mlx_vlm.utils.load_image_processor", return_value=None),
    ):
        mock_get_model_path.return_value = Path("/tmp/model")

        model, processor = load("repo", revision="abc")

        assert model is model_mock
        assert processor is processor_mock
        mock_get_model_path.assert_called_with(
            "repo", revision="abc", force_download=False
        )


def test_get_model_and_args_rejects_unknown_text_configs():
    with pytest.raises(ValueError):
        get_model_and_args({"model_type": "unknown_text_arch"})


class TestDropModulesWithoutWeights:
    class ParameterlessHelper(nn.Module):
        pass

    class FakeModel(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config
            self.language_model = nn.Linear(2, 2, bias=False)
            self.vision_tower = nn.Linear(2, 2, bias=True)
            self.parameterless_helper = (
                TestDropModulesWithoutWeights.ParameterlessHelper()
            )

    def test_keeps_module_declared_in_manifest_but_not_loaded(self):
        # Manifest declares vision weights but none loaded -> keep, strict fails (#1963).
        model = self.FakeModel()
        weights = {"language_model.weight": mx.zeros((2, 2))}
        declared = {"language_model.weight", "vision_tower.weight", "vision_tower.bias"}

        _drop_modules_without_weights(model, weights, declared)

        assert model.vision_tower is not None
        with pytest.raises(ValueError, match="Missing"):
            model.load_weights(list(weights.items()), strict=True)

    def test_drops_module_absent_from_manifest(self, caplog):
        # The manifest also omits vision -> an intentional text-only conversion.
        model = self.FakeModel()
        weights = {"language_model.weight": mx.zeros((2, 2))}
        declared = {"language_model.weight"}

        with caplog.at_level(logging.WARNING):
            _drop_modules_without_weights(model, weights, declared)

        assert model.vision_tower is None
        assert "vision_tower" in caplog.text


def test_load_safetensors_reinterprets_f8_e8m0_header(tmp_path):
    path = tmp_path / "model.safetensors"
    header = {"weight": {"dtype": "F8_E8M0", "shape": [1], "data_offsets": [0, 1]}}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00")

    loaded = {"weight": mx.array([1], dtype=mx.uint8)}

    def fake_mx_load(file_path):
        current = json.loads(path.read_bytes()[8 : 8 + len(header_bytes)])
        if current["weight"]["dtype"] == "F8_E8M0":
            raise RuntimeError("unsupported dtype F8_E8M0")
        assert current["weight"]["dtype"] == "U8"
        return loaded

    with patch("mlx_vlm.utils.mx.load", side_effect=fake_mx_load):
        assert _load_safetensors(str(path)) is loaded

    restored = json.loads(path.read_bytes()[8 : 8 + len(header_bytes)])
    assert restored["weight"]["dtype"] == "F8_E8M0"


class _CheckpointConfig:
    @classmethod
    def from_dict(cls, config):
        return cls()


class _CheckpointModel(nn.Module):
    def __init__(self, config, **modules):
        super().__init__()
        self.config = config
        for name, module in modules.items():
            setattr(self, name, module)

    def load_weights(self, weights, strict=True):
        self.loaded_weights, self.loaded_strict = weights, strict


@contextmanager
def _checkpoint_loading(config, model_class, weights, *, side_effect=None):
    with (
        patch("mlx_vlm.utils.load_config", return_value=config),
        patch("mlx_vlm.utils.glob.glob", return_value=["/tmp/model/model.safetensors"]),
        patch("mlx_vlm.utils._load_safetensors", return_value=weights),
        patch(
            "mlx_vlm.utils.get_model_and_args",
            return_value=(
                SimpleNamespace(ModelConfig=_CheckpointConfig, Model=model_class),
                config["model_type"],
            ),
        ),
        patch("mlx_vlm.utils.nn.quantize", side_effect=side_effect) as quantize,
    ):
        yield quantize


def test_load_model_uses_deepseek_v4_fp8_quantization_config():

    quantization = {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
        "language_model.weight": {"group_size": 64, "bits": 8, "mode": "affine"},
    }
    with (
        patch(
            "mlx_vlm.models.deepseek_v4.language.make_quantization_config",
            return_value=quantization,
        ) as make_quantization_config,
        _checkpoint_loading(
            {
                "model_type": "deepseek_v4",
                "quantization_config": {"quant_method": "fp8"},
            },
            lambda config: _CheckpointModel(
                config, language_model=nn.Linear(2, 2, bias=False)
            ),
            {},
        ) as quantize,
    ):
        model = load_model(Path("/tmp/model"), lazy=True)
    make_quantization_config.assert_called_once_with(model)
    quantize.assert_called_once()
    assert quantize.call_args.kwargs["group_size"] == 64
    assert quantize.call_args.kwargs["bits"] == 8
    assert quantize.call_args.kwargs["mode"] == "affine"


def test_load_model_transforms_fine_grained_fp8_by_format():

    source_config = {
        "model_type": "future_compatible_model",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        },
    }
    with _checkpoint_loading(
        source_config,
        lambda config: _CheckpointModel(config, proj=nn.Linear(128, 128, bias=False)),
        {
            "proj.weight": mx.zeros((128, 128), dtype=mx.uint8),
            "proj.weight_scale_inv": mx.ones((1, 1), dtype=mx.bfloat16),
        },
    ) as quantize:
        model = load_model(Path("/tmp/model"), lazy=True)
    quantize.assert_called_once()
    assert quantize.call_args.kwargs["group_size"] == 32
    assert quantize.call_args.kwargs["bits"] == 8
    assert quantize.call_args.kwargs["mode"] == "mxfp8"
    loaded = dict(model.loaded_weights)
    assert loaded["proj.weight"].dtype == mx.uint32
    assert loaded["proj.scales"].dtype == mx.uint8
    assert "proj.weight_scale_inv" not in loaded


def test_load_model_quantizes_projector_with_scales_when_skip_vision():

    def model(config):
        projector = nn.Module()
        projector.linear_1 = nn.Linear(64, 64, bias=False)
        return _CheckpointModel(
            config,
            vision_tower=nn.Linear(64, 64, bias=False),
            multi_modal_projector=projector,
            language_model=nn.Linear(64, 64, bias=False),
        )

    weights = {
        "language_model.weight": mx.zeros((64, 16), dtype=mx.uint32),
        "language_model.scales": mx.zeros((64, 1), dtype=mx.float16),
        "multi_modal_projector.linear_1.weight": mx.zeros((64, 16), dtype=mx.uint32),
        "multi_modal_projector.linear_1.scales": mx.zeros((64, 1), dtype=mx.float16),
        "vision_tower.weight": mx.zeros((64, 64), dtype=mx.float16),
    }
    selected = {}

    def fake_quantize(model, *args, **kwargs):
        predicate = kwargs["class_predicate"]
        selected["language"] = predicate("language_model", model.language_model)
        selected["projector"] = predicate(
            "multi_modal_projector.linear_1", model.multi_modal_projector.linear_1
        )
        selected["vision"] = predicate("vision_tower", model.vision_tower)

    with _checkpoint_loading(
        {
            "model_type": "kimi_vl",
            "quantization": {"group_size": 64, "bits": 8},
            "vision_config": {"skip_vision": True},
        },
        model,
        weights,
        side_effect=fake_quantize,
    ):
        load_model(Path("/tmp/model"), lazy=True)
    assert selected == {"language": True, "projector": True, "vision": False}


# Local Python model files

MODEL_PY = textwrap.dedent("""
    import mlx.core as mx
    import mlx.nn as nn


    class ModelConfig:
        # Deliberately minimal: exposing text_config/vision_config attributes
        # opts in to update_module_configs, which then requires TextConfig /
        # VisionConfig classes in this module. A model_file module controls
        # both sides of that contract.
        def __init__(self, model_type="custom"):
            self.model_type = model_type

        @classmethod
        def from_dict(cls, params):
            return cls(model_type=params.get("model_type", "custom"))


    class Model(nn.Module):
        loaded_via_model_file = True

        def __init__(self, config):
            super().__init__()
            self.config = config
            self.proj = nn.Linear(4, 4, bias=False)

        def __call__(self, x):
            return self.proj(x)
    """)


def _write_checkpoint(path, config_extra=None):
    config = {"model_type": "does-not-exist-in-registry", "model_file": "model.py"}
    config.update(config_extra or {})
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "model.py").write_text(MODEL_PY, encoding="utf-8")
    mx.save_safetensors(
        str(path / "model.safetensors"),
        {"proj.weight": mx.zeros((4, 4))},
        metadata={"format": "mlx"},
    )


def test_load_model_uses_checkpoint_model_file(tmp_path):
    _write_checkpoint(tmp_path)

    model = load_model(tmp_path)

    # the class must come from the checkpoint's model.py, not the registry
    # (the registry would have raised: model_type does not exist there)
    assert getattr(model, "loaded_via_model_file", False) is True
    assert model.proj.weight.shape == (4, 4)


def test_missing_model_file_raises_clearly(tmp_path):
    _write_checkpoint(tmp_path)
    (tmp_path / "model.py").unlink()

    with pytest.raises(FileNotFoundError, match="model_file"):
        load_model(tmp_path)


def test_qwen3_omni_audio_batch_matches_repeated_input():
    from mlx_vlm.models.qwen3_omni_moe.audio import AudioModel
    from mlx_vlm.models.qwen3_omni_moe.config import AudioConfig

    config = AudioConfig(
        d_model=8,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        num_mel_bins=8,
        output_dim=8,
        downsample_hidden_size=4,
        conv_chunksize=4,
        max_source_positions=64,
    )
    model = AudioModel(config)
    sample = mx.random.normal((8, 170))
    output = model(mx.concatenate([sample, sample], axis=1), mx.array([170, 170]))

    assert output.shape == (44, 8)
    assert mx.allclose(output[:22], output[22:])


def test_qwen3_omni_rope_delta_ignores_batch_padding():
    from mlx_vlm.models.qwen3_omni_moe.language import LanguageModel

    model = LanguageModel.__new__(LanguageModel)
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=10,
        video_token_id=11,
        vision_start_token_id=12,
    )
    input_ids = mx.array([[0, 0, 1, 2], [1, 2, 3, 4]])
    attention_mask = mx.array([[0, 0, 1, 1], [1, 1, 1, 1]])

    _, rope_deltas = model.get_rope_index(input_ids, attention_mask=attention_mask)

    assert rope_deltas.tolist() == [[0], [0]]


# Patch embedding layouts


QWEN_PATCH_EMBED_KEY = "model.visual.patch_embed.proj.weight"


QWEN_SANITIZED_KEY = "vision_tower.patch_embed.proj.weight"


def _qwen_patch_model(
    in_channels=3, temporal_patch_size=2, patch_size=4, hidden_size=8
):
    qwen = importlib.import_module("mlx_vlm.models.qwen3_5")
    text_config = qwen.TextConfig(
        model_type="qwen3_5_text",
        hidden_size=32,
        intermediate_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=1,
        max_position_embeddings=128,
        full_attention_interval=2,
        head_dim=16,
    )
    vision_config = qwen.VisionConfig(
        model_type="qwen3_5",
        depth=1,
        hidden_size=hidden_size,
        intermediate_size=16,
        out_hidden_size=32,
        num_heads=1,
        in_channels=in_channels,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=1,
        num_position_embeddings=4,
    )
    config = qwen.ModelConfig(
        text_config=text_config, vision_config=vision_config, model_type="qwen3_5"
    )
    return qwen.Model(config), vision_config


def test_patch_embed_is_transposed_from_ncdhw_to_ndhwc():
    """Qwen3.5 stores the Conv3d patch embed as NCDHW; MLX expects NDHWC."""
    model, vision_config = _qwen_patch_model()
    expected = model.vision_tower.patch_embed.proj.weight.shape

    ncdhw = mx.zeros(
        (
            vision_config.hidden_size,
            vision_config.in_channels,
            vision_config.temporal_patch_size,
            vision_config.patch_size,
            vision_config.patch_size,
        ),
        dtype=mx.bfloat16,
    )
    sanitized = model.sanitize({QWEN_PATCH_EMBED_KEY: ncdhw})

    assert sanitized[QWEN_SANITIZED_KEY].shape == expected


def test_glm_quantized_head_sanitization_loads_strictly():
    module = importlib.import_module("mlx_vlm.models.glm5_next")
    model = module.Model(
        module.ModelConfig(
            text_config=tiny_config("glm", hidden_size=32),
            vision_config=module.VisionConfig(
                depth=1,
                hidden_size=16,
                intermediate_size=32,
                out_hidden_size=32,
                projection_intermediate_size=32,
                num_heads=2,
                image_size=8,
                patch_size=2,
            ),
        )
    )
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda path, _: path.endswith("language_model.lm_head"),
    )
    checkpoint = dict(tree_flatten(model.parameters()))
    head = {
        f"lm_head.{suffix}": checkpoint.pop(f"language_model.lm_head.{suffix}")
        for suffix in ("weight", "scales", "biases")
    }
    sanitized = model.sanitize(head)
    assert sanitized.keys() == {
        f"language_model.lm_head.{suffix}" for suffix in ("weight", "scales", "biases")
    }
    again = model.sanitize(sanitized)
    assert again.keys() == sanitized.keys()
    for key, value in sanitized.items():
        assert mx.array_equal(again[key], value).item()
    model.load_weights(list(model.sanitize(checkpoint | head).items()), strict=True)


def test_moondream2_sanitize_remaps_checkpoint_layout():
    from mlx_vlm.models.moondream2 import Model

    source = {
        "model.text.wte": "text.model.embed_tokens.weight",
        "model.text.blocks.0.attn.qkv.weight": "text.model.layers.0.attn.qkv.weight",
        "model.text.post_ln.weight": "text.model.post_ln.weight",
        "model.text.lm_head.weight": "text.lm_head.weight",
        "model.vision.patch_emb.weight": "vision.encoder.patch_emb.weight",
        "model.vision.blocks.0.ln1.weight": "vision.encoder.blocks.0.ln1.weight",
        "model.vision.proj_mlp.fc1.weight": "vision.proj_mlp.fc1.weight",
    }
    weights = {key: mx.zeros((1,)) for key in source}
    weights["model.region.coord_decoder.fc1.weight"] = mx.zeros((1,))

    assert set(Model.sanitize(None, weights)) == set(source.values())


def test_moondream3_sanitize_remaps_raw_and_preserves_converted_keys():
    from mlx_vlm.models.moondream3 import Model

    raw = {
        "model.text.blocks.0.attn.qkv.weight": mx.zeros((1,)),
        "model.vision.blocks.0.ln1.weight": mx.zeros((1,)),
    }
    converted = Model.sanitize(None, raw)
    assert set(converted) == {
        "text.model.blocks.0.attn.qkv.weight",
        "vision.encoder.blocks.0.ln1.weight",
    }
    converted["text.lm_head.weight"] = mx.zeros((1,))
    converted["vision.proj_mlp.fc1.weight"] = mx.zeros((1,))
    assert Model.sanitize(None, converted).keys() == converted.keys()


class TestKolibri1Sanitize(unittest.TestCase):
    def _model(self):
        from mlx_vlm.models.kolibri1 import ModelConfig
        from mlx_vlm.models.kolibri1.language import LanguageModel

        case = next(case for case in DATA["cases"] if case["module"] == "kolibri1")
        config = copy.deepcopy(case["config"])
        config["num_hidden_layers"] = 1
        config["layer_types"] = ["full_attention"]
        return LanguageModel(ModelConfig.from_dict(config))

    def _checkpoint(self, model, layout):
        checkpoint = {}
        for key, value in tree_flatten(model.parameters()):
            if ".mlp.gate.e_score_correction_bias" in key:
                if layout == "per_expert":
                    key = key.replace(
                        ".mlp.gate.e_score_correction_bias",
                        ".moe.router.expert_bias",
                    )
                elif layout == "stacked":
                    key = key.replace(
                        ".mlp.gate.e_score_correction_bias", ".mlp.expert_bias"
                    )
            if ".mlp.switch_mlp." in key:
                prefix, projection = key.split(".mlp.switch_mlp.")
                if layout == "stacked":
                    checkpoint[f"{prefix}.mlp.experts.{projection}"] = value
                    continue
                if layout == "per_expert":
                    for expert_idx, expert_value in enumerate(value):
                        checkpoint[
                            f"{prefix}.mlp.experts.{expert_idx}.{projection}"
                        ] = expert_value
                    continue
            checkpoint[key] = value
        return checkpoint

    def test_checkpoint_layouts_sanitize_to_the_model_exactly(self):
        model = self._model()
        expected = dict(tree_flatten(model.parameters()))
        for layout in ("per_expert", "stacked", "canonical"):
            with self.subTest(layout=layout):
                sanitized = model.sanitize(self._checkpoint(model, layout))
                self.assertEqual(sanitized.keys(), expected.keys())
                for key, value in expected.items():
                    self.assertTrue(mx.array_equal(sanitized[key], value).item(), key)
                model.load_weights(list(sanitized.items()), strict=True)


class TestQwen3_5MoeText(unittest.TestCase):
    """Decoder-only Qwen3.5 MoE checkpoints (model_type qwen3_5_moe_text)."""

    def _model(self):
        from mlx_vlm.models.qwen3_5_moe_text import Model, ModelConfig

        case = next(
            case for case in DATA["cases"] if case["module"] == "qwen3_5_moe_text"
        )
        model = Model(ModelConfig.from_dict(copy.deepcopy(case["config"])))
        model.update(
            tree_map(
                lambda p: (mx.random.randint(-8, 8, p.shape) / 4).astype(p.dtype),
                model.parameters(),
            )
        )
        return model

    def _raw_checkpoint(self, model, prefix, fused):
        """Rebuild a published checkpoint from the model's own parameters."""
        from mlx_vlm.models.qwen3_5.qwen3_5 import NORM_WEIGHT_SUFFIXES

        raw = {}
        for key, value in tree_flatten(model.parameters()):
            if ".switch_mlp." in key:
                continue
            if key.startswith("language_model.model."):
                raw_key = prefix + key[len("language_model.model.") :]
            else:
                raw_key = key.replace("language_model.lm_head", "lm_head", 1)
            if "conv1d.weight" in key:
                value = value.swapaxes(1, 2)
            if any(key.endswith(sfx) for sfx in NORM_WEIGHT_SUFFIXES):
                value = value - 1.0
            raw[raw_key] = value
        for layer_idx, layer in enumerate(model.layers):
            experts = f"{prefix}layers.{layer_idx}.mlp.experts"
            switch = layer.mlp.switch_mlp
            if fused:
                raw[f"{experts}.gate_up_proj"] = mx.concatenate(
                    [switch.gate_proj.weight, switch.up_proj.weight], axis=-2
                )
                raw[f"{experts}.down_proj"] = switch.down_proj.weight
            else:
                for name in ("gate_proj", "up_proj", "down_proj"):
                    weight = getattr(switch, name).weight
                    for e in range(weight.shape[0]):
                        raw[f"{experts}.{e}.{name}.weight"] = weight[e]
        return raw

    def test_published_layouts_sanitize_to_the_model_exactly(self):
        model = self._model()
        expected = dict(tree_flatten(model.parameters()))
        for prefix in ("model.language_model.", "model."):
            for fused in (True, False):
                with self.subTest(prefix=prefix, fused=fused):
                    raw = self._raw_checkpoint(model, prefix, fused)
                    raw[f"{prefix}layers.0.mlp.gate.input_global_scale"] = mx.ones(1)
                    sanitized = model.sanitize(raw)
                    self.assertEqual(sanitized.keys(), expected.keys())
                    for key, value in expected.items():
                        self.assertTrue(
                            mx.array_equal(sanitized[key], value).item(), key
                        )
                    model.load_weights(list(sanitized.items()), strict=True)

    def test_ragged_expert_tensors_fail_clearly(self):
        model = self._model()
        raw = self._raw_checkpoint(model, "model.", fused=True)
        raw["model.layers.0.mlp.experts.gate_up_proj"] = mx.zeros((123,))
        with self.assertRaisesRegex(ValueError, "expected \\[num_experts"):
            model.sanitize(raw)

    def test_sanitize_is_idempotent_on_converted_weights(self):
        model = self._model()
        converted = dict(tree_flatten(model.parameters()))
        again = model.sanitize(dict(converted))
        self.assertEqual(again.keys(), converted.keys())
        for key, value in converted.items():
            self.assertTrue(mx.array_equal(again[key], value).item(), key)

    def test_missing_mrope_section_is_plain_rope(self):
        from mlx_vlm.models.qwen3_5_moe_text import Model, ModelConfig

        model = self._model()
        weights = list(tree_flatten(model.parameters()))
        ids = mx.array([[3, 1, 4, 1, 5, 9, 2, 6]])

        def logits(rope_parameters):
            config = vars(model.config) | {"rope_parameters": rope_parameters}
            other = Model(ModelConfig.from_dict(config))
            other.load_weights(weights, strict=True)
            return other(ids).logits

        base = {
            "rope_type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 1.0,
        }
        missing = logits(dict(base))
        for section in ([2, 1, 1], [1, 1, 2], [4, 0, 0]):
            with self.subTest(section=section):
                explicit = logits(dict(base, mrope_section=section))
                self.assertTrue(mx.allclose(missing, explicit, atol=1e-5).item())


# DeepSeek-V4.1 regressions beyond the shared model contracts


class TestDeepseekV41EndToEnd(unittest.TestCase):
    def setUp(self):
        mx.random.seed(0)

    def test_index_rotary_matches_reference_at_long_positions(self):
        from mlx_vlm.models.deepseek_v41.language import (
            _apply_index_rotary,
            _index_cos_sin,
        )

        dim, length, theta = 64, 32769, 160000
        factor, beta_fast, beta_slow, original_length = 16, 32, 1, 65536

        def correction(rotations):
            return (
                dim
                * np.log(original_length / (rotations * 2 * np.pi))
                / (2 * np.log(theta))
            )

        low = int(np.floor(correction(beta_fast)))
        high = int(np.ceil(correction(beta_slow)))
        ramp = (np.arange(dim // 2, dtype=np.float32) - low) / (high - low)
        smooth = 1 - np.clip(ramp, 0, 1)
        powers = np.power(np.float64(theta), np.arange(0, dim, 2) / dim)
        inv = 1 / powers.astype(np.float32)
        inv = inv / factor * (1 - smooth) + inv * smooth
        phase = np.arange(length, dtype=np.float32)[:, None] * inv[None, :]
        cos, sin = _index_cos_sin(
            length, dim, theta, (factor, beta_fast, beta_slow, original_length)
        )
        phase = phase.astype(np.float64)
        np.testing.assert_array_equal(np.array(cos), np.cos(phase).astype(np.float32))
        np.testing.assert_array_equal(np.array(sin), np.sin(phase).astype(np.float32))

        for batch, count in ((1, 1), (2, 17)):
            with self.subTest(batch=batch, length=count):
                x = mx.random.normal((batch, count, 4, 80)).astype(mx.bfloat16)
                c, s = cos[None, -count:, None], sin[None, -count:, None]
                data = np.array(x.astype(mx.float32))
                rotated = data[..., -dim:].copy().view(np.complex64)
                frequencies = np.array(c) + np.complex64(1j) * np.array(s)
                rotated = (rotated * frequencies).view(np.float32)
                expected = np.concatenate([data[..., :-dim], rotated], axis=-1)
                actual = _apply_index_rotary(x, c, s, dim)
                self.assertEqual(actual.dtype, x.dtype)
                self.assertTrue(
                    mx.array_equal(actual, mx.array(expected).astype(x.dtype)).item()
                )

    def test_hc_expansion_preserves_reference_rounding(self):
        from mlx_vlm.models.deepseek_v41.language import hc_expand

        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            for batch, length in ((1, 1), (2, 5), (1, 256)):
                with self.subTest(dtype=dtype, batch=batch, length=length):
                    x = mx.random.normal((batch, length, 128)).astype(dtype)
                    residual = mx.random.normal((batch, length, 4, 128)).astype(dtype)
                    post = mx.random.uniform(low=0, high=2, shape=(batch, length, 4))
                    comb = mx.softmax(mx.random.normal((batch, length, 4, 4)), axis=-1)
                    xx, rr, pp, cc = [
                        np.array(a.astype(mx.float32))
                        for a in (x, residual, post, comb)
                    ]
                    # Independent CPU evaluation of the source's explicit
                    # products and reduction; matmul changes the rounding.
                    expected = pp[..., None] * xx[:, :, None, :] + np.sum(
                        cc[..., None] * rr[..., None, :], axis=2, dtype=np.float32
                    )
                    actual = hc_expand(x, residual, post, comb)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertTrue(
                        mx.array_equal(actual, mx.array(expected).astype(dtype)).item()
                    )

    def test_activation_fp8_rounding_matches_reference(self):
        from mlx_vlm.models.deepseek_v41.fakequant import fake_quant_fp8_ue8m0

        # Exercise every E4M3 rounding tie and its immediate FP32 neighbors.
        # The 448 anchor fixes the block scale at one.
        levels = np.array(mx.from_fp8(mx.arange(127, dtype=mx.uint8), mx.float32))
        midpoints = (levels[:-1] + levels[1:]) / 2
        values = np.concatenate(
            [
                np.nextafter(midpoints, -np.inf),
                midpoints,
                np.nextafter(midpoints, np.inf),
            ]
        )
        ties = np.zeros((2 * len(values), 32), dtype=np.float32)
        ties[:, 0] = np.concatenate([values, -values])
        ties[:, 1] = 448
        samples = [
            mx.array(ties),
            mx.random.normal((4, 3, 512)).transpose(1, 0, 2),
            mx.random.normal((2, 64)) * 1e-8,
            mx.zeros((1, 32)),
        ]
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            for sample in samples:
                with self.subTest(dtype=dtype, shape=sample.shape):
                    x = sample.astype(dtype)
                    blocks = np.array(x.astype(mx.float32)).reshape(-1, 32)
                    amax = np.maximum(np.abs(blocks).max(-1, keepdims=True), 1e-4)
                    fraction, exponent = np.frexp(amax * np.float32(1 / 448))
                    scales = np.ldexp(np.ones_like(amax), exponent - (fraction == 0.5))
                    encoded = mx.to_fp8(mx.array(np.clip(blocks / scales, -448, 448)))
                    expected = (
                        (mx.from_fp8(encoded, mx.float32) * mx.array(scales))
                        .reshape(x.shape)
                        .astype(dtype)
                    )
                    actual = fake_quant_fp8_ue8m0(x)
                    self.assertTrue(mx.array_equal(actual, expected).item())

    def test_config_accepts_unused_prediction_layers(self):
        from mlx_vlm.models import deepseek_v41

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        source = copy.deepcopy(case["config"])
        source["compress_ratios"] += [0, 0, 0]
        source["num_nextn_predict_layers"] = 3
        config = deepseek_v41.ModelConfig.from_dict(source)
        self.assertEqual(config.compress_ratios, case["config"]["compress_ratios"])

    def test_head_and_router_load_bf16_checkpoint_weights_as_fp32(self):
        from mlx_vlm.models import deepseek_v41

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        model = deepseek_v41.Model(build_config(deepseek_v41, case["config"]))
        head = model.language_model.head
        gate = model.layers[0].ffn.gate
        weight = mx.random.normal(head.weight.shape).astype(mx.bfloat16)
        gate_weight = mx.random.normal(gate.weight.shape).astype(mx.bfloat16)
        x = mx.random.normal((1, 1, model.config.hidden_size)).astype(mx.bfloat16)
        gate.weight = gate_weight
        expected_indices, expected_weights = gate(x)
        mx.eval(expected_indices, expected_weights)
        sanitized = model.sanitize(
            {"head.weight": weight, "layers.0.ffn.gate.weight": gate_weight}
        )
        model.load_weights(list(sanitized.items()), strict=False)
        self.assertEqual(head.weight.dtype, mx.float32)
        self.assertEqual(gate.weight.dtype, mx.float32)
        expected = x.astype(mx.float32) @ weight.T
        self.assertTrue(mx.array_equal(head(x), expected).item())
        indices, weights = gate(x)
        self.assertTrue(mx.array_equal(indices, expected_indices).item())
        self.assertTrue(mx.array_equal(weights, expected_weights).item())

    def test_sparse_attention_preserves_precision_and_masks(self):
        from mlx_vlm.models import deepseek_v41
        from mlx_vlm.models.deepseek_v41.language import DeepseekV41Attention

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        config = build_config(deepseek_v41, case["config"])
        attention = DeepseekV41Attention(config, 1)
        for dtype in (mx.float32, mx.bfloat16):
            for length in (1, 3):
                with self.subTest(dtype=dtype, length=length):
                    q = mx.random.normal(
                        (2, config.num_attention_heads, length, config.head_dim)
                    ).astype(dtype)
                    window = mx.random.normal((2, 7, config.head_dim)).astype(dtype)
                    pool = mx.random.normal((2, 9, config.head_dim)).astype(dtype)
                    indices = mx.broadcast_to(
                        mx.array([[[2, -1, 0, 7]]]), (2, length, 4)
                    )
                    mask = (
                        mx.arange(7)[None, None, None]
                        <= mx.arange(length)[None, None, :, None] + 3
                    )
                    attention.attn_sink = mx.random.normal(
                        (config.num_attention_heads,)
                    )
                    actual = attention._sparse_attention(q, window, pool, indices, mask)
                    expected = np.zeros(q.shape, dtype=np.float32)
                    query, local, pooled = [
                        np.array(a.astype(mx.float32)) for a in (q, window, pool)
                    ]
                    sinks = np.array(attention.attn_sink.astype(mx.float32))
                    # Select only valid rows in an independent dense FP32 oracle.
                    for batch in range(2):
                        for pos in range(length):
                            kv = np.concatenate(
                                [local[batch, : pos + 4], pooled[batch, [2, 0, 7]]]
                            )
                            scores = (query[batch, :, pos] @ kv.T) * attention.scale
                            maximum = np.maximum(scores.max(-1), sinks)[:, None]
                            probabilities = np.exp(scores - maximum)
                            denominator = probabilities.sum(-1, keepdims=True) + np.exp(
                                sinks[:, None] - maximum
                            )
                            expected[batch, :, pos] = (probabilities / denominator) @ kv
                    # BF16 output rounds the FP32 result to its nearest value.
                    tolerance = 1e-5 if dtype == mx.float32 else 2**-8
                    self.assertEqual(actual.dtype, dtype)
                    self.assertTrue(
                        mx.allclose(
                            actual.astype(mx.float32),
                            mx.array(expected),
                            atol=1e-5,
                            rtol=tolerance,
                        ).item()
                    )

    def test_engram_offloading_matches_resident_rows(self):
        import tempfile

        from mlx_vlm.models import deepseek_v41
        from mlx_vlm.models.deepseek_v41.engram import (
            OffloadedEngramEmbedding,
            QuantizedEngramEmbedding,
        )

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        prefix = "language_model.layers.1.engram.embed."
        ids = mx.array([[0, 127, 3], [3, 1, 0]])
        for bits, mode, sharded in (
            (None, "affine", False),
            (4, "affine", False),
            (8, "affine", True),
            (4, "mxfp4", False),
            (8, "mxfp8", True),
        ):
            with (
                self.subTest(bits=bits, mode=mode, sharded=sharded),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory)
                config = copy.deepcopy(case["config"])
                model = deepseek_v41.Model(build_config(deepseek_v41, config))
                table = model.layers[1].engram.embed
                table.weight = table.weight.astype(mx.bfloat16)
                if bits is not None:
                    config["quantization"] = dict(group_size=32, bits=bits, mode=mode)
                    packed = QuantizedEngramEmbedding(128, 64, **config["quantization"])
                    packed.weight, packed.scales, *biases = mx.quantize(
                        table.weight, **config["quantization"]
                    )
                    packed.biases = biases[0] if biases else None
                    model.layers[1].engram.embed = packed
                    config["quantization"][prefix.rstrip(".")] = dict(
                        group_size=32, bits=bits, mode=mode
                    )
                expected = model.layers[1].engram.embed(ids)
                mx.eval(expected)
                self.assertEqual(expected.dtype, mx.bfloat16)
                weights = dict(tree_flatten(model.parameters()))
                if mode.startswith("mxfp"):
                    native_prefix = prefix.removeprefix("language_model.")
                    weights[native_prefix + "weight"] = weights.pop(
                        prefix + "weight"
                    ).view(mx.uint8)
                    weights[native_prefix + "scale"] = weights.pop(prefix + "scales")
                (path / "config.json").write_text(json.dumps(config))
                if sharded:
                    tables = {
                        key: weights.pop(key)
                        for key in list(weights)
                        if ".engram.embed." in key
                    }
                    mx.save_safetensors(str(path / "engram.safetensors"), tables)
                    (path / "model.safetensors.index.json").write_text(
                        json.dumps(
                            {
                                "weight_map": {
                                    **dict.fromkeys(weights, "model.safetensors"),
                                    **dict.fromkeys(tables, "engram.safetensors"),
                                }
                            }
                        )
                    )
                mx.save_safetensors(str(path / "model.safetensors"), weights)

                with patch("mlx_vlm.utils.mx.eval", wraps=mx.eval) as evaluate:
                    loaded = load_model(path)
                # Exportable tensors must be exposed after eager loading.
                self.assertNotIn(
                    prefix + "weight", dict(tree_flatten(evaluate.call_args.args[0]))
                )
                self.assertEqual(loaded.config.model_path, str(path))
                self.assertIsInstance(
                    loaded.layers[1].engram.embed, OffloadedEngramEmbedding
                )
                self.assertEqual(loaded.layers[1].engram.embed(ids).dtype, mx.bfloat16)
                self.assertTrue(
                    mx.array_equal(loaded.layers[1].engram.embed(ids), expected).item()
                )
                self.assertIn(
                    prefix + "weight", dict(tree_flatten(loaded.parameters()))
                )

                # Conversion uses lazy loading and must keep exportable table weights.
                lazy = load_model(path, lazy=True)
                self.assertEqual(lazy.config.model_path, str(path))
                self.assertIn(prefix + "weight", dict(tree_flatten(lazy.parameters())))
                self.assertTrue(
                    mx.array_equal(lazy.layers[1].engram.embed(ids), expected).item()
                )
                with patch.object(QuantizedEngramEmbedding, "_quantize_chunk_rows", 41):
                    quantized = lazy.layers[1].engram.embed.to_quantized(
                        group_size=32, bits=4
                    )
                resident = model.layers[1].engram.embed.to_quantized(
                    group_size=32, bits=4
                )
                for name in ("weight", "scales", "biases"):
                    self.assertTrue(
                        mx.array_equal(quantized[name], resident[name]).item()
                    )
                self.assertTrue(mx.array_equal(quantized(ids), resident(ids)).item())

                exported = str(path / "export.safetensors")
                lazy.save_weights(exported)
                self.assertIn(prefix + "weight", mx.load(exported))

    def test_fp8_scale_layouts_decode_exactly(self):
        from mlx_vlm.models.deepseek_v41.deepseek_v41 import _pack_source_weight

        raw = mx.full((64, 64), 56, dtype=mx.uint8)  # E4M3 encoding of 1.0.
        for rowwise in (False, True):
            with self.subTest(rowwise=rowwise):
                scale_rows = 64 if rowwise else 2
                scales = (
                    mx.arange(scale_rows * 2).reshape(scale_rows, 2) % 4 + 125
                ).astype(mx.uint8)
                packed, expanded, mode = _pack_source_weight(raw, scales)
                decoded = mx.dequantize(
                    packed, expanded, group_size=32, bits=8, mode=mode
                )
                # Keep the reference exact, independent of GPU pow rounding.
                expected = mx.array([0.25, 0.5, 1.0, 2.0])[
                    scales.astype(mx.int32) - 125
                ]
                expected = mx.repeat(expected, 32, axis=-1)
                if not rowwise:
                    expected = mx.repeat(expected, 32, axis=0)
                self.assertTrue(mx.array_equal(decoded, expected).item())

    def test_engram_chunked_requantization(self):
        from mlx_vlm.models.deepseek_v41.engram import QuantizedEngramEmbedding

        source = QuantizedEngramEmbedding(
            7, 256, group_size=32, bits=8, mode="mxfp8", scale_dtype=mx.uint8
        )
        source.weight, source.scales = mx.quantize(
            mx.random.normal((7, 256)).astype(mx.bfloat16),
            group_size=32,
            bits=8,
            mode="mxfp8",
        )
        expected = mx.quantize(
            mx.dequantize(
                source.weight, source.scales, group_size=32, bits=8, mode="mxfp8"
            ),
            group_size=64,
            bits=4,
        )
        with patch.object(QuantizedEngramEmbedding, "_quantize_chunk_rows", 3):
            converted = source.to_quantized(group_size=64, bits=4)
        for actual, reference in zip(
            (converted.weight, converted.scales, converted.biases), expected
        ):
            self.assertTrue(mx.array_equal(actual, reference).item())
        self.assertIs(converted.to_quantized(group_size=64, bits=4), converted)

    def test_indexer_uses_unrotated_latents_and_adjacent_pair_rope(self):
        from mlx_vlm.models import deepseek_v41
        from mlx_vlm.models.deepseek_v41 import language
        from mlx_vlm.models.deepseek_v41.fakequant import (
            fake_quant_fp4_e4m3,
            fake_quant_fp4_ue8m0,
            fake_quant_fp8_ue8m0,
        )

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        config = build_config(deepseek_v41, case["config"])

        def rotate(value, positions):
            # DeepSeek's reference apply_rotary_emb views adjacent pairs as complex.
            dtype = value.dtype
            value = np.array(value.astype(mx.float32))
            rd = config.qk_rope_head_dim
            frequencies = config.compress_rope_theta ** (
                -np.arange(0, rd, 2, dtype=np.float32) / rd
            )
            angles = np.asarray(positions, dtype=np.float32)[:, None] * frequencies
            phases = np.exp(1j * angles).astype(np.complex64)
            phases = phases.reshape(1, len(positions), *((1,) * (value.ndim - 3)), -1)
            pairs = np.ascontiguousarray(value[..., -rd:]).view(np.complex64)
            rotated = np.ascontiguousarray(pairs * phases).view(np.float32)
            return mx.array(
                np.concatenate([value[..., :-rd], rotated], axis=-1)
            ).astype(dtype)

        for dtype in (mx.float32, mx.bfloat16):
            with self.subTest(dtype=dtype):
                attn = language.DeepseekV41Attention(config, 1)
                attn.update(tree_map(lambda p: p.astype(dtype), attn.parameters()))
                # Ratio-2 pooling projections remain FP32 in the reference.
                attn.compressor.wkv.weight = attn.compressor.wkv.weight.astype(
                    mx.float32
                )
                attn.compressor.wgate.weight = attn.compressor.wgate.weight.astype(
                    mx.float32
                )
                x = mx.random.normal((2, 16, config.hidden_size)).astype(dtype)
                qr = attn.q_norm(attn.wq_a(fake_quant_fp8_ue8m0(x)))
                cache = language.DeepseekV41Cache(config.num_hidden_layers)
                latent = attn.compressor(x, 0, cache)
                positions = range(0, x.shape[1], attn.compress_ratio)
                expected_keys = fake_quant_fp4_ue8m0(
                    rotate(attn.indexer.k_norm(attn.indexer.wk(latent)), positions)
                )
                q = attn.indexer.wq_b(fake_quant_fp8_ue8m0(qr)).reshape(
                    *x.shape[:2], config.index_n_heads, config.index_head_dim
                )
                expected_queries = fake_quant_fp4_ue8m0(rotate(q, range(x.shape[1])))
                expected_pool = fake_quant_fp4_e4m3(rotate(latent, positions))
                with patch.object(
                    language, "_index_scores", wraps=language._index_scores
                ) as score:
                    pool, indices = attn._compress_part(x, qr, 0, cache)
                for actual, expected in (
                    (cache.index_k, expected_keys),
                    (score.call_args.args[0], expected_queries),
                    (pool, expected_pool),
                ):
                    self.assertEqual(actual.dtype, dtype)
                    self.assertTrue(mx.allclose(actual, expected, atol=1e-6).item())

                # Match the reference's BF16 rounding after the dot product,
                # head weighting, and head reduction, including non-tied top-k.
                weights = attn.indexer.weights_proj(x) * (
                    config.index_head_dim**-0.5 * config.index_n_heads**-0.5
                )
                dots = mx.einsum(
                    "bshd,btd->bsht",
                    expected_queries.astype(mx.float32),
                    expected_keys.astype(mx.float32),
                ).astype(dtype)
                weighted = (
                    mx.maximum(dots, 0).astype(mx.float32)
                    * weights.astype(mx.float32)[..., None]
                ).astype(dtype)
                expected_scores = weighted.astype(mx.float32).sum(2).astype(dtype)
                lengths = (mx.arange(1, x.shape[1] + 1) // attn.compress_ratio)[:, None]
                expected_scores = mx.where(
                    mx.arange(expected_keys.shape[1]) < lengths,
                    expected_scores,
                    -mx.inf,
                )
                scores = language._index_scores(*score.call_args.args)
                self.assertEqual(scores.dtype, dtype)
                self.assertTrue(mx.array_equal(scores, expected_scores).item())
                selected = mx.take_along_axis(
                    expected_scores, mx.maximum(indices, 0), -1
                )
                selected = mx.where(indices < 0, -mx.inf, selected)
                best = mx.sort(expected_scores, axis=-1)[..., -indices.shape[-1] :]
                self.assertTrue(mx.array_equal(mx.sort(selected), best).item())

    def test_moe_matches_reference_activation_and_routing_precision(self):
        from mlx_vlm.models import deepseek_v41
        from mlx_vlm.models.deepseek_v41.language import DeepseekV41MoE

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        config = build_config(deepseek_v41, case["config"])
        # MLX's sorted affine GEMM can read past a partial 64-element K tile.
        # Use aligned projections while retaining decode and sorted dispatch.
        config.hidden_size = config.moe_intermediate_size = 64

        def project(layer, x, expert=None):
            # Use MLX's native MXFP8 codec as an independent rounding oracle.
            q, scales = mx.quantize(x, group_size=32, bits=8, mode="mxfp8")
            x = mx.dequantize(
                q, scales, group_size=32, bits=8, mode="mxfp8", dtype=x.dtype
            )
            weight = layer.weight if expert is None else layer.weight[expert]
            if hasattr(layer, "scales"):
                # Match MLX's decode/prefill GEMM rounding, keeping the expert
                # loop, activation codec and routing order independent.
                if expert is not None and x.shape[1] == 1:
                    return mx.gather_qmm(
                        x[..., None, :],
                        layer.weight,
                        layer.scales,
                        layer.biases,
                        rhs_indices=mx.full(x.shape[:-1], expert, dtype=mx.int32),
                        transpose=True,
                        group_size=layer.group_size,
                        bits=layer.bits,
                    ).squeeze(-2)
                return mx.quantized_matmul(
                    x,
                    weight,
                    layer.scales if expert is None else layer.scales[expert],
                    layer.biases if expert is None else layer.biases[expert],
                    transpose=True,
                    group_size=layer.group_size,
                    bits=layer.bits,
                )
            return x @ weight.T

        def expert(module, x, weight=None, index=None):
            gate = project(module.gate_proj, x, index).astype(mx.float32)
            up = project(module.up_proj, x, index).astype(mx.float32)
            gate = mx.minimum(gate, config.swiglu_limit)
            up = mx.clip(up, -config.swiglu_limit, config.swiglu_limit)
            activated = (gate / (1 + mx.exp(-gate))) * up
            if weight is not None:
                activated = activated * weight[..., None]
            return project(module.down_proj, activated.astype(x.dtype), index)

        for bits in (None, 4, 8):
            for length in (1, 17):  # Unsorted decode and sorted expert dispatch.
                with self.subTest(bits=bits, length=length):
                    model = DeepseekV41MoE(config)
                    model.gate.weight = mx.random.normal(model.gate.weight.shape) * 0.1
                    model.update(
                        tree_map(lambda p: p.astype(mx.bfloat16), model.parameters())
                    )
                    if bits is not None:
                        nn.quantize(
                            model,
                            group_size=32,
                            bits=bits,
                            class_predicate=lambda path, module: hasattr(
                                module, "to_quantized"
                            )
                            and path.startswith(("switch_mlp.", "shared_experts.")),
                        )
                    x = (mx.random.normal((2, length, config.hidden_size)) * 8).astype(
                        mx.bfloat16
                    )
                    indices, weights = model.gate(x)
                    expected = mx.zeros(x.shape, dtype=mx.float32)
                    for index in range(config.n_routed_experts):
                        weight = mx.where(indices == index, weights, 0).sum(-1)
                        expected = expected + expert(
                            model.switch_mlp, x, weight, index
                        ).astype(mx.float32)
                    expected = (
                        expected + expert(model.shared_experts, x).astype(mx.float32)
                    ).astype(x.dtype)
                    actual = model(x)
                    self.assertEqual(actual.dtype, x.dtype)
                    self.assertTrue(
                        mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()
                    )

    def test_image_tokens_use_visual_routing_and_break_engram_history(self):
        """The generation path supplies token ids, including during chunked prefill.

        Match the reference's explicit image mask and masked n-gram hashes,
        even when a prefill boundary falls inside an image span.
        """
        from mlx_vlm.generate.common import _chunked_prefill_enabled
        from mlx_vlm.models import deepseek_v41
        from mlx_vlm.models.deepseek_v41.engram import NgramHashState
        from mlx_vlm.models.deepseek_v41.language import LanguageModel

        case = next(case for case in DATA["cases"] if case["module"] == "deepseek_v41")
        config = build_config(deepseek_v41, case["config"])
        model = LanguageModel(config)
        self.assertTrue(_chunked_prefill_enabled(model))
        token_map = [i % 7 for i in range(config.vocab_size)]
        token_map[0] = 6
        model.engram_hash = NgramHashState(config, model.layout, token_map=token_map)
        model.head.weight = mx.random.normal(model.head.weight.shape) * 0.05
        for layer in model.layers:
            layer.ffn.gate.bias = mx.array([10.0, 9.0, 0.0, 0.0])
            layer.ffn.gate.bias_vl = mx.array([0.0, 0.0, 10.0, 9.0])

        ids = mx.array([[3, 7, 5, 5, 5, 11, 15, 19]])
        image_mask = ids == config.image_token_id
        embeds = model.embed_tokens(ids)
        embeds = mx.where(image_mask[..., None], mx.random.normal(embeds.shape), embeds)
        reference_cache = model.make_cache()
        hashes = model.engram_hash(ids, 0, reference_cache[0], token_mask=~image_mask)
        self.assertLess(mx.max(hashes).item(), config.engram_num_embeddings[0])
        output = model(
            ids,
            inputs_embeds=embeds,
            cache=reference_cache,
            image_mask=image_mask,
            engram_hashes=hashes,
        )
        self.assertIsNone(output.hidden_states)
        expected = output.logits
        mx.eval(expected)

        for chunks in ((8,), (3, 2, 3), (1,) * 8):
            with self.subTest(chunks=chunks):
                cache = model.make_cache()
                outputs, start = [], 0
                for length in chunks:
                    stop = start + length
                    outputs.append(
                        model(
                            inputs=ids[:, start:stop],
                            inputs_embeds=embeds[:, start:stop],
                            cache=cache,
                            n_to_process=length,
                        ).logits
                    )
                    start = stop
                actual = mx.concatenate(outputs, axis=1)
                mx.eval(actual)
                self.assertTrue(
                    bool(mx.array_equal(cache[0].engram, reference_cache[0].engram))
                )
                self.assertTrue(bool(mx.all(cache[0].engram[:, 2:5] == -1)))
                self.assertTrue(bool(mx.allclose(actual, expected, atol=1e-4)))


class TestMoERouterStopGradient:
    def test_gather_indices_are_stop_gradiented(self):
        import re

        import mlx_vlm

        offenders = []
        for path in sorted((Path(mlx_vlm.__file__).parent / "models").rglob("*.py")):
            src = path.read_text(encoding="utf-8")
            if "take_along_axis" not in src or "argpartition" not in src:
                continue
            from_argsort = set(
                re.findall(r"(\w+)\s*=\s*[^\n]*arg(?:partition|sort)", src)
            )
            for m in re.finditer(r"(\w+)\s*=\s*(\w+)\[", src):
                if m.group(2) in from_argsort:
                    from_argsort.add(m.group(1))
            stopped = set(re.findall(r"(\w+)\s*=\s*mx\.stop_gradient", src))
            for m in re.finditer(
                r"take_along_axis\(\s*[^,]+?\s*,\s*(.+?)\s*,\s*axis", src
            ):
                idx = m.group(1).strip()
                if "stop_gradient" in idx:
                    continue
                var = re.match(r"[A-Za-z_]\w*", idx)
                var = var.group(0) if var else ""
                if var and var not in stopped and var in from_argsort:
                    offenders.append(
                        f"{path.parent.name}/{path.name}: take_along_axis(..., {idx})"
                    )
        assert (
            not offenders
        ), "gather indices need mx.stop_gradient (MLX >= 0.32.1):\n" + "\n".join(
            offenders
        )

    @pytest.mark.parametrize(
        "module,cls,extra",
        [
            (
                "qwen3_5_moe",
                "Qwen3_5MoeSparseMoeBlock",
                {"shared_expert_intermediate_size": 8, "norm_topk_prob": True},
            ),
            ("qwen3_moe", "Qwen3MoeSparseMoeBlock", {"norm_topk_prob": True}),
        ],
    )
    def test_moe_router_backpropagates_through_dispatch(self, module, cls, extra):
        import importlib
        import types

        block_cls = getattr(
            importlib.import_module(f"mlx_vlm.models.{module}.language"), cls
        )
        args = types.SimpleNamespace(
            hidden_size=16,
            moe_intermediate_size=8,
            num_experts=4,
            num_experts_per_tok=2,
            **extra,
        )
        block = block_cls(args)
        x = mx.random.normal((1, 3, 16))
        _, grads = nn.value_and_grad(block, lambda m, inp: m(inp).sum())(block, x)
        total = sum(float(mx.sum(mx.abs(g))) for _, g in tree_flatten(grads))
        assert total > 0 and bool(mx.isfinite(mx.array(total)))


class TestPhiMoE:
    def test_sparsemixer(self):
        from mlx_vlm.models.phimoe.language import sparsemixer

        gates = mx.array([[3.0, 1.0, 0.0, -1.0], [3.0, 2.99, 0.0, -1.0]])
        inds, scores = sparsemixer(gates, top_k=2, jitter_eps=0.01)

        assert inds.tolist() == [[0, 1], [0, 1]]
        assert mx.allclose(scores, mx.array([[1.0, 1.0], [0.5025, 1.0]]))
        mx.eval(
            mx.grad(lambda g: sparsemixer(g, top_k=2, jitter_eps=0.01)[1].sum())(gates)
        )

    def test_plain_rope_without_scaling(self):
        from mlx_vlm.models.phimoe.config import ModelConfig
        from mlx_vlm.models.phimoe.language import Attention

        config = ModelConfig(
            hidden_size=16,
            num_attention_heads=2,
            num_key_value_heads=2,
            rope_scaling=None,
        )

        assert isinstance(Attention(config).rope, nn.RoPE)


class TestKimodoBackbone:
    @staticmethod
    def config(**overrides):
        from mlx_vlm.models.kimodo.config import ModelConfig

        values = dict(
            latent_dim=16,
            ff_size=32,
            num_layers=2,
            num_heads=4,
            llm_shape=(1, 8),
            num_text_tokens_override=3,
            max_position_embeddings=64,
            num_base_steps=32,
        )
        return ModelConfig(**dict(values, **overrides))

    def test_nested_checkpoint_config(self):
        from mlx_vlm.models.kimodo.config import ModelConfig

        config = ModelConfig.from_dict(
            {
                "_target_": "kimodo.model.Kimodo",
                "num_base_steps": 1000,
                "denoiser": {
                    "_target_": "kimodo.model.twostage_denoiser.TwostageDenoiser",
                    "latent_dim": 128,
                    "ckpt_path": "${checkpoint_dir}/model.safetensors",
                    "motion_rep": {
                        "_target_": "kimodo.motion_rep.ArdyMotionRep",
                        "fps": 60,
                        "skeleton": {"_target_": "kimodo.skeleton.G1Skeleton34"},
                    },
                },
            }
        )
        assert config.motion_rep == "ArdyMotionRep"
        assert config.skeleton == "G1Skeleton34"
        assert config.fps == 60
        assert config.latent_dim == 128
        assert ModelConfig.from_dict(config.to_dict()) == config
        assert ModelConfig.from_dict(None) == ModelConfig()
        with pytest.raises(ValueError, match="divisible"):
            self.config(latent_dim=18)

    def test_feature_and_mask_padding(self):
        from mlx_vlm.models.kimodo.backbone import pad_x_and_mask_to_fixed_size

        x = mx.arange(12, dtype=mx.float32).reshape(1, 4, 3)
        mask = mx.array([[True, True, True, False]])
        cropped, cropped_mask = pad_x_and_mask_to_fixed_size(x, mask, 2)
        assert mx.array_equal(cropped, x[:, :2])
        assert mx.array_equal(cropped_mask, mask[:, :2])
        padded, padded_mask = pad_x_and_mask_to_fixed_size(x, mask, 6)
        assert padded.shape == (1, 6, 3)
        assert mx.array_equal(padded[:, :4], x)
        assert mx.all(padded[:, 4:] == 0)
        assert mx.array_equal(padded_mask[:, :4], mask)
        assert not mx.any(padded_mask[:, 4:])

    @pytest.mark.parametrize("norm_first", [False, True])
    @pytest.mark.parametrize("activation", ["gelu", "relu"])
    @pytest.mark.parametrize("use_text_mask", [False, True])
    def test_reference_transformer(self, norm_first, activation, use_text_mask):
        torch = pytest.importorskip("torch")
        from mlx_vlm.models.kimodo.backbone import TransformerEncoderBlock

        config = self.config(
            norm_first=norm_first,
            activation=activation,
            use_text_mask=use_text_mask,
        )

        class Reference(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_text = torch.nn.Linear(8, 16)
                self.embed_timestep = torch.nn.Module()
                self.embed_timestep.time_embed = torch.nn.Sequential(
                    torch.nn.Linear(16, 16),
                    torch.nn.SiLU(),
                    torch.nn.Linear(16, 16),
                )
                self.input_linear = torch.nn.Linear(10, 16)
                self.output_linear = torch.nn.Linear(16, 5)
                self.linear_first_heading_angle = torch.nn.Linear(2, 16)
                self.seqTransEncoder = torch.nn.TransformerEncoder(
                    torch.nn.TransformerEncoderLayer(
                        16,
                        4,
                        dim_feedforward=32,
                        dropout=0,
                        activation=activation,
                        batch_first=True,
                        norm_first=norm_first,
                    ),
                    2,
                    enable_nested_tensor=False,
                )
                phase = torch.arange(64)[:, None] * torch.pow(
                    10000.0, -torch.arange(0, 16, 2).float() / 16
                )
                self.register_buffer(
                    "pe",
                    torch.stack((phase.sin(), phase.cos()), dim=-1).reshape(1, 64, 16),
                    persistent=False,
                )

            def forward(self, x, x_mask, text, text_mask, times, heading):
                text = torch.nn.functional.pad(text, (0, 0, 0, 3 - text.shape[1]))
                text_mask = torch.nn.functional.pad(
                    text_mask, (0, 3 - text_mask.shape[1]), value=False
                )
                if not use_text_mask:
                    text_mask = torch.ones_like(text_mask)
                heading = torch.stack((heading.cos(), heading.sin()), dim=-1)
                prefix = torch.cat(
                    (
                        self.embed_text(text),
                        self.embed_timestep.time_embed(self.pe[0, times])[:, None],
                        self.linear_first_heading_angle(heading)[:, None],
                    ),
                    dim=1,
                )
                valid = torch.cat(
                    (text_mask, torch.ones((x.shape[0], 2), dtype=torch.bool), x_mask),
                    dim=1,
                )
                sequence = torch.cat((prefix, self.input_linear(x)), dim=1)
                sequence = sequence + self.pe[:, : sequence.shape[1]]
                encoded = self.seqTransEncoder(sequence, src_key_padding_mask=~valid)
                return self.output_linear(encoded[:, prefix.shape[1] :])

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(12)
            reference = Reference().eval()
        model = TransformerEncoderBlock(config, 10, 5)
        model.eval()
        model.load_weights(
            [
                (name, mx.array(weight.detach().numpy()))
                for name, weight in reference.state_dict().items()
            ],
            strict=True,
        )
        rng = np.random.default_rng(12)
        for text_length in (2, 3):
            args = (
                torch.from_numpy(rng.normal(size=(2, 6, 10)).astype(np.float32)),
                torch.tensor([[True] * 6, [True] * 4 + [False] * 2]),
                torch.from_numpy(
                    rng.normal(size=(2, text_length, 8)).astype(np.float32)
                ),
                torch.tensor(
                    [[True] * text_length, [True] * (text_length - 1) + [False]]
                ),
                torch.tensor([8, 3]),
                torch.tensor([0.2, -0.4]),
            )
            with torch.no_grad():
                expected = reference(*args).numpy()
            with patch.object(
                mx, "eval", side_effect=AssertionError("Eager evaluation")
            ):
                actual = model(*(mx.array(value.numpy()) for value in args))
            np.testing.assert_allclose(np.array(actual), expected, rtol=2e-5, atol=1e-6)

    def test_heading_requirement(self):
        from mlx_vlm.models.kimodo.backbone import TransformerEncoderBlock

        model = TransformerEncoderBlock(self.config(), 10, 5)
        with pytest.raises(ValueError, match="heading"):
            model(
                mx.zeros((1, 2, 10)),
                mx.ones((1, 2), mx.bool_),
                mx.zeros((1, 1, 8)),
                mx.ones((1, 1), mx.bool_),
                mx.array([0]),
            )

    def test_conversion_failure_is_retryable(self, tmp_path):
        import sys
        import types

        pytest.importorskip("omegaconf")
        from mlx_vlm.models.kimodo.convert import convert
        from mlx_vlm.models.kimodo.language import LLM2VecEncoder

        source = tmp_path / "source"
        (source / "kimodo" / "skeleton").mkdir(parents=True)
        (source / "config.yaml").write_text("model_type: kimodo\n")
        mx.save_safetensors(str(source / "model.safetensors"), {"weight": mx.ones(1)})
        output = tmp_path / "converted"
        tensor = SimpleNamespace(numpy=lambda: np.ones(1, dtype=np.float32))
        stats = SimpleNamespace(mean=tensor, std=tensor)
        motion_rep = SimpleNamespace(
            global_root_stats=stats, local_root_stats=stats, body_stats=stats
        )
        source_package = types.ModuleType("kimodo")
        source_package.skeleton = SimpleNamespace(SOMASkeleton30=lambda: object())
        source_reps = types.ModuleType("kimodo.motion_rep")
        source_reps.KimodoMotionRep = source_reps.ArdyMotionRep = (
            lambda *args, **kwargs: motion_rep
        )
        fake_model = MagicMock()
        fake_model.sanitize.side_effect = lambda weights: weights

        def save(path, model):
            (path / "model.safetensors").write_bytes(b"test checkpoint")

        with (
            patch.dict(
                sys.modules,
                {"kimodo": source_package, "kimodo.motion_rep": source_reps},
            ),
            patch(
                "mlx_vlm.models.kimodo.convert.get_model_path", return_value=source
            ) as resolve,
            patch("mlx_vlm.models.kimodo.convert.skeleton_metadata", return_value={}),
            patch("mlx_vlm.models.kimodo.convert.Model", return_value=fake_model),
            patch("mlx_vlm.models.kimodo.convert.save_weights", side_effect=save),
        ):
            with patch.object(
                LLM2VecEncoder,
                "from_pretrained",
                side_effect=RuntimeError("adapter failure"),
            ):
                with pytest.raises(RuntimeError, match="adapter failure"):
                    convert(
                        "source", output, kimodo_source=source, text_encoder="missing"
                    )
            assert not output.exists()
            assert not list(tmp_path.glob(".converted-*"))
            resolve.assert_called_with(
                "source", allow_patterns=["*.safetensors", "*.yaml", "*.npy", "*.json"]
            )
            assert convert("source", output, kimodo_source=source) == output
            assert (output / "config.json").is_file()
            assert (output / "model.safetensors").read_bytes() == b"test checkpoint"
            with pytest.raises(FileExistsError, match="not empty"):
                convert("source", output, kimodo_source=source)
            assert (output / "model.safetensors").read_bytes() == b"test checkpoint"


class TestKimodoTextEncoder:
    @staticmethod
    def config():
        return dict(
            model_type="llama",
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
            max_position_embeddings=512,
            rope_theta=500000.0,
        )

    @staticmethod
    def tokenizer():
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        from transformers import PreTrainedTokenizerFast

        vocab = {
            "<unk>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<|start_header_id|>": 3,
            "<|end_header_id|>": 4,
            "<|eot_id|>": 5,
            "user": 6,
            "walk": 7,
            "forward": 8,
            "wave": 9,
        }
        backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        backend.post_processor = processors.TemplateProcessing(
            single="<bos> $A", special_tokens=[("<bos>", 1)]
        )
        return PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="<unk>",
            bos_token="<bos>",
            eos_token="<eos>",
            additional_special_tokens=[
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
            ],
        )

    def test_source_prompt_format_and_document_pooling(self):
        from mlx_vlm.models.kimodo.language import LLM2VecEncoder

        encoder = LLM2VecEncoder(self.config(), self.tokenizer())
        prompt = encoder.prepare_for_tokenization(
            encoder._convert_to_str("", "walk forward")
        )
        assert prompt == (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "!@#$%^&*()walk forward<|eot_id|>"
        )
        features = encoder.tokenize([prompt])
        np.testing.assert_array_equal(
            np.array(features["input_ids"]), [[1, 3, 6, 4, 7, 8, 5]]
        )
        np.testing.assert_array_equal(
            np.array(features["embed_mask"]), [[0, 0, 0, 0, 1, 1, 1]]
        )
        encoder.doc_max_length = 2
        assert encoder._convert_to_str("  move  ", "walk forward wave") == (
            "move !@#$%^&*()walk forward"
        )
        with patch.object(mx, "eval", side_effect=AssertionError("Eager evaluation")):
            batch, lengths = encoder(["walk forward", "wave"])
            single, length = encoder("walk forward")
        assert batch.shape == (2, 1, 16) and lengths == [1, 1]
        assert single.shape == (1, 16) and length == 1
        np.testing.assert_array_equal(np.array(single), np.array(batch[0]))

    def test_bidirectional_torch_parity(self):
        torch = pytest.importorskip("torch")
        from transformers import LlamaConfig, LlamaModel

        from mlx_vlm.models.kimodo.language import LLM2VecEncoder

        config = self.config()
        torch_config = LlamaConfig(**config)
        torch_config._attn_implementation = "eager"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(14)
            reference = LlamaModel(torch_config).eval()
        encoder = LLM2VecEncoder(torch_config.to_dict())
        encoder.load_weights(
            list(
                encoder.sanitize(
                    {
                        name: mx.array(value.detach().numpy())
                        for name, value in reference.state_dict().items()
                    }
                ).items()
            ),
            strict=True,
        )
        ids = torch.tensor([[1, 7, 8, 9], [2, 1, 7, 8]])
        mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
        embed_mask = torch.tensor([[0, 0, 1, 1], [0, 0, 0, 1]])
        bidirectional_mask = torch.where(
            mask[:, None, None, :].bool(), 0.0, -torch.inf
        ).expand(-1, 1, ids.shape[-1], -1)
        with torch.no_grad():
            hidden = reference(ids, attention_mask=bidirectional_mask).last_hidden_state
            expected = torch.stack((hidden[0, -2:].mean(0), hidden[1, -1])).numpy()
        actual = encoder.encode_tokens(
            mx.array(ids.numpy()), mx.array(mask.numpy()), mx.array(embed_mask.numpy())
        )
        np.testing.assert_allclose(np.array(actual), expected, atol=1e-6, rtol=2e-5)

    def test_both_adapters_and_converted_roundtrip(self, tmp_path):
        from mlx_vlm.models.kimodo.language import LLM2VecEncoder

        config = self.config()
        config["_name_or_path"] = "meta-llama/Meta-Llama-3-8B-Instruct"
        base, mntp, supervised = (
            tmp_path / name for name in ("base", "mntp", "supervised")
        )
        for path in (base, mntp, supervised):
            path.mkdir()
        tokenizer = self.tokenizer()
        initial = LLM2VecEncoder(config, tokenizer)
        weights = dict(tree_flatten(initial.parameters()))
        mx.save_safetensors(str(base / "model.safetensors"), weights)
        (base / "config.json").write_text(json.dumps(config))
        (mntp / "config.json").write_text(json.dumps(config))
        tokenizer.save_pretrained(mntp)
        target = "model.layers.0.self_attn.q_proj.weight"
        expected = weights[target]
        for scale, path in ((1, mntp), (2, supervised)):
            adapter_config = dict(
                peft_type="LORA", r=2, lora_alpha=4, base_model_name_or_path=str(base)
            )
            (path / "adapter_config.json").write_text(json.dumps(adapter_config))
            prefix = "base_model.model.layers.0.self_attn.q_proj"
            adapter = {
                prefix + ".lora_A.weight": mx.full((2, 16), 0.01 * scale),
                prefix + ".lora_B.weight": mx.full((16, 2), 0.02 * scale),
            }
            mx.save_safetensors(str(path / "adapter_model.safetensors"), adapter)
            expected = expected + 2 * (
                adapter[prefix + ".lora_B.weight"] @ adapter[prefix + ".lora_A.weight"]
            )
        encoder = LLM2VecEncoder.from_pretrained(
            str(mntp), str(supervised), dtype="float32"
        )
        actual_weights = dict(tree_flatten(encoder.parameters()))
        np.testing.assert_allclose(np.array(actual_weights[target]), np.array(expected))
        before, _ = encoder(["walk forward", "wave"])
        converted = tmp_path / "converted"
        encoder.save_pretrained(converted)
        reloaded = LLM2VecEncoder.from_pretrained(str(converted))
        after, _ = reloaded(["walk forward", "wave"])
        np.testing.assert_array_equal(np.array(before), np.array(after))


class TestKimodoPipeline:
    def model(self):
        from mlx_vlm.models.kimodo import Model, ModelConfig
        from mlx_vlm.models.kimodo.motion import KimodoMotionRep, Skeleton

        skeleton = Skeleton(
            neutral_joints=[[0, 0, 0], [-0.2, 0, 0], [0.2, 0, 0]],
            joint_parents=[-1, 0, 0],
            hip_joint_idx=[2, 1],
        )
        config = ModelConfig(
            latent_dim=16,
            ff_size=24,
            num_layers=1,
            num_heads=4,
            llm_shape=(1, 16),
            num_text_tokens_override=3,
            num_base_steps=16,
            max_position_embeddings=64,
        )
        model = Model(config, KimodoMotionRep(skeleton))
        model.eval()
        return model

    def test_ddim_reference(self):
        from mlx_vlm.models.kimodo.diffusion import Diffusion

        torch = pytest.importorskip("torch")
        schedule = Diffusion(1000)

        def alpha(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = torch.tensor(
            [
                min(1 - alpha((i + 1) / 1000) / alpha(i / 1000), 0.999)
                for i in range(1000)
            ],
            dtype=torch.float32,
        )
        base = torch.cumprod(1 - betas, 0)
        for steps in (1, 4, 15, 17, 1000):
            indices, actual, previous = schedule.schedule(steps)
            expected_indices = torch.round(
                torch.arange(1000) * (999 / max(1, steps - 1))
            ).long()[:steps]
            np.testing.assert_array_equal(np.array(indices), expected_indices.numpy())
            selected = base[expected_indices]
            prior = torch.cat([torch.ones(1), selected[:-1]])
            expected = torch.cumprod(1 - (1 - selected / prior), 0).clamp(min=1e-9)
            expected_prior = torch.cat([torch.ones(1), expected[:-1]])
            noisy = torch.linspace(-2, 2, 12).reshape(1, 3, 4)
            clean = noisy * 0.2
            for index in (0, steps - 1):
                eps = (torch.rsqrt(expected[index]) * noisy - clean) / torch.sqrt(
                    1 / expected[index] - 1
                )
                out = (
                    torch.sqrt(expected_prior[index]) * clean
                    + torch.sqrt(1 - expected_prior[index]) * eps
                )
                result = schedule.step(
                    mx.array(noisy.numpy()),
                    mx.array(clean.numpy()),
                    actual[index],
                    previous[index],
                )
                np.testing.assert_allclose(
                    np.array(result), out.numpy(), rtol=2e-5, atol=2e-6
                )
        with pytest.raises(ValueError):
            schedule.schedule(1001)

    @pytest.mark.parametrize(
        "strategy,weight", [("nocfg", 1), ("regular", 1.7), ("separated", (2, 0.5))]
    )
    def test_batched_guidance_matches_independent_branches(self, strategy, weight):
        from mlx_vlm.models.kimodo.diffusion import guided_prediction

        model = self.model()
        dim = model.motion_rep.motion_rep_dim
        x = mx.random.normal((2, 4, dim))
        pad = mx.array([[True] * 4, [True, True, False, False]])
        text = mx.random.normal((2, 1, 16))
        text_mask = mx.ones((2, 1), mx.bool_)
        time = mx.array([3, 9])
        heading = mx.array([0.2, -0.4])
        mask = (mx.arange(dim) % 3 == 0).astype(mx.float32)[None, None] * mx.ones(
            x.shape
        )
        observed = mx.random.normal(x.shape)

        def run(features, valid, observed_mask):
            return model(
                x,
                pad,
                features,
                valid,
                time,
                first_heading_angle=heading,
                motion_mask=observed_mask,
                observed_motion=observed,
            )

        actual = guided_prediction(
            model,
            x,
            pad,
            text,
            text_mask,
            time,
            first_heading_angle=heading,
            motion_mask=mask,
            observed_motion=observed,
            cfg_type=strategy,
            cfg_weight=weight,
        )
        unconditional = run(
            mx.zeros_like(text), mx.zeros_like(text_mask), mx.zeros_like(mask)
        )
        if strategy == "nocfg":
            expected = run(text, text_mask, mask)
        elif strategy == "regular":
            expected = unconditional + weight * (
                run(text, text_mask, mask) - unconditional
            )
        else:
            expected = (
                unconditional
                + weight[0]
                * (run(text, text_mask, mx.zeros_like(mask)) - unconditional)
                + weight[1]
                * (
                    run(mx.zeros_like(text), mx.zeros_like(text_mask), mask)
                    - unconditional
                )
            )
        assert mx.allclose(actual, expected, atol=2e-6, rtol=2e-5)

    @pytest.mark.parametrize("training", [False, True])
    def test_unconstrained_guidance_reuses_the_unconditional_branch(self, training):
        from mlx_vlm.models.kimodo.diffusion import guided_prediction

        model = self.model()
        model.train(training)
        x = mx.random.normal((2, 5, model.motion_rep.motion_rep_dim))
        pad = mx.array([[True] * 5, [True] * 3 + [False] * 2])
        text = mx.random.normal((2, 1, 16))
        mask = mx.ones((2, 1), mx.bool_)
        time = mx.array([3, 9])
        heading = mx.array([0.2, -0.4])
        conditional = model(x, pad, text, mask, time, first_heading_angle=heading)
        unconditional = model(
            x,
            pad,
            mx.zeros_like(text),
            mx.zeros_like(mask),
            time,
            first_heading_angle=heading,
        )
        recorder = MagicMock(wraps=model)
        recorder.training = training
        result = guided_prediction(
            recorder,
            x,
            pad,
            text,
            mask,
            time,
            first_heading_angle=heading,
            cfg_weight=(1.7, 9.0),
        )
        assert recorder.call_args.args[0].shape[0] == (6 if training else 4)
        assert mx.allclose(
            result,
            unconditional + 1.7 * (conditional - unconditional),
            atol=2e-6,
            rtol=2e-5,
        )

    @pytest.mark.parametrize("constrained", [False, True])
    @pytest.mark.parametrize(
        "strategy, weight", [("nocfg", 1), ("regular", 1.7), ("separated", (2, 0.5))]
    )
    def test_compiled_sampling_matches_eager_and_tracks_weights(
        self, constrained, strategy, weight
    ):
        from mlx_vlm.models.kimodo import KimodoPipeline

        model = self.model()
        pipeline = KimodoPipeline(model)
        shape = (2, 5, model.motion_rep.motion_rep_dim)
        text = mx.random.normal((2, 1, 16))
        text_mask = mx.ones((2, 1), mx.bool_)
        kwargs = dict(
            initial_noise=mx.random.normal(shape), cfg_type=strategy, cfg_weight=weight
        )
        if constrained:
            kwargs.update(
                motion_mask=mx.ones(shape), observed_motion=mx.random.normal(shape)
            )
        eager = pipeline.sample(text, text_mask, [5, 3], 3, **kwargs)
        with (
            patch.object(mx, "eval", side_effect=AssertionError("eager evaluation")),
            patch.object(mx, "synchronize", side_effect=AssertionError("synchronize")),
        ):
            compiled = pipeline.sample(
                text, text_mask, [5, 3], 3, compile=True, **kwargs
            )
        assert mx.allclose(compiled, eager, atol=3e-5, rtol=3e-5)
        model.denoiser.root_model.output_linear.bias += 0.1
        updated = pipeline.sample(text, text_mask, [5, 3], 3, compile=True, **kwargs)
        expected = pipeline.sample(text, text_mask, [5, 3], 3, **kwargs)
        assert mx.allclose(updated, expected, atol=3e-5, rtol=3e-5)
        assert not mx.allclose(updated, compiled, atol=1e-5)
        model.train()
        with pytest.raises(ValueError, match="model.eval"):
            pipeline.sample(text, text_mask, [5, 3], 3, compile=True, **kwargs)

    def test_lazy_native_text_to_skeleton_and_seed(self):
        from mlx_vlm.models.kimodo.language import LLM2VecEncoder
        from mlx_vlm.models.kimodo.pipeline import KimodoPipeline

        class Tokenizer:
            eos_token = "eos"
            bos_token_id = 1

            def __call__(self, texts, **kwargs):
                single = isinstance(texts, str)
                texts = [texts] if single else texts
                rows = [[1] + [2 + ord(c) % 28 for c in text[:12]] for text in texts]
                width = max(map(len, rows))
                ids = [[0] * (width - len(row)) + row for row in rows]
                mask = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
                return {
                    "input_ids": ids[0] if single else ids,
                    "attention_mask": mask[0] if single else mask,
                }

        model = self.model()
        encoder = LLM2VecEncoder(
            dict(
                model_type="llama",
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                vocab_size=32,
                rms_norm_eps=1e-5,
            ),
            tokenizer=Tokenizer(),
        )
        encoder.eval()
        pipeline = KimodoPipeline(model, encoder)
        with (
            patch.object(mx, "eval", side_effect=AssertionError("eager evaluation")),
            patch.object(
                mx, "synchronize", side_effect=AssertionError("GPU synchronization")
            ),
            patch.object(np, "array", side_effect=AssertionError("host transfer")),
        ):
            result = pipeline.generate("a person walks", 5, 4, seed=19)
        assert result["posed_joints"].shape == (5, 3, 3)
        assert result["global_rot_mats"].shape == (5, 3, 3, 3)
        assert all(isinstance(value, mx.array) for value in result.values())
        assert mx.all(mx.isfinite(result["posed_joints"]))
        same = pipeline.generate("a person walks", 5, 4, seed=19)
        assert mx.array_equal(result["posed_joints"], same["posed_joints"])
        different = pipeline.generate("a person walks", 5, 4, seed=20)
        assert not mx.array_equal(result["posed_joints"], different["posed_joints"])

    def test_shared_checkpoint_loader(self, tmp_path):
        from mlx_vlm.models.kimodo import ModelConfig
        from mlx_vlm.utils import load, save_weights

        model = self.model()
        assets = tmp_path / "assets"
        assets.mkdir()
        skeleton = dict(
            neutral_joints=[[0, 0, 0], [-0.2, 0, 0], [0.2, 0, 0]],
            joint_parents=[-1, 0, 0],
            hip_joint_idx=[2, 1],
        )
        (assets / "skeleton.json").write_text(json.dumps(skeleton))
        rep = model.motion_rep
        mx.save_safetensors(
            str(assets / "motion_stats.safetensors"),
            {
                name + "." + stat: getattr(getattr(rep, name + "_stats"), stat)
                for name in ("global_root", "local_root", "body")
                for stat in ("mean", "std")
            },
        )
        (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))
        save_weights(tmp_path, model)
        loaded, processor = load(str(tmp_path), lazy=True)
        assert isinstance(loaded.config, ModelConfig)
        assert processor("walk")["prompts"] == "walk"
        assert set(dict(tree_flatten(loaded.parameters()))) == set(
            dict(tree_flatten(model.parameters()))
        )
        assert all(
            mx.array_equal(value, dict(tree_flatten(loaded.parameters()))[name])
            for name, value in tree_flatten(model.parameters())
        )


class TestKimodoCLI:
    @pytest.mark.parametrize(
        "prompts, flags, batch",
        [
            (["Walk"], [], 1),
            (["Walk", "Wave"], [], 2),
            (["Walk"], ["--num-samples", "3"], 3),
            (["Walk", "Wave"], ["--num-samples", "2"], 2),
            (["Walk", "Wave"], ["--multi-prompt"], 1),
            (["Walk", "Wave"], ["--multi-prompt", "--num-samples", "2"], 2),
            (["Walk"], ["--compile"], 1),
            (["Walk", "Wave"], ["--multi-prompt", "--compile"], 1),
        ],
    )
    def test_sample_count_inference(self, prompts, flags, batch, tmp_path):
        cli = importlib.import_module("mlx_vlm.models.kimodo.generate")
        pipeline = TestKimodoMultiPrompt.pipeline()
        output = tmp_path / "motion.npz"

        def sample(text_features, text_mask, lengths, steps, **kwargs):
            rep = pipeline.motion_rep
            local = mx.broadcast_to(
                mx.eye(3), (len(lengths), max(lengths), rep.skeleton.nbjoints, 3, 3)
            )
            roots = mx.zeros((len(lengths), max(lengths), 3))
            return rep(local, roots, lengths=mx.array(lengths))

        with (
            patch(
                "sys.argv",
                [
                    "kimodo.generate",
                    "--model",
                    "unused",
                    "--prompt",
                    *prompts,
                    "--duration",
                    "0.2",
                    "--output",
                    str(output),
                    *flags,
                ],
            ),
            patch.object(cli, "load", return_value=pipeline.model),
            patch.object(cli, "KimodoPipeline", return_value=pipeline),
            patch.object(pipeline, "sample", side_effect=sample) as sampling,
        ):
            cli.main()
        assert all(
            call.kwargs["compile"] == ("--compile" in flags)
            for call in sampling.call_args_list
        )
        with np.load(output) as saved:
            frames = round(0.2 * pipeline.model.config.fps)
            if "--multi-prompt" in flags:
                frames *= len(prompts)
            assert saved["posed_joints"].shape == (batch, frames, 7, 3)


class TestKimodoMotion:
    @staticmethod
    def skeleton():
        from mlx_vlm.models.kimodo.motion import Skeleton

        return Skeleton(
            neutral_joints=[
                [0, 0, 0],
                [-0.2, 0, 0],
                [0.2, 0, 0],
                [-0.5, 0.4, 0],
                [0.5, 0.4, 0],
                [-0.2, -1, 0],
                [0.2, -1, 0],
            ],
            joint_parents=[-1, 0, 0, 1, 2, 1, 2],
            joint_names=[
                "Hips",
                "LeftLeg",
                "RightLeg",
                "LeftHand",
                "RightHand",
                "LeftFoot",
                "RightFoot",
            ],
            hip_joint_idx=[2, 1],
            left_hand_joint_idx=[3],
            right_hand_joint_idx=[4],
            left_foot_joint_idx=[5, 5],
            right_foot_joint_idx=[6, 6],
        )

    def test_forward_kinematics_and_representation_roundtrip(self):
        from mlx_vlm.models.kimodo.geometry import angle_to_y_rotation_matrix
        from mlx_vlm.models.kimodo.motion import ArdyMotionRep

        skeleton = self.skeleton()
        rotation = angle_to_y_rotation_matrix(mx.array(0.7))
        local = mx.broadcast_to(mx.eye(3), (2, 5, 7, 3, 3))
        local = mx.array(local)
        local[..., 0, :, :] = rotation
        root = mx.random.normal((2, 5, 3))
        global_rots, positions, no_root = skeleton.fk(local, root)
        expected = (rotation @ skeleton.neutral_joints[..., None])[..., 0]
        assert mx.allclose(no_root, expected, atol=1e-6)
        assert mx.allclose(positions[..., 0, :], root)
        rep = ArdyMotionRep(skeleton)
        features = rep(local, root, to_normalize=True, lengths=mx.array([5, 3]))
        decoded = rep.inverse(features, is_normalized=True, return_all=True)
        assert mx.allclose(decoded["global_rot_mats"], global_rots, atol=1e-6)
        assert mx.allclose(decoded["posed_joints"], positions, atol=1e-6)
        assert mx.allclose(decoded["all_positions"]["positions"], positions, atol=1e-6)
        assert mx.allclose(
            skeleton.global_rots_to_local_rots(global_rots), local, atol=1e-6
        )

    def test_variable_lengths_and_heading_wrap(self):
        from mlx_vlm.models.kimodo.motion import KimodoMotionRep

        rep = KimodoMotionRep(self.skeleton(), fps=10)
        angles = mx.array([[3.0, -3.0, -2.8, -2.7], [0.0, 0.1, 0.3, 0.8]])
        positions = mx.stack(
            (mx.broadcast_to(mx.arange(4), (2, 4)), mx.ones((2, 4)), mx.zeros((2, 4))),
            axis=-1,
        )
        root = mx.concatenate(
            (positions, mx.cos(angles)[..., None], mx.sin(angles)[..., None]), axis=-1
        )
        local = rep.global_root_to_local_root(
            root, normalized=False, lengths=mx.array([4, 2])
        )
        assert mx.allclose(local[0, 0, 0], mx.array((2 * math.pi - 6) * 10), atol=1e-5)
        assert mx.allclose(local[:, :2, 1], mx.array(10.0))
        assert mx.allclose(local[1, 1], local[1, 0])
        assert mx.allclose(local[0, -1], local[0, -2])
        restored = rep.local_root_to_global_root(local, normalized=False)
        assert mx.allclose(restored[0, :, :3], positions[0] - mx.array([0, 0, 0]))
        assert mx.allclose(
            rep.get_root_heading_angle(restored)[0],
            mx.array([0, 2 * math.pi - 6, 2 * math.pi - 5.8, 2 * math.pi - 5.7]),
            atol=1e-5,
        )

    def test_sparse_conditions_preserve_reference_layouts(self):
        from mlx_vlm.models.kimodo.motion import ArdyMotionRep, KimodoMotionRep

        for cls in (KimodoMotionRep, ArdyMotionRep):
            rep = cls(self.skeleton())
            stats = {
                name
                + "."
                + key: (mx.ones((size,)) if key == "std" else mx.full((size,), 2.0))
                for name, size in (
                    ("global_root", 5),
                    ("local_root", 4),
                    ("body", rep.body_dim),
                )
                for key in ("mean", "std")
            }
            rep.set_stats(stats)
            obs, mask = rep.create_conditions(
                {
                    "smooth_root_2d": [mx.array([1])],
                    "global_joints_positions": [mx.array([[1, 0], [1, 3]])],
                },
                {
                    "smooth_root_2d": [mx.array([[4.0, 6.0]])],
                    "global_joints_positions": [
                        mx.array([[4.0, 1.0, 6.0], [5.0, 2.0, 9.0]])
                    ],
                },
                3,
                to_normalize=True,
            )
            assert mask.shape == (3, rep.motion_rep_dim)
            assert mask.dtype == (mx.float32 if rep._ardy else mx.bool_)
            assert mx.all(mask[1, [0, 2]])
            start = (
                rep.slice_dict["local_joints_positions"].start
                + (3 - int(rep._ardy)) * 3
            )
            raw = rep.unnormalize(obs)
            assert mx.allclose(
                raw[1, start : start + 3], mx.array([1.0, 2.0, 3.0]), atol=1e-6
            )
            if rep._ardy:
                assert mx.all(obs[0] == 0)
            else:
                assert mx.allclose(
                    obs[0], rep.normalize(mx.zeros((rep.motion_rep_dim,)))
                )

    def test_native_constraints_and_lazy_batch_adapter(self):
        from unittest.mock import patch

        from mlx_vlm.models.kimodo.motion import KimodoMotionRep

        rep = KimodoMotionRep(self.skeleton())
        constraints = [
            {
                "type": "fullbody",
                "frame_indices": [1],
                "local_joints_rot": [[[0.0, 0.0, 0.0]] * 7],
                "root_positions": [[3.0, 1.0, 4.0]],
            }
        ]
        with (
            patch.object(mx, "eval", side_effect=AssertionError("eager evaluation")),
            patch.object(
                mx, "synchronize", side_effect=AssertionError("GPU synchronization")
            ),
        ):
            observed, mask = rep.create_conditions_from_constraints_batched(
                [constraints, []], [3, 2]
            )
            decoded = rep.inverse(
                observed,
                is_normalized=True,
                posed_joints_from="positions",
                need_local_rots=False,
            )
        assert observed.shape == (2, 3, rep.motion_rep_dim)
        assert mx.all(mask[0, 1, :5])
        assert mx.all(mask[1] == 0)
        assert mx.allclose(
            decoded["posed_joints"][0, 1],
            rep.skeleton.neutral_joints + mx.array([3.0, 1.0, 4.0]),
            atol=1e-6,
        )

    def test_soma_output_expansion(self):
        from mlx_vlm.models.kimodo.motion import Skeleton

        expanded = dict(
            neutral_joints=[[0, 0, 0], [1, 0, 0], [2, 0, 0]], joint_parents=[-1, 0, 1]
        )
        skeleton = Skeleton(
            [[0, 0, 0], [2, 0, 0]],
            [-1, 0],
            output_skeleton=expanded,
            output_joint_indices=[0, 2],
            output_rest_local_rots=mx.broadcast_to(mx.eye(3), (3, 3, 3)),
        )
        local = mx.broadcast_to(mx.eye(3), (1, 2, 3, 3))
        result = skeleton.output_to_SOMASkeleton77(
            {
                "local_rot_mats": local,
                "root_positions": mx.array([[2, 3, 4]]),
                "foot_contacts": mx.array([[True, False, False, True]]),
            }
        )
        assert result["local_rot_mats"].shape == (1, 3, 3, 3)
        assert mx.array_equal(
            result["posed_joints"], mx.array([[[2, 3, 4], [3, 3, 4], [4, 3, 4]]])
        )
        assert mx.array_equal(
            result["foot_contacts"], mx.array([[True, False, False, False, True, True]])
        )


class TestKimodoMultiPrompt:
    @staticmethod
    def pipeline():
        from mlx_vlm.models.kimodo import KimodoPipeline, Model, ModelConfig
        from mlx_vlm.models.kimodo.motion import KimodoMotionRep

        class Encoder:
            def __call__(self, texts):
                return mx.zeros((len(texts), 2, 8)), [2] * len(texts)

        config = ModelConfig(
            latent_dim=16,
            ff_size=24,
            num_layers=1,
            num_heads=4,
            llm_shape=(1, 8),
            num_text_tokens_override=2,
            num_base_steps=8,
            max_position_embeddings=64,
        )
        model = Model(config, KimodoMotionRep(TestKimodoMotion.skeleton()))
        model.eval()
        return KimodoPipeline(model, Encoder())

    def test_single_segment_matches_regular_generation(self):
        pipeline = self.pipeline()
        sequential = pipeline.generate(
            ["Walk"], [7], 3, seed=17, multi_prompt=True, return_features=True
        )
        regular = pipeline.generate("Walk", 7, 3, seed=17, return_features=True)
        assert sequential.shape == regular.shape
        assert mx.allclose(sequential, regular, atol=1e-5)

    @pytest.mark.parametrize("num_samples", [None, 2])
    @pytest.mark.parametrize("with_mask", [False, True])
    @pytest.mark.parametrize(
        "prompts, frames", [(["Walk"], [7]), (["Walk", "Turn"], [7, 6])]
    )
    def test_precomputed_text_conditioning(
        self, num_samples, with_mask, prompts, frames
    ):
        pipeline = self.pipeline()
        pipeline.text_encoder = None
        text = mx.ones((num_samples or 1, 2, 8))
        mask = (
            mx.broadcast_to(mx.array([[True, False]]), text.shape[:2])
            if with_mask
            else None
        )
        with (
            patch.object(
                pipeline, "_encoder", side_effect=AssertionError("text encoder loaded")
            ),
            patch.object(pipeline, "sample", wraps=pipeline.sample) as sample,
            patch.object(mx, "eval", side_effect=AssertionError("eager evaluation")),
        ):
            result = pipeline.generate(
                prompts,
                frames,
                3,
                multi_prompt=True,
                num_samples=num_samples,
                text_features=text,
                text_mask=mask,
                return_features=True,
            )
        assert sample.call_count == len(prompts)
        for call in sample.call_args_list:
            assert call.args[0] is text
            expected_mask = mask if with_mask else mx.ones(text.shape[:2], mx.bool_)
            assert mx.array_equal(call.args[1], expected_mask)
        shape = (sum(frames), pipeline.motion_rep.motion_rep_dim)
        assert result.shape == (shape if num_samples is None else (num_samples, *shape))
        assert mx.all(mx.isfinite(result))

    def test_constraints_and_three_segment_stitching_stay_lazy(self):
        pipeline = self.pipeline()
        constraints = [
            {
                "type": "root2d",
                "frame_indices": [0, 5, 11, 17],
                "smooth_root_2d": [[0, 0], [0, 1], [1, 2], [2, 2]],
            }
        ]
        with (
            patch.object(mx, "eval", side_effect=AssertionError("eager evaluation")),
            patch.object(
                mx, "synchronize", side_effect=AssertionError("GPU synchronization")
            ),
            patch.object(np, "array", side_effect=AssertionError("host transfer")),
        ):
            result = pipeline.generate(
                ["Walk", "Turn", "Stop"],
                [6, 7, 5],
                3,
                seed=17,
                multi_prompt=True,
                num_samples=2,
                num_transition_frames=3,
                constraint_lst=constraints,
            )
        assert result["posed_joints"].shape == (2, 18, 7, 3)
        assert result["root_positions"].shape == (2, 18, 3)
        assert all(isinstance(value, mx.array) for value in result.values())
        assert all(mx.all(mx.isfinite(value)) for value in result.values())

    def test_transition_length_validation(self):
        pipeline = self.pipeline()
        with pytest.raises(ValueError, match="transition"):
            pipeline.generate(
                ["Walk", "Turn"], [2, 4], 1, multi_prompt=True, num_transition_frames=3
            )


class TestKimodoTrajectory:
    @staticmethod
    def reference_solve(reference, observations, margins, iterations=100):
        """Independent dense double-precision form of the source ADMM system."""
        frames = reference.shape[0]
        identity = np.eye(frames)
        velocity = np.diff(identity, axis=0)
        acceleration = np.diff(identity, n=2, axis=0)
        normal = (
            float(np.float32(0.001)) * identity
            + velocity.T @ velocity
            + 10.0 * acceleration.T @ acceleration
        )
        hard, soft = margins == 0, margins > 0
        if not np.any(hard | soft):
            return reference.copy()
        free = ~hard
        rho = float(np.float32(0.5) * np.sqrt(np.float32(normal.diagonal().max())))
        system = normal + rho * np.diag(soft.astype(float))
        inverse = np.linalg.inv(system[np.ix_(free, free)])
        rhs = normal @ (reference - np.where(hard[:, None], observations, 0))
        result = reference.copy()
        z = reference[soft].copy()
        u = np.zeros_like(z)
        for _ in range(iterations if np.any(soft) else 1):
            current_rhs = rhs.copy()
            current_rhs[soft] += rho * (z - u)
            result[free] = inverse @ current_rhs[free]
            result[hard] = observations[hard]
            difference = result[soft] + u - observations[soft]
            norm = np.linalg.norm(difference, axis=-1, keepdims=True)
            z = observations[soft] + difference * np.minimum(
                1.0, margins[soft, None] / np.maximum(norm, 1e-30)
            )
            u += result[soft] - z
        return result

    @pytest.mark.parametrize(
        "frames,channels", [(1, 1), (2, 3), (3, 6), (24, 2), (40, 65)]
    )
    @pytest.mark.parametrize(
        "constraint", ["free", "hard", "margin", "mixed", "allhard"]
    )
    def test_gpu_band_solve_matches_independent_dense_admm(
        self, frames, channels, constraint
    ):
        from mlx_vlm.models.kimodo.trajectory import trajectory_correct

        rng = np.random.default_rng(71)
        reference = rng.normal(size=(frames, channels)).astype(np.float32)
        observations = rng.normal(size=(frames, channels)).astype(np.float32)
        margins = np.full(frames, -1.0, dtype=np.float32)
        selected = np.arange(frames) % 7 == 0
        if constraint in ("hard", "mixed"):
            margins[selected] = 0
        if constraint == "margin":
            margins[selected] = 0.2
        if constraint == "mixed":
            margins[np.arange(frames) % 7 == 3] = 0.2
        if constraint == "allhard":
            margins[:] = 0
        actual = trajectory_correct(
            mx.array(reference), mx.array(observations), mx.array(margins)
        )
        expected = self.reference_solve(
            reference.astype(float), observations.astype(float), margins
        )
        np.testing.assert_allclose(np.array(actual), expected, atol=5e-4, rtol=5e-5)
        np.testing.assert_array_equal(
            np.array(actual)[margins == 0], observations[margins == 0]
        )
        if constraint == "free":
            np.testing.assert_array_equal(np.array(actual), reference)

    def test_broadcast_batch_and_solver_build_remain_lazy(self):
        from mlx_vlm.models.kimodo.trajectory import smooth_channels, trajectory_correct

        reference = mx.arange(48, dtype=mx.float32).reshape(2, 1, 8, 3) * 0.01
        observations = mx.zeros((8, 3))
        margins = mx.array([[0, -1, -1, 0.2, -1, -1, -1, 0], [-1] * 8])
        with (
            patch.object(mx, "eval", side_effect=AssertionError("eager evaluation")),
            patch.object(
                mx, "synchronize", side_effect=AssertionError("GPU synchronization")
            ),
            patch.object(np, "array", side_effect=AssertionError("host transfer")),
        ):
            result = trajectory_correct(reference, observations, margins)
            smoothed = smooth_channels(result, margins >= 0)
        assert result.shape == (2, 2, 8, 3)
        assert smoothed.shape == result.shape
        assert mx.all(mx.isfinite(smoothed))
        assert mx.array_equal(result[:, 1], reference[:, 0])

    def test_boundary_smoothing_matches_time_ordered_in_place_scan(self):
        from mlx_vlm.models.kimodo.trajectory import smooth_channels

        rng = np.random.default_rng(18)
        reference = rng.normal(size=(2, 13, 6)).astype(np.float32)
        mask = np.array(
            [
                [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1],
                [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            ],
            dtype=np.float32,
        )
        expected = reference.copy()
        for b in range(2):
            for t in range(13):
                previous, following = max(0, t - 1), min(12, t + 1)
                if t > 0 and mask[b, t] > 0 and mask[b, previous] == 0:
                    expected[b, t] = 0.5 * (
                        expected[b, previous] + expected[b, following]
                    )
                if mask[b, t] > 0 and mask[b, following] == 0:
                    expected[b, t] = 0.5 * (
                        expected[b, previous] + expected[b, following]
                    )
        actual = smooth_channels(mx.array(reference), mx.array(mask))
        np.testing.assert_array_equal(np.array(actual), expected)


class TestKimodoContactCleanup:
    @staticmethod
    def rig():
        from mlx_vlm.models.kimodo.contact_cleanup import CleanupRig

        return CleanupRig(
            (-1, 0, 1),
            mx.array([[0, 0, 0], [0, -1, 0.1], [0, -1, -0.1]], mx.float32),
        )

    @pytest.mark.parametrize("one_bone", [False, True])
    def test_gpu_contact_intervals_match_source_boundary_rules(self, one_bone):
        from mlx_vlm.models.kimodo.contact_cleanup import contact_intervals

        contacts = np.array(
            [
                [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
                [0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
            ],
            dtype=np.float32,
        )
        mask = np.array(
            [
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        expected_active = np.zeros_like(contacts)
        expected_frames = np.broadcast_to(np.arange(12), (2, 12)).copy()
        expected_boundary = np.zeros_like(mask)
        for b in range(2):
            start = None
            for frame in range(13):
                active = frame < 12 and contacts[b, frame] > 0.5 and not mask[b, frame]
                if active and start is None:
                    start = frame
                elif not active and start is not None:
                    sc = start > 0 and mask[b, start - 1]
                    ec = frame < 12 and mask[b, frame]
                    if not (sc or ec if one_bone else sc and ec):
                        representative = (
                            start - 1 if sc else frame if ec else (start + frame) // 2
                        )
                        expected_active[b, start:frame] = 1
                        expected_frames[b, start:frame] = representative
                        expected_boundary[b, start:frame] = sc or ec
                    start = None
        active, frames, boundary = contact_intervals(
            mx.array(contacts), mx.array(mask), one_bone=one_bone
        )
        np.testing.assert_array_equal(np.array(active), expected_active)
        np.testing.assert_array_equal(np.array(frames), expected_frames)
        np.testing.assert_array_equal(np.array(boundary), expected_boundary)

    def test_metal_two_bone_reaches_target_and_zero_weight_is_identity(self):
        from mlx_vlm.models.kimodo.ik import one_bone_ik, two_bone_ik
        from mlx_vlm.models.kimodo.ik_kernels import joint_global

        rig = self.rig()
        rotations = mx.broadcast_to(mx.eye(3), (2, 4, 3, 3, 3))
        roots = mx.zeros((2, 4, 3))
        target = mx.broadcast_to(mx.array([0.3, -1.7, 0.2]), roots.shape)
        corrected = two_bone_ik(rotations, roots, rig, 2, target)
        _, position = joint_global(corrected, roots, rig, 2)
        np.testing.assert_allclose(np.array(position), np.array(target), atol=3e-6)
        np.testing.assert_array_equal(
            np.array(two_bone_ik(rotations, roots, rig, 2, target, weight=0.0)),
            np.array(rotations),
        )
        np.testing.assert_array_equal(
            np.array(one_bone_ik(rotations, roots, rig, 2, target, weight=0.0)),
            np.array(rotations),
        )

    @pytest.mark.parametrize("target", [[0, -4, 0], [0, 0, 0], [0, 2, 0]])
    def test_metal_solvers_remain_finite_at_collinear_or_unreachable_targets(
        self, target
    ):
        from mlx_vlm.models.kimodo.contact_cleanup import CleanupRig
        from mlx_vlm.models.kimodo.ik import one_bone_ik, two_bone_ik

        rig = CleanupRig(
            (-1, 0, 1), mx.array([[0, 0, 0], [0, -1, 0], [0, -1, 0]], mx.float32)
        )
        rotations = mx.broadcast_to(mx.eye(3), (1, 3, 3, 3, 3))
        roots = mx.zeros((1, 3, 3))
        targets = mx.broadcast_to(mx.array(target, mx.float32), roots.shape)
        for solver in (one_bone_ik, two_bone_ik):
            result = solver(rotations, roots, rig, 2, targets)
            assert mx.all(mx.isfinite(result))
            gram = mx.swapaxes(result, -1, -2) @ result
            assert mx.allclose(gram, mx.eye(3), atol=2e-5)

    def test_effector_pin_restores_global_rotation_and_builds_without_evaluation(self):
        from mlx_vlm.models.kimodo.contact_cleanup import (
            TWO_BONE,
            ContactInfo,
            do_effector_ik,
        )
        from mlx_vlm.models.kimodo.geometry import angle_to_y_rotation_matrix
        from mlx_vlm.models.kimodo.ik_kernels import joint_global

        rig = self.rig()
        rotations = mx.broadcast_to(mx.eye(3), (1, 5, 3, 3, 3))
        roots = mx.zeros((1, 5, 3))
        target_rotations = mx.array(rotations)
        target_rotations[..., 2, :, :] = angle_to_y_rotation_matrix(
            mx.full((1, 5), 0.4)
        )
        target_roots = mx.broadcast_to(mx.array([0.2, 0.1, 0.05]), roots.shape)
        pins = [ContactInfo(2, TWO_BONE, mx.ones((1, 5), mx.bool_))]
        with (
            patch.object(mx, "eval", side_effect=AssertionError("eager evaluation")),
            patch.object(
                mx, "synchronize", side_effect=AssertionError("GPU synchronization")
            ),
            patch.object(np, "array", side_effect=AssertionError("host transfer")),
        ):
            result = do_effector_ik(
                rotations,
                roots,
                target_rotations,
                target_roots,
                mx.zeros((1, 5), mx.bool_),
                pins,
                rig,
            )
        actual_rotation, _ = joint_global(result, roots, rig, 2)
        target_rotation, _ = joint_global(target_rotations, target_roots, rig, 2)
        np.testing.assert_allclose(
            np.array(actual_rotation), np.array(target_rotation), atol=3e-6
        )


class TestKimodoPostprocess:
    @staticmethod
    def skeleton():
        from mlx_vlm.models.kimodo.motion import Skeleton

        return Skeleton(
            name="somaskel_test",
            joint_parents=[-1, 0, 1, 2, 3, 4, 1, 6, 7, 8, 0, 10, 11, 12, 0, 14, 15, 16],
            neutral_joints=[
                [0, 0, 0],
                [0, 0.2, 0],
                [-0.2, 0.4, 0],
                [-0.3, 0.4, 0],
                [-0.6, 0.35, 0],
                [-0.8, 0.4, 0],
                [0.2, 0.4, 0],
                [0.3, 0.4, 0],
                [0.6, 0.35, 0],
                [0.8, 0.4, 0],
                [-0.1, -0.05, 0],
                [-0.1, -0.5, 0.18],
                [-0.1, -0.95, 0],
                [-0.1, -1, 0.15],
                [0.1, -0.05, 0],
                [0.1, -0.5, 0.18],
                [0.1, -0.95, 0],
                [0.1, -1, 0.15],
            ],
            joint_names=[
                "Hips",
                "Spine",
                "LeftShoulder",
                "LeftArm",
                "LeftForeArm",
                "LeftHand",
                "RightShoulder",
                "RightArm",
                "RightForeArm",
                "RightHand",
                "LeftLeg",
                "LeftKnee",
                "LeftFoot",
                "LeftToe",
                "RightLeg",
                "RightKnee",
                "RightFoot",
                "RightToe",
            ],
            hip_joint_idx=[14, 10],
            left_hand_joint_idx=[5],
            right_hand_joint_idx=[9],
            left_foot_joint_idx=[12, 13],
            right_foot_joint_idx=[16, 17],
        )

    def motion(self, frames=12, batch=1):
        skel = self.skeleton()
        local = mx.broadcast_to(mx.eye(3), (batch, frames, skel.nbjoints, 3, 3))
        time = mx.arange(frames, dtype=mx.float32)
        roots = mx.broadcast_to(
            mx.stack((0.01 * time, 1.05 + time * 0, 0.015 * time), axis=-1),
            (batch, frames, 3),
        )
        contacts = mx.ones((batch, frames, 4), mx.bool_)
        return skel, local, roots, contacts

    def test_native_constraint_targets_and_crop_stay_lazy(self):
        from mlx_vlm.models.kimodo.postprocess import (
            crop_constraints,
            prepare_constraint_targets,
        )

        skel, local, roots, contacts = self.motion()
        global_rots, positions, _ = skel.fk(local, roots)
        constraints = [
            dict(
                type="fullbody",
                frame_indices=[3],
                global_joints_rots=global_rots[0, 3:4],
                global_joints_positions=positions[0, 3:4],
                smooth_root_2d=mx.array([[9.0, 8.0]]),
            ),
            dict(
                type="root2d",
                frame_indices=[1, 3, 99],
                smooth_root_2d=[[2.0, 3.0], [4.0, 5.0], [99.0, 99.0]],
            ),
        ]
        with patch.object(mx, "eval", side_effect=AssertionError("eager evaluation")):
            targets, rotations, masks = prepare_constraint_targets(
                constraints, skel, 12, 1
            )
            cropped = prepare_constraint_targets(
                crop_constraints(constraints, 2, 8), skel, 6, 1
            )
        assert mx.array_equal(targets[0, 1], mx.array([2.0, 0.0, 3.0]))
        assert mx.allclose(targets[0, 3], roots[0, 3])
        assert masks["FullBody"][0, 3]
        assert masks["Root"][0, 1] and masks["Root"][0, 3]
        assert mx.allclose(cropped[0][0, 1], roots[0, 3])
        assert not cropped[2]["Root"][0, -1]

    @pytest.mark.parametrize(
        "joint_names",
        [
            ["LeftHand"],
            ["RightHand"],
            ["LeftFoot"],
            ["RightFoot"],
            ["LeftHand", "RightFoot", "Hips"],
        ],
    )
    def test_generic_end_effector_masks_activate_pins(self, joint_names):
        from mlx_vlm.models.kimodo.postprocess import (
            _contacts_and_pins,
            create_cleanup_rig,
            prepare_constraint_targets,
        )

        skel, local, roots, contacts = self.motion(batch=2)
        global_rots, positions, _ = skel.fk(local, roots)
        pose = dict(
            frame_indices=[2, 4, 99],
            global_joints_rots=global_rots[0, :3],
            global_joints_positions=positions[0, :3],
        )
        fullbody = dict(
            type="fullbody",
            frame_indices=[4],
            global_joints_rots=global_rots[0, 4:5],
            global_joints_positions=positions[0, 4:5],
        )
        constraints = [
            [dict(pose, type="end-effector", joint_names=joint_names), fullbody],
            [],
        ]
        with patch.object(mx, "eval", side_effect=AssertionError("eager evaluation")):
            _, _, masks = prepare_constraint_targets(constraints, skel, 12, 2)
            _, pins = _contacts_and_pins(
                skel, contacts, masks, create_cleanup_rig(skel)
            )
        for name in ("LeftHand", "RightHand", "LeftFoot", "RightFoot", "Root"):
            active = ("Hips" if name == "Root" else name) in joint_names
            expected = mx.array(
                [[active and i in (2, 4) for i in range(12)], [False] * 12]
            )
            assert mx.array_equal(masks[name], expected)
        for name, pin in zip(("LeftHand", "RightHand", "LeftFoot", "RightFoot"), pins):
            assert mx.array_equal(pin.mask, masks[name] & ~masks["FullBody"])

    def test_generic_end_effector_cleanup_matches_named_constraint(self):
        from mlx_vlm.models.kimodo import post_process_motion

        skel, local, roots, contacts = self.motion()
        target_roots = roots + mx.array([0.02, 0, 0])
        global_rots, positions, _ = skel.fk(local, target_roots)
        pose = dict(
            frame_indices=list(range(12)),
            global_joints_rots=global_rots[0],
            global_joints_positions=positions[0],
        )
        generic = post_process_motion(
            local,
            roots,
            contacts,
            skel,
            [dict(pose, type="end-effector", joint_names=["LeftHand"])],
        )
        named = post_process_motion(
            local, roots, contacts, skel, [dict(pose, type="left-hand")]
        )
        for name, value in generic.items():
            assert mx.allclose(value, named[name], atol=1e-6)
        initial_hand = skel.fk(local, roots)[1][..., 5, :]
        target_hand = positions[..., 5, :]
        initial_error = mx.linalg.norm(initial_hand - target_hand, axis=-1)
        corrected_error = mx.linalg.norm(
            generic["posed_joints"][..., 5, :] - target_hand, axis=-1
        )
        assert mx.all(corrected_error < initial_error)

    def test_cleanup_batch_matches_individual_and_preserves_inputs(self):
        from mlx_vlm.models.kimodo import post_process_motion

        skel, local, roots, contacts = self.motion(batch=2)
        before = mx.array(local)
        with (
            patch.object(mx, "eval", side_effect=AssertionError("eager evaluation")),
            patch.object(mx, "synchronize", side_effect=AssertionError("synchronize")),
            patch.object(np, "array", side_effect=AssertionError("host transfer")),
        ):
            corrected = post_process_motion(local, roots, contacts, skel)
        single = post_process_motion(local[0], roots[0], contacts[0], skel)
        assert mx.array_equal(local, before)
        for name, value in corrected.items():
            assert isinstance(value, mx.array)
            assert mx.all(mx.isfinite(value))
            assert mx.allclose(value[0], single[name], atol=2e-6)
            assert mx.allclose(value[0], value[1], atol=2e-6)
        assert mx.array_equal(corrected["root_positions"], roots)
        before_joints = skel.fk(local, roots)[1]
        before_drift = mx.mean(
            mx.abs(before_joints[:, 1:, 12, 0] - before_joints[:, :-1, 12, 0])
        )
        after_drift = mx.mean(
            mx.abs(
                corrected["posed_joints"][:, 1:, 12, 0]
                - corrected["posed_joints"][:, :-1, 12, 0]
            )
        )
        assert after_drift < before_drift * 0.5

    def test_hard_fullbody_targets_are_preserved(self):
        from mlx_vlm.models.kimodo import post_process_motion

        skel, local, roots, contacts = self.motion()
        target_root = roots + mx.array([0.02, 0.01, -0.025])
        global_rots, positions, _ = skel.fk(local, target_root)
        constraints = [
            dict(
                type="fullbody",
                frame_indices=list(range(12)),
                global_joints_rots=global_rots[0],
                global_joints_positions=positions[0],
            )
        ]
        corrected = post_process_motion(local, roots, contacts, skel, constraints)
        assert mx.array_equal(corrected["root_positions"], target_root)
        assert mx.allclose(corrected["local_rot_mats"], local, atol=1e-6)
        assert mx.allclose(corrected["posed_joints"], positions, atol=1e-6)

    def test_pipeline_cleanup_is_opt_in_and_lazy(self):
        from mlx_vlm.models.kimodo import KimodoPipeline, Model, ModelConfig
        from mlx_vlm.models.kimodo.motion import KimodoMotionRep
        from mlx_vlm.models.kimodo.postprocess import post_process_motion

        skel, local, roots, contacts = self.motion()
        rep = KimodoMotionRep(skel)
        config = ModelConfig(
            latent_dim=16,
            ff_size=24,
            num_layers=1,
            num_heads=4,
            llm_shape=(1, 16),
            num_base_steps=16,
            max_position_embeddings=64,
            num_text_tokens_override=3,
        )
        pipeline = KimodoPipeline(Model(config, rep))
        features = rep(local, roots, lengths=mx.array([12]), to_normalize=True)
        text = mx.zeros((1, 1, 16))
        with patch.object(pipeline, "sample", return_value=features):
            plain = pipeline.generate("walk", 12, 4, text_features=text)
            with patch(
                "mlx_vlm.models.kimodo.postprocess.post_process_motion",
                wraps=post_process_motion,
            ) as cleanup:
                disabled = pipeline.generate(
                    "walk", 12, 4, text_features=text, post_processing=False
                )
                cleanup.assert_not_called()
                with (
                    patch.object(
                        mx, "eval", side_effect=AssertionError("eager evaluation")
                    ),
                    patch.object(
                        mx, "synchronize", side_effect=AssertionError("synchronize")
                    ),
                ):
                    corrected = pipeline.generate(
                        "walk", 12, 4, text_features=text, post_processing=True
                    )
                cleanup.assert_called_once()
        assert mx.array_equal(plain["posed_joints"], disabled["posed_joints"])
        assert corrected["posed_joints"].shape == (12, skel.nbjoints, 3)
        assert mx.all(mx.isfinite(corrected["posed_joints"]))

    @pytest.mark.parametrize("multi_prompt", [False, True])
    @pytest.mark.parametrize("threshold", [0.0, 0.5, 0.9, 1.0])
    def test_pipeline_cleanup_uses_contact_threshold(self, multi_prompt, threshold):
        from mlx_vlm.models.kimodo import KimodoPipeline, Model, ModelConfig
        from mlx_vlm.models.kimodo.motion import KimodoMotionRep

        skel = self.skeleton()
        rep = KimodoMotionRep(skel)
        config = ModelConfig(
            latent_dim=16,
            ff_size=24,
            num_layers=1,
            num_heads=4,
            llm_shape=(1, 16),
            num_base_steps=16,
            max_position_embeddings=64,
            num_text_tokens_override=3,
        )
        pipeline = KimodoPipeline(Model(config, rep))
        scores = mx.array([0.0, 0.4, 0.6, 1.0])

        def sample(text_features, text_mask, lengths, steps, **kwargs):
            _, local, roots, _ = self.motion(frames=max(lengths), batch=len(lengths))
            features = rep(local, roots, lengths=mx.array(lengths), to_normalize=False)
            features = mx.array(features)
            features[..., rep.slice_dict["foot_contacts"]] = scores
            return rep.normalize(features)

        def cleanup(local, roots, *args, **kwargs):
            return {"local_rot_mats": local, "root_positions": roots}

        with (
            patch.object(pipeline, "sample", side_effect=sample),
            patch(
                "mlx_vlm.models.kimodo.postprocess.post_process_motion",
                side_effect=cleanup,
            ) as clean,
        ):
            pipeline.generate(
                ["Walk", "Wave"] if multi_prompt else "Walk",
                12,
                4,
                text_features=mx.zeros((1, 1, 16)),
                multi_prompt=multi_prompt,
                post_processing=True,
                contact_threshold=threshold,
            )
        assert clean.call_count == (2 if multi_prompt else 1)
        for call in clean.call_args_list:
            labels = call.args[2]
            assert mx.array_equal(
                labels, mx.broadcast_to(scores > threshold, labels.shape)
            )
            assert call.kwargs["contact_threshold"] == threshold

    @pytest.mark.parametrize("unbatched", [False, True])
    def test_compiled_cleanup_tracks_targets_masks_and_options(self, unbatched):
        from mlx_vlm.models.kimodo import post_process_motion

        skel, local, roots, _ = self.motion(batch=2)
        contacts = mx.broadcast_to(mx.array([0.6, 0.1, 0.8, 0.9]), (2, 12, 4))
        target_roots = roots + mx.array([0.02, 0.01, -0.02])
        global_rots, positions, _ = skel.fk(local, target_roots)
        pose = dict(
            frame_indices=list(range(12)),
            global_joints_rots=global_rots[0],
            global_joints_positions=positions[0],
        )
        if unbatched:
            local, roots, contacts = local[0], roots[0], contacts[0]
        cases = [
            (None, 0.5, 0.04),
            ([dict(pose, type="end-effector", joint_names=["LeftHand"])], 0.9, 0.0),
            ([dict(pose, type="fullbody")], 0.1, 0.1),
        ]
        for constraints, threshold, margin in cases:
            kwargs = dict(
                constraint_lst=constraints,
                contact_threshold=threshold,
                root_margin=margin,
            )
            eager = post_process_motion(local, roots, contacts, skel, **kwargs)
            with (
                patch.object(
                    mx, "eval", side_effect=AssertionError("eager evaluation")
                ),
                patch.object(
                    mx, "synchronize", side_effect=AssertionError("synchronize")
                ),
            ):
                compiled = post_process_motion(
                    local, roots, contacts, skel, compile=True, **kwargs
                )
            for name, value in eager.items():
                assert mx.allclose(compiled[name], value, atol=1e-5, rtol=1e-5)
        assert mx.array_equal(
            compiled["root_positions"], target_roots[0] if unbatched else target_roots
        )
        # A new duration must retrace shape-dependent kernels.
        short = post_process_motion(
            local[..., :1, :, :, :],
            roots[..., :1, :],
            contacts[..., :1, :],
            skel,
            compile=True,
        )
        assert mx.all(mx.isfinite(short["posed_joints"]))

    def test_short_motion_and_validation(self):
        from mlx_vlm.models.kimodo import post_process_motion

        skel, local, roots, contacts = self.motion(frames=1)
        result = post_process_motion(local, roots, contacts, skel)
        assert mx.all(mx.isfinite(result["posed_joints"]))
        with pytest.raises(ValueError):
            post_process_motion(local, roots, contacts, skel, root_margin=-0.1)


class TestKimodoCleanupReference:
    @pytest.mark.parametrize("compile", [False, True])
    @pytest.mark.parametrize(
        "double_ankle, expected",
        [
            (
                False,
                [
                    [-0.04, 0.1, 0.09],
                    [-0.04, 0.02, 0.22638208],
                    [0.16, 0.1, 0.09],
                    [0.16, 0.02, 0.22638208],
                ],
            ),
            (
                True,
                [
                    [-0.49849010, 0.43295389, -0.30248523],
                    [-0.59328073, 0.30865681, -0.27871582],
                    [0.21368122, 0.23422104, 0.14068779],
                    [0.17916445, 0.08811975, 0.19031662],
                ],
            ),
        ],
    )
    def test_contact_cleanup_matches_original_cpp_oracle(
        self, double_ankle, expected, compile
    ):
        from mlx_vlm.models.kimodo import post_process_motion

        # Fixed source MotionCorrection outputs; no source extension is needed in CI.
        skeleton, rotations, roots, contacts = TestKimodoPostprocess().motion()
        if double_ankle:
            skeleton.name = "g1skel34"
        result = post_process_motion(
            rotations, roots, contacts, skeleton, compile=compile
        )
        selected = result["posed_joints"][0, 5, mx.array([12, 13, 16, 17])]
        np.testing.assert_allclose(np.array(selected), expected, atol=4e-5, rtol=1e-5)
        np.testing.assert_array_equal(
            np.array(result["root_positions"]), np.array(roots)
        )


class TestKimodoCleanupTransitions:
    @pytest.mark.parametrize("compile", [False, True])
    def test_each_segment_is_cleaned_with_its_transition_history(self, compile):
        from mlx_vlm.models.kimodo import (
            KimodoPipeline,
            Model,
            ModelConfig,
            post_process_motion,
        )
        from mlx_vlm.models.kimodo.motion import KimodoMotionRep

        fixture = TestKimodoPostprocess()
        skeleton = fixture.skeleton()
        rep = KimodoMotionRep(skeleton)
        config = ModelConfig(
            latent_dim=16,
            ff_size=24,
            num_layers=1,
            num_heads=4,
            llm_shape=(1, 16),
            num_base_steps=16,
            max_position_embeddings=64,
            num_text_tokens_override=3,
        )
        encoder = lambda texts: (mx.zeros((len(texts), 1, 16)), [1] * len(texts))
        pipeline = KimodoPipeline(Model(config, rep), encoder)

        def sample(text_features, text_mask, lengths, steps, **kwargs):
            _, local, roots, _ = fixture.motion(frames=max(lengths), batch=len(lengths))
            return rep(local, roots, to_normalize=True, lengths=mx.array(lengths))

        with (
            patch.object(pipeline, "sample", side_effect=sample),
            patch(
                "mlx_vlm.models.kimodo.postprocess.post_process_motion",
                wraps=post_process_motion,
            ) as cleanup,
        ):
            with (
                patch.object(
                    mx, "eval", side_effect=AssertionError("eager evaluation")
                ),
                patch.object(
                    mx, "synchronize", side_effect=AssertionError("synchronize")
                ),
            ):
                result = pipeline.generate(
                    ["walk", "wave"],
                    [12, 12],
                    4,
                    multi_prompt=True,
                    post_processing=True,
                    compile=compile,
                )
        assert cleanup.call_count == 2
        assert all(call.kwargs["compile"] == compile for call in cleanup.call_args_list)
        assert [call.args[0].shape[1] for call in cleanup.call_args_list] == [12, 17]
        history_constraints = cleanup.call_args_list[1].args[4][0]
        assert history_constraints[0]["type"] == "fullbody"
        assert history_constraints[0]["frame_indices"] == list(range(5))
        assert history_constraints[1]["type"] == "end-effector"
        assert result["posed_joints"].shape == (24, skeleton.nbjoints, 3)
        assert mx.all(mx.isfinite(result["posed_joints"]))
