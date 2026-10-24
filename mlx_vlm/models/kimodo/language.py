# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy MLX LLM2Vec text conditioning for Kimodo."""

import json
import math
import os
import re
from pathlib import Path
from typing import Optional

import mlx.core as mx

from ..llama_bidirec import Model as BidirectionalLlama
from ..llama_bidirec.config import ModelConfig as LlamaConfig
from ..pooling import mean_pooling

_INSTRUCTION_SEPARATOR = "!@#$%^&*()"


def _read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _read_weights(path: Path, adapter: bool = False) -> dict:
    if adapter:
        files = [path / "adapter_model.safetensors"]
    elif (index_path := path / "model.safetensors.index.json").exists():
        files = [
            path / filename
            for filename in sorted(set(_read_json(index_path)["weight_map"].values()))
        ]
    else:
        files = sorted(path.glob("model*.safetensors"))
    if not files or any(not filename.exists() for filename in files):
        raise FileNotFoundError(f"No complete safetensors checkpoint in {path}")
    weights = {}
    for filename in files:
        weights.update(mx.load(str(filename)))
    return weights


def _base_weight_name(name: str) -> str:
    if name.startswith("base_model.model."):
        name = name[len("base_model.model.") :]
    name = name.replace(".base_layer.", ".")
    if not name.startswith("model."):
        name = "model." + name
    return name


def merge_lora_weights(weights: dict, adapter_weights: dict, config: dict) -> dict:
    """Merge a PEFT Llama LoRA adapter into an MLX checkpoint lazily."""
    if config.get("peft_type", "LORA") != "LORA" or config.get("use_dora", False):
        raise ValueError("Kimodo text encoding supports plain PEFT LoRA adapters.")
    merged = dict(weights)
    pairs = {}
    for name, value in adapter_weights.items():
        match = re.fullmatch(r"(.*)\.lora_([AB])(?:\.[^.]+)?\.weight", name)
        if match:
            prefix, side = match.groups()
            pairs.setdefault(prefix, {})[side] = value
        elif ".modules_to_save." in name:
            name = re.sub(r"\.modules_to_save\.[^.]+\.", ".", name)
            merged[_base_weight_name(name)] = value
        elif name.endswith(".bias"):
            merged[_base_weight_name(name)] = value
        else:
            raise ValueError(f"Unsupported LLM2Vec adapter weight: {name}")

    for prefix, pair in pairs.items():
        name = _base_weight_name(prefix + ".weight")
        if set(pair) != {"A", "B"}:
            raise ValueError(f"Incomplete LoRA pair for {prefix}")
        if name not in merged:
            raise ValueError(f"LoRA target is absent from the base model: {name}")
        rank = pair["A"].shape[0]
        alpha = config.get("lora_alpha", rank)
        for pattern, value in config.get("alpha_pattern", {}).items():
            if re.search(rf"(^|\.){pattern}$", prefix):
                alpha = value
                break
        scale = alpha / (math.sqrt(rank) if config.get("use_rslora", False) else rank)
        delta = pair["B"].astype(mx.float32) @ pair["A"].astype(mx.float32)
        if config.get("fan_in_fan_out", False):
            delta = delta.T
        base = merged[name]
        if delta.shape != base.shape:
            raise ValueError(
                f"LoRA shape mismatch for {name}: {delta.shape} != {base.shape}"
            )
        merged[name] = (base.astype(mx.float32) + scale * delta).astype(base.dtype)
    return merged


class LLM2VecEncoder(BidirectionalLlama):
    """Bidirectional Llama embeddings with the source LLM2Vec prompt format.

    Tokenization runs on the host. All model, attention, and pooling operations
    return lazy MLX arrays, without evaluating or copying embeddings to the host.
    """

    def __init__(
        self,
        config: LlamaConfig | dict,
        tokenizer=None,
        llm_dim: Optional[int] = None,
        pooling_mode: str = "mean",
        max_length: int = 512,
        doc_max_length: int = 400,
        skip_instruction: bool = True,
        base_model_name_or_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    ):
        if isinstance(config, dict):
            config = dict(config)
            if rope_parameters := config.get("rope_parameters"):
                config.setdefault(
                    "rope_theta", rope_parameters.get("rope_theta", 10000)
                )
                if rope_parameters.get("rope_type", "default") != "default":
                    config.setdefault("rope_scaling", rope_parameters)
            config = LlamaConfig.from_dict(config)
        if config.model_type not in ("llama", "llama_bidirec"):
            raise ValueError("Kimodo's LLM2Vec text encoder requires a Llama model.")
        super().__init__(config)
        if pooling_mode not in (
            "mean",
            "weighted_mean",
            "eos_token",
            "last_token",
            "bos_token",
        ):
            raise ValueError(f"Unsupported LLM2Vec pooling mode: {pooling_mode}")
        if max_length < 1 or doc_max_length < 1:
            raise ValueError("Text length limits must be positive.")
        self.tokenizer = tokenizer
        self.llm_dim = llm_dim or config.hidden_size
        if self.llm_dim != config.hidden_size:
            raise ValueError("llm_dim must equal the text model hidden size.")
        self.pooling_mode = pooling_mode
        self.max_length = max_length
        self.doc_max_length = doc_max_length
        self.skip_instruction = skip_instruction
        self.base_model_name_or_path = base_model_name_or_path
        if tokenizer is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

    def encode_tokens(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        embed_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Encode tokens without text processing or a host synchronization."""
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        hidden_states = self._last_hidden_state(input_ids, attention_mask)
        mask = (
            embed_mask
            if self.skip_instruction and embed_mask is not None
            else attention_mask
        )
        if self.pooling_mode == "mean":
            # The reference slices [-length:], which includes all tokens at zero.
            mask = mx.where(mx.sum(mask, axis=-1, keepdims=True) > 0, mask, 1)
            pooled = mean_pooling(hidden_states, mask).astype(hidden_states.dtype)
        elif self.pooling_mode == "weighted_mean":
            weights = mx.cumsum(mask, axis=-1) * mask
            pooled = mean_pooling(hidden_states, weights)
        elif self.pooling_mode in ("eos_token", "last_token"):
            pooled = hidden_states[:, -1]
        else:
            if self.tokenizer is None or self.tokenizer.bos_token_id is None:
                raise ValueError("BOS pooling requires a tokenizer with a BOS token.")
            # LLM2Vec inputs have one BOS token per sentence.
            pooled = mean_pooling(
                hidden_states, input_ids == self.tokenizer.bos_token_id
            ).astype(hidden_states.dtype)
        return pooled.astype(mx.float32)

    def prepare_for_tokenization(self, text: str) -> str:
        if self.base_model_name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct":
            return (
                "<|start_header_id|>user<|end_header_id|>\n\n"
                + text.strip()
                + "<|eot_id|>"
            )
        if self.base_model_name_or_path == "meta-llama/Llama-2-7b-chat-hf":
            text = "[INST] " + text.strip() + " [/INST]"
        if self.pooling_mode == "eos_token":
            suffix = (
                "<|end_of_text|>"
                if self.base_model_name_or_path == "meta-llama/Meta-Llama-3-8B"
                else " </s>"
            )
            text = text.strip() + suffix
        return text

    def _convert_to_str(self, instruction: str, text: str) -> str:
        def token_length(value):
            return len(
                self.tokenizer(
                    value,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=False,
                )["input_ids"]
            )

        length = token_length(text)
        while length > self.doc_max_length:
            reduced_length = int(len(text.split()) * self.doc_max_length / length)
            text = " ".join(text.split()[:reduced_length])
            length = token_length(text)
        prefix = instruction.strip() + " " if instruction else ""
        return prefix + _INSTRUCTION_SEPARATOR + text

    def tokenize(self, texts: list[str]) -> dict[str, mx.array]:
        if self.tokenizer is None:
            raise ValueError("A tokenizer is required to encode text.")
        split = [text.split(_INSTRUCTION_SEPARATOR) for text in texts]
        features = self.tokenizer(
            ["".join(parts) for parts in split],
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        sequence_length = len(features["input_ids"][0])
        embed_masks = []
        for parts in split:
            document = parts[1] if len(parts) > 1 else ""
            token_ids = self.tokenizer(
                document,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )["input_ids"]
            length = min(len(token_ids), sequence_length)
            embed_masks.append([0] * (sequence_length - length) + [1] * length)
        return {
            "input_ids": mx.array(features["input_ids"], dtype=mx.int32),
            "attention_mask": mx.array(features["attention_mask"], dtype=mx.int32),
            "embed_mask": mx.array(embed_masks, dtype=mx.int32),
        }

    def encode(self, sentences: str | list[str] | list[list[str]]) -> mx.array:
        """Return unnormalized [batch, hidden_size] embeddings lazily."""
        if isinstance(sentences, str):
            sentences = [sentences]
        if not sentences:
            raise ValueError("At least one sentence is required.")
        embeddings = []
        # Preserve Kimodo's internal batch size of one for repeatability.
        for sentence in sentences:
            instruction, text = (
                ("", sentence) if isinstance(sentence, str) else sentence[:2]
            )
            prepared = self.prepare_for_tokenization(
                self._convert_to_str(instruction, text)
            )
            embeddings.append(self.encode_tokens(**self.tokenize([prepared])))
        return mx.concatenate(embeddings, axis=0)

    def __call__(self, text: str | list[str]):
        encoded = self.encode(text)[:, None, :]
        if isinstance(text, str):
            return encoded[0], 1
        return encoded, [1] * len(text)

    def save_pretrained(self, path: str | Path):
        from ...utils import save_weights

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = self.config.to_dict()
        config["kimodo_llm2vec"] = {
            "base_model_name_or_path": self.base_model_name_or_path,
            "pooling_mode": self.pooling_mode,
            "max_length": self.max_length,
            "doc_max_length": self.doc_max_length,
            "skip_instruction": self.skip_instruction,
            "llm_dim": self.llm_dim,
        }
        if hasattr(self, "quantization"):
            config["quantization"] = self.quantization
        (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        if self.tokenizer is None:
            raise ValueError("A tokenizer is required to save the text encoder.")
        self.tokenizer.save_pretrained(path)
        save_weights(path, self)

    @classmethod
    def from_converted(cls, path: str | Path):
        from transformers import AutoTokenizer

        from ...utils import load_model_weights

        path = Path(path)
        config = _read_json(path / "config.json")
        encoder = cls(
            config,
            tokenizer=AutoTokenizer.from_pretrained(path),
            **config.get("kimodo_llm2vec", {}),
        )
        weights = encoder.sanitize(_read_weights(path))
        load_model_weights(encoder, config, weights, path, lazy=True)
        if "quantization" in config:
            encoder.quantization = config["quantization"]
        return encoder

    @classmethod
    def from_pretrained(
        cls,
        base_model_name_or_path: str,
        peft_model_name_or_path: Optional[str] = None,
        dtype: str = "bfloat16",
        llm_dim: Optional[int] = None,
        base_model_override: Optional[str] = None,
        **kwargs,
    ):
        """Load the MNTP adapter and supervised adapter into a native MLX Llama."""
        from transformers import AutoTokenizer

        from ...utils import get_model_path

        if text_dir := os.environ.get("TEXT_ENCODERS_DIR"):
            base_model_name_or_path = str(Path(text_dir) / base_model_name_or_path)
            if peft_model_name_or_path is not None:
                peft_model_name_or_path = str(Path(text_dir) / peft_model_name_or_path)
        base_path = get_model_path(base_model_name_or_path)
        if (base_path / "config.json").exists():
            config = _read_json(base_path / "config.json")
            if "kimodo_llm2vec" in config:
                if peft_model_name_or_path is not None:
                    raise ValueError(
                        "A converted text encoder already includes its adapters."
                    )
                return cls.from_converted(base_path)
        else:
            config = {}

        adapter_paths = []
        weight_path = base_path
        visited_paths = set()
        while (adapter_config_path := weight_path / "adapter_config.json").exists():
            if weight_path in visited_paths:
                raise ValueError(f"Circular text encoder adapter base: {weight_path}")
            visited_paths.add(weight_path)
            adapter_config = _read_json(adapter_config_path)
            adapter_paths.insert(0, (weight_path, adapter_config))
            next_base = base_model_override or adapter_config["base_model_name_or_path"]
            base_model_override = None
            next_path = get_model_path(next_base)
            if next_path == weight_path:
                raise ValueError(f"Circular text encoder adapter base: {next_base}")
            weight_path = next_path
        base_config = _read_json(weight_path / "config.json")
        if base_config.get("quantization") is not None:
            raise ValueError(
                "Merge the Kimodo adapters into an unquantized text encoder before quantizing."
            )
        config = config or base_config
        if peft_model_name_or_path is not None:
            supervised_path = get_model_path(peft_model_name_or_path)
            adapter_paths.append(
                (supervised_path, _read_json(supervised_path / "adapter_config.json"))
            )
        settings_path = adapter_paths[-1][0] if adapter_paths else base_path
        settings = {}
        if (settings_file := settings_path / "llm2vec_config.json").exists():
            settings.update(_read_json(settings_file))
        settings.update(kwargs)
        settings["base_model_name_or_path"] = config.get(
            "_name_or_path"
        ) or base_config.get("_name_or_path", str(weight_path))
        encoder = cls(
            config,
            tokenizer=AutoTokenizer.from_pretrained(base_path),
            llm_dim=llm_dim,
            **settings,
        )
        weights = encoder.sanitize(_read_weights(weight_path))
        target_dtype = getattr(mx, dtype) if isinstance(dtype, str) else dtype
        weights = {name: value.astype(target_dtype) for name, value in weights.items()}
        for adapter_path, adapter_config in adapter_paths:
            weights = merge_lora_weights(
                weights, _read_weights(adapter_path, adapter=True), adapter_config
            )
        encoder.load_weights(list(weights.items()), strict=True)
        encoder.eval()
        return encoder
