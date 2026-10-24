"""Processor registration for motion checkpoints loaded with mlx_vlm.load."""

from ..base import install_auto_processor_patch


class KimodoProcessor:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls()

    def __call__(self, prompts, **kwargs):
        return {"prompts": prompts, **kwargs}


install_auto_processor_patch(["kimodo"], KimodoProcessor)
