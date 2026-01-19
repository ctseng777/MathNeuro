from __future__ import annotations

from pathlib import Path
from typing import Optional

from lm_eval import utils


eval_logger = utils.eval_logger


def _dataset_key(dataset_path: Optional[str], dataset_name: Optional[str]) -> str:
    if dataset_path is None:
        raise ValueError("dataset_path is required for promptsource prompts")
    if dataset_name:
        return f"{dataset_path}/{dataset_name}"
    return dataset_path


def get_prompt(prompt_name: str, dataset_path: Optional[str], dataset_name: Optional[str]):
    if prompt_name is None:
        return None

    if prompt_name.startswith("promptsource:"):
        try:
            from promptsource.templates import DatasetTemplates
        except Exception as exc:
            raise ImportError(
                "promptsource is required for promptsource:* prompts. Install via `pip install promptsource`."
            ) from exc

        template_name = prompt_name.split(":", 1)[1].strip()
        templates = DatasetTemplates(_dataset_key(dataset_path, dataset_name))
        if template_name in ("", "*"):
            first = next(iter(templates.templates.values()))
            eval_logger.warning(
                "promptsource:* selected the first available template: %s",
                first.name,
            )
            return first
        if template_name in templates:
            return templates[template_name]
        raise KeyError(f"Prompt template '{template_name}' not found for {dataset_path}")

    if prompt_name.startswith("file:"):
        prompt_path = Path(prompt_name.split(":", 1)[1]).expanduser()
        return prompt_path.read_text(encoding="utf-8")

    eval_logger.warning("Unknown prompt spec '%s'; skipping prompt.", prompt_name)
    return None
