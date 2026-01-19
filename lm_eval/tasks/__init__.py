from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from lm_eval import utils
from lm_eval.api.group import ConfigurableGroup
from lm_eval.api.task import ConfigurableTask


eval_logger = utils.eval_logger


class TaskManager:
    def __init__(
        self,
        verbosity: Union[str, int] = "INFO",
        include_path: Optional[Union[str, Iterable[str]]] = None,
        include_defaults: bool = True,
    ) -> None:
        self.verbosity = verbosity
        self.include_defaults = include_defaults
        self.include_path = include_path
        self._task_config_paths: Dict[str, Path] = {}
        self._group_config_paths: Dict[str, Path] = {}
        self._tag_index: Dict[str, List[str]] = {}
        self._index_tasks()

    def _iter_task_paths(self) -> Iterable[Path]:
        roots: List[Path] = []
        if self.include_defaults:
            roots.append(Path(__file__).resolve().parent)
        if self.include_path:
            if isinstance(self.include_path, (str, os.PathLike)):
                roots.append(Path(self.include_path))
            else:
                roots.extend(Path(path) for path in self.include_path)
        for root in roots:
            if root.exists():
                yield from root.rglob("*.yaml")

    def _index_tasks(self) -> None:
        for yaml_path in self._iter_task_paths():
            task_name, group_name, tags = self._quick_parse_yaml(yaml_path)
            if task_name:
                self._task_config_paths[task_name] = yaml_path
                for tag in tags:
                    self._tag_index.setdefault(tag, []).append(task_name)
            if group_name:
                self._group_config_paths[group_name] = yaml_path

    def _quick_parse_yaml(self, yaml_path: Path):
        task_name = None
        group_name = None
        tags: List[str] = []
        try:
            lines = yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return None, None, []

        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if task_name is None:
                task_match = re.match(r'^(\"?task\"?)\s*:\s*(.+)$', line)
                if task_match:
                    value = task_match.group(2).strip()
                    if value and not value.startswith("["):
                        task_name = self._strip_yaml_value(value)

            if group_name is None:
                group_match = re.match(r'^(\"?group\"?)\s*:\s*(.+)$', line)
                if group_match:
                    value = group_match.group(2).strip()
                    if value and not value.startswith("["):
                        group_name = self._strip_yaml_value(value)

            tag_match = re.match(r'^(\"?tag\"?)\s*:\s*(.+)$', line)
            if tag_match:
                value = tag_match.group(2).strip()
                if value.startswith("[") and value.endswith("]"):
                    inner = value[1:-1].strip()
                    if inner:
                        tags.extend(
                            [self._strip_yaml_value(item) for item in inner.split(",")]
                        )
                else:
                    tags.append(self._strip_yaml_value(value))

            if task_name and group_name and tags:
                break

        return task_name, group_name, [tag for tag in tags if tag]

    def _strip_yaml_value(self, value: str) -> str:
        value = value.strip()
        if value.endswith(","):
            value = value[:-1]
        if (value.startswith("'") and value.endswith("'")) or (
            value.startswith('"') and value.endswith('"')
        ):
            return value[1:-1]
        return value.split()[0]

    def list_all_tasks(
        self,
        list_groups: bool = True,
        list_subtasks: bool = True,
        list_tags: bool = True,
    ) -> List[str]:
        items: List[str] = []
        if list_groups:
            items.extend(sorted(self._group_config_paths.keys()))
        if list_subtasks:
            items.extend(sorted(self._task_config_paths.keys()))
        if list_tags:
            items.extend(sorted(self._tag_index.keys()))
        return items

    def match_tasks(self, task_list: List[str]) -> List[str]:
        available = set(self._task_config_paths) | set(self._group_config_paths)
        matches: List[str] = []
        for task in task_list:
            if "*" in task:
                matches.extend(sorted(fnmatch.filter(available, task)))
            elif task in available:
                matches.append(task)
            elif task in self._tag_index:
                matches.append(task)
        return matches

    def load_task(self, task_name: str) -> ConfigurableTask:
        if task_name not in self._task_config_paths:
            raise KeyError(f"Task '{task_name}' not found")
        yaml_path = self._task_config_paths[task_name]
        config = utils.load_yaml_config(yaml_path=yaml_path, mode="full")
        return ConfigurableTask(config=config)

    def load_group(self, group_name: str):
        if group_name not in self._group_config_paths:
            raise KeyError(f"Group '{group_name}' not found")
        yaml_path = self._group_config_paths[group_name]
        config = utils.load_yaml_config(yaml_path=yaml_path, mode="full")
        group = ConfigurableGroup(config=config)
        return group, config

    def _expand_group_tasks(self, task_entries: List[Union[str, dict]]):
        task_dict: Dict[Union[str, ConfigurableGroup], object] = {}
        for entry in task_entries:
            if isinstance(entry, dict):
                task_name = entry.get("task") or entry.get("group")
                if task_name is None:
                    continue
                task_dict[task_name] = ConfigurableTask(config=entry)
                continue

            if entry in self._group_config_paths:
                group_obj, group_config = self.load_group(entry)
                child_entries = group_config.get("task", [])
                task_dict[group_obj] = self._expand_group_tasks(child_entries)
                continue

            if entry in self._task_config_paths:
                task_dict[entry] = self.load_task(entry)
                continue

            if entry in self._tag_index:
                for tagged_task in self._tag_index[entry]:
                    if tagged_task in task_dict:
                        continue
                    task_dict[tagged_task] = self.load_task(tagged_task)
        return task_dict


def get_task_dict(
    task_list: List[Union[str, dict, object]],
    task_manager: Optional[TaskManager] = None,
) -> Dict[Union[str, ConfigurableGroup], object]:
    if task_manager is None:
        task_manager = TaskManager()

    task_dict: Dict[Union[str, ConfigurableGroup], object] = {}
    for task in task_list:
        if isinstance(task, dict):
            task_name = task.get("task") or task.get("group") or "custom_task"
            task_dict[task_name] = ConfigurableTask(config=task)
            continue

        if not isinstance(task, str):
            task_name = getattr(task, "task_name", type(task).__name__)
            task_dict[task_name] = task
            continue

        if task in task_manager._group_config_paths:
            group_obj, group_config = task_manager.load_group(task)
            entries = group_config.get("task", [])
            task_dict[group_obj] = task_manager._expand_group_tasks(entries)
            continue

        if task in task_manager._task_config_paths:
            task_dict[task] = task_manager.load_task(task)
            continue

        if task in task_manager._tag_index:
            for tagged_task in task_manager._tag_index[task]:
                task_dict[tagged_task] = task_manager.load_task(tagged_task)
            continue

        raise KeyError(f"Task '{task}' not found")

    return task_dict
