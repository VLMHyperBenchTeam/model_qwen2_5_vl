# mypy: ignore-errors
"""Пакет *model_qwen2_5_vl*
+--------------------------------
+Предоставляет классы и вспомогательные функции для работы с семейством
+мультимодальных моделей **Qwen-2.5-VL**.  Знаковые функции:
+
* ``create_qwen_model_config`` – генерирует вложенную конфигурацию, совместимую
  с ``ModelFactory.initialize_model``.
* ``initialize_qwen_model`` – «одна кнопка» для получения готового
  инстанса модели.  Под капотом вызывает фабрику.

Пакет **НЕ** импортируется внутри *model_interface* — тем самым сохраняется
направленная зависимость: верхний уровень ничего не знает о конкретных
семействах моделей.
"""

from __future__ import annotations

from typing import Any

# Публичные константы -------------------------------------------------------

# Самая популярная конфигурация весов на момент написания кода
DEFAULT_QWEN_MODEL = "Qwen2.5-VL-7B-Instruct"

# ---------------------------------------------------------------------------
# Вспомогательные функции конфигурации / инициализации
# ---------------------------------------------------------------------------


def create_qwen_model_config(
    model_name: str = DEFAULT_QWEN_MODEL,
    cache_dir: str = "model_cache",
    device_map: str = "auto",
    system_prompt: str = "",
    specific_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Сформировать вложенный конфиг для ModelFactory.

    Parameters
    ----------
    model_name
        Полное имя модели в репозитории HuggingFace.
    cache_dir
        Папка под кэш весов/процессора.
    device_map
        Карта распределения слоёв по устройствам (``"auto"``, ``"cuda:0"``…).
    system_prompt
        Базовый системный промпт (может быть пустым).
    specific_params
        Любые дополнительные поля, специфичные для семейства Qwen.
    """

    # Примечание: модель должна быть зарегистрирована вызовом register_models()
    # перед использованием этой функции

    return {
        "model_family": "Qwen2.5-VL",
        "common_params": {
            "model_name": model_name,
            "cache_dir": cache_dir,
            "device_map": device_map,
            "system_prompt": system_prompt or "",
        },
        "specific_params": specific_params or {},
    }


def initialize_qwen_model(
    model_name: str = DEFAULT_QWEN_MODEL,
    cache_dir: str = "model_cache",
    device_map: str = "auto",
    system_prompt: str = "",
    specific_params: dict[str, Any] | None = None,
):
    """Упрощённая инициализация модели Qwen-2.5-VL.

    Возвращает готовый объект, реализующий ``ModelInterface``.
    """

    from model_interface.model_factory import ModelFactory

    # Регистрируем модели перед использованием
    register_models()

    config = create_qwen_model_config(
        model_name=model_name,
        cache_dir=cache_dir,
        device_map=device_map,
        system_prompt=system_prompt,
        specific_params=specific_params,
    )

    return ModelFactory.initialize_model(config)


# What we export ----------------------------------------------------------------

__all__ = [
    "DEFAULT_QWEN_MODEL",
    "create_qwen_model_config",
    "initialize_qwen_model",
    "register_models",
]


def register_models() -> None:
    """Регистрирует модели семейства Qwen2.5-VL в ModelFactory.

    Эта функция может быть вызвана напрямую для явной регистрации модели.
    """
    from model_qwen2_5_vl.models import register_models as _register_models
    _register_models()
