"""models.py
Модуль содержит реализацию `Qwen2_5_VLModel` — обёртки вокруг мультимодальной
инструкционной модели Qwen-2.5-VL. Класс предоставляет удобные методы для
инференса по одному или нескольким изображениям, принимая как пути к файлам,
объекты PIL/PyTorch, так и URL.

Google-style docstrings на русском языке используются для единообразной
документации API.
"""

from typing import Any

from model_interface.model_interface import ModelInterface
from qwen_vl_utils import process_vision_info
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class Qwen2_5_VLModel(ModelInterface):
    """Инференс-класс для моделей семейства Qwen-2.5-VL.

    Класс инкапсулирует логику загрузки весов, подготовку входных данных и
    генерацию ответов. Предоставляет методы для работы как с одним, так и с
    несколькими изображениями.
    """

    def __init__(
        self,
        model_config: dict[str, Any]
    ) -> None:
        """Инициализирует модель.

        Args:
            model_config (Dict[str, Any]): Конфигурация модели в формате:
                {
                    "common_params": {
                        "model_name": "Qwen2.5-VL-7B-Instruct",
                        "system_prompt": "",
                        "cache_dir": "model_cache",
                        "device_map": "auto"
                    },
                    "specific_params": {
                        "min_pixels": 256 * 28 * 28,
                        "max_pixels": 1280 * 28 * 28
                    }
                }
        """
        # Инициализируем параметры с значениями по умолчанию
        default_common_params = {
            "model_name": "Qwen2.5-VL-7B-Instruct",
            "system_prompt": "",
            "cache_dir": "model_cache",
            "device_map": "auto"
        }

        default_specific_params = {
            "min_pixels": 256 * 28 * 28,    # 200,704 - согласно документации
            "max_pixels": 1280 * 28 * 28    # 1,003,520 - согласно документации
        }

        # Применяем конфигурацию поверх значений по умолчанию
        self.common_params = default_common_params.copy()
        self.common_params.update(model_config.get("common_params", {}))

        self.specific_params = default_specific_params.copy()
        self.specific_params.update(model_config.get("specific_params", {}))

        self.framework = "Hugging_Face"

        # Проверяем, указан ли attn_implementation в specific_params
        if "attn_implementation" in self.specific_params:
            attn_implementation = self.specific_params["attn_implementation"]
            print(f"INFO: Используется принудительно заданная реализация внимания: {attn_implementation}")
        else:
            # Проверяем доступность flash_attn без импорта модуля (избегаем F401)
            import importlib.util  # локальный импорт, чтобы не тянуть в глобалы

            if importlib.util.find_spec("flash_attn") is not None:
                attn_implementation = "flash_attention_2"
                print("INFO: Используется FlashAttention2 для оптимизации производительности")
            else:
                attn_implementation = "eager"
                print("WARNING: flash_attn не установлен, используется стандартная реализация внимания")

        # default: Load the model on the available device(s)
        model_path = f"Qwen/{self.common_params['model_name']}"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.common_params["device_map"],
            attn_implementation=attn_implementation,
            cache_dir=self.common_params["cache_dir"],
        )

        # default processor
        self.processor = AutoProcessor.from_pretrained(model_path, cache_dir=self.common_params["cache_dir"])

    def get_message(self, image: Any, prompt: str) -> dict:
        """Формирует сообщение в формате Qwen-VL.

        Args:
            image (Any): Изображение — путь к файлу, URL либо готовый объект
                (PIL.Image, ``torch.Tensor`` и т.д.).
            prompt (str): Текстовый промпт, задаваемый модели.

        Returns:
            dict: Словарь, соответствующий ожидаемому формату чата Qwen.
        """
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": self.specific_params["min_pixels"],
                    "max_pixels": self.specific_params["max_pixels"],
                },
                {"type": "text", "text": prompt},
            ],
        }

        return message

    # ------------------------------------------------------------------
    # Внутренние сервисные методы
    # ------------------------------------------------------------------

    def _generate_answer(self, messages: list[dict]) -> str:
        """Запускает пайплайн инференса для переданных сообщений.

        Args:
            messages (List[dict]): Список сообщений в формате Qwen-VL chat.

        Returns:
            str: Сгенерированный моделью ответ.
        """
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # process_vision_info может возвращать 2 или 3 элемента в зависимости от версии
        vision_info = process_vision_info(messages)  # type: ignore
        if isinstance(vision_info, tuple) and len(vision_info) == 3:  # type: ignore[arg-type]
            image_inputs, video_inputs, _ = vision_info  # compat с более старыми версиями, где возвращается 3 значения
        else:
            image_inputs, video_inputs = vision_info  # type: ignore[assignment]
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        # Приводим к list для избежания ошибок статического анализа ("Never is not iterable")
        in_ids_list = list(inputs.input_ids)  # type: ignore[arg-type]
        gen_ids_list = list(generated_ids)  # type: ignore[arg-type]

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(in_ids_list, gen_ids_list, strict=False)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def get_messages(self, images: list[Any], prompt: str) -> list[dict]:
        """Формирует список сообщений (один элемент) для нескольких изображений.

        Args:
            images (List[Any]): Коллекция изображений (пути, URL или объекты).
            prompt (str): Текстовый промпт.

        Returns:
            List[dict]: Список сообщений, совместимых с Qwen-VL chat.
        """
        return [
            {
                "role": "user",
                "content": [
                    *[
                        {
                            "type": "image",
                            "image": img,
                            "min_pixels": self.specific_params["min_pixels"],
                            "max_pixels": self.specific_params["max_pixels"],
                        }
                        for img in images
                    ],
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    # ------------------------------------------------------------------
    # Публичные методы инференса
    # ------------------------------------------------------------------

    def predict_on_image(self, image: Any, prompt: str) -> str:
        """Генерирует ответ по одному изображению.

        Args:
            image (Any): Изображение или ссылка/путь к нему.
            prompt (str): Промпт, описывающий требуемый ответ.

        Returns:
            str: Сгенерированный текстовый ответ модели.
        """
        messages = [self.get_message(image, prompt)]
        return self._generate_answer(messages)

    def predict_on_images(self, images: list[Any], prompt: str) -> str:
        """Генерирует ответ по списку изображений.

        Args:
            images (List[Any]): Список изображений (пути, URL или объекты).
            prompt (str): Промпт, задаваемый модели.

        Returns:
            str: Сгенерированный текстовый ответ модели.
        """
        return self._generate_answer(self.get_messages(images, prompt))


# Флаг для предотвращения повторной регистрации
_models_registered = False

def register_models() -> None:
    """Регистрирует модели семейства Qwen2.5-VL в ModelFactory.

    Эта функция должна быть вызвана явно для регистрации модели.
    Такой подход предотвращает случайное удаление импорта статическими анализаторами.
    Функция идемпотентна - повторные вызовы не будут дублировать регистрацию.
    """
    global _models_registered

    if _models_registered:
        return

    from model_interface.model_factory import ModelFactory

    ModelFactory.register_model("Qwen2.5-VL", "model_qwen2_5_vl.models:Qwen2_5_VLModel")
    # ModelFactory уже логирует успешную регистрацию, дублировать не нужно

    _models_registered = True
