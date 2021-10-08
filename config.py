from typing import List
from pathlib import Path
import logging
import yaml
import sys


logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


class BaseConfig:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super().__new__(cls)
            cls._instance.load_from_file()
        return cls._instance

    @classmethod
    def get_config(cls) -> "BaseConfig":
        return cls.__call__()

    def load_from_file(self) -> None:
        self.config_path = Path("config.yaml")
        if not self.config_path.is_file():
            logger.error(f"config.yaml not found: {self.config_path}")
            sys.exit(1)

        else:
            with open(self.config_path, "r") as f:
                try:
                    config_dict = yaml.safe_load(f)["base"]
                    self._image_width = config_dict["image_width"]
                    self._image_height = config_dict["image_height"]
                    self._max_length = config_dict["max_length"]
                    self._downsample_factor = config_dict["downsample_factor"]
                    self._characters = sorted(config_dict["characters"])
                    self._data_dir = Path(config_dict["data_dir"])
                    self._model_dir = Path(config_dict["model_dir"])
                except yaml.YAMLError as exc:
                    logger.error(exc)
                    raise

    @property
    def image_width(self) -> int:
        return self._image_width

    @property
    def image_height(self) -> int:
        return self._image_height

    @property
    def max_length(self) -> int:
        return self._max_length

    @max_length.setter
    def max_length(self, value) -> None:
        self._max_length = value

        # load
        with open(self.config_path, "r") as f:
            try:
                config_dict = yaml.safe_load(f)["base"]
                config_dict["max_length"] = self._max_length
            except yaml.YAMLError as exc:
                logger.error(exc)
                raise

        # update
        with open(self.config_path, "w") as f:
            try:
                config_dict = {"base": config_dict}
                yaml.safe_dump(config_dict, f, indent=4)
            except yaml.YAMLError as exc:
                logger.error(exc)
                raise

    @property
    def downsample_factor(self) -> int:
        return self._downsample_factor

    @property
    def characters(self) -> List[str]:
        return self._characters

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    @characters.setter
    def characters(self, value: List[str]) -> None:
        if len(value) > 0:
            self._characters = value

            # load
            with open(self.config_path, "r") as f:
                try:
                    config_dict = yaml.safe_load(f)["base"]
                    config_dict["characters"] = self._characters
                except yaml.YAMLError as exc:
                    logger.error(exc)
                    raise

            # update
            with open(self.config_path, "w") as f:
                try:
                    config_dict = {"base": config_dict}
                    yaml.safe_dump(config_dict, f, indent=4)
                except yaml.YAMLError as exc:
                    logger.error(exc)
                    raise
