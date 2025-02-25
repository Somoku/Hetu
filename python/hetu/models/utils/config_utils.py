import os
import json
from typing import Any, Dict, Optional, Union
from hetu.models.utils.hub import is_remote_url
from hetu.models.utils import CONFIG_NAME

class PreTrainedConfig(object):
    model_type: str = ""
    sub_configs: Dict[str, "PreTrainedConfig"] = {}
    attribute_map: Dict[str, str] = {}
    
    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)
    
    def __init__(self, **kwargs):
        self.model_dtype = kwargs.pop("model_dtype", None)
        # Name or path to the pretrained checkpoint
        self._name_or_path = str(kwargs.pop("name_or_path", ""))
        
        # Tokenizer arguments
        self.tokenizer_class = kwargs.pop("tokenizer_class", None)
        self.prefix = kwargs.pop("prefix", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.sep_token_id = kwargs.pop("sep_token_id", None)
        
        if self.model_dtype is not None and isinstance(self.model_dtype, str):
            import hetu as ht
            self.model_dtype = getattr(ht, self.model_dtype)
    
    @property
    def name_or_path(self):
        return getattr(self, "_name_or_path", None)

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ):
        kwargs["cache_dir"] = cache_dir
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ):
        subfolder = kwargs.pop("subfolder", "")
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder)):
            resolved_config_file = pretrained_model_name_or_path
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            # TODO: support remote download
            raise NotImplementedError
        else:
            configuration_file = kwargs.pop("_configuration_file", CONFIG_NAME)
            resolved_config_file = os.path.join(pretrained_model_name_or_path, subfolder, configuration_file)
        
        if is_local:
            config_dict = cls._dict_from_json_file(resolved_config_file)
        else:
            # TODO: support remote download
            raise NotImplementedError
        return config_dict, kwargs
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        return cls(**config_dict)
    
    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)
    
    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def update(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            setattr(self, key, value)
        return self

__all__ = ["PreTrainedConfig"]