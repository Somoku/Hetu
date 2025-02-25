import hetu
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum, verify, EnumCheck
from typing import Any, Dict, List, Union, Iterable
from hetu.utils.common_utils import to_py_obj

SPECIAL_TOKENS_ATTRIBUTE = {
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
    "additional_special_tokens",
}

@verify(EnumCheck.UNIQUE, EnumCheck.EXACT)
class PaddingStrategy(Enum):
    NO_PAD = "no_pad"
    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    
    def __getattr__(self, name):
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @classmethod
    def _missing_(cls, value):
        raise ValueError(f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}")

@verify(EnumCheck.UNIQUE, EnumCheck.EXACT)
class TruncationStrategy(Enum):
    NO_TRUNCATE = "no_truncate"
    MAX_LENGTH = "longest_first"

class SpecialToken(object):
    special_tokens_attribute = SPECIAL_TOKENS_ATTRIBUTE
    
    def __init__(self, **kwargs):
        self.special_tokens_map = {attr: None for attr in SPECIAL_TOKENS_ATTRIBUTE}
        self.special_tokens_map["additional_special_tokens"] = []
        
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in self.special_tokens_attribute:
                if key == "additional_special_tokens":
                    assert isinstance(value, (list, tuple)), f"Value {value} is not a list or tuple"
                    assert all(
                        isinstance(t, str) for t in value
                    ), "One of the tokens is not a string or an AddedToken"
                    setattr(self, key, value)
                elif isinstance(value, str):
                    setattr(self, key, value)
                else:
                    raise TypeError(f"Special token {key} has to be either str or AddedToken but got: {type(value)}")
    
    def __setattr__(self, key, value):
        # can pass in tokens/ids
        key_without_id = key
        key_is_id = key.endswith("_id") or key.endswith("_ids")
        if key_is_id:
            key_without_id = key[:-3] if not key.endswith("_ids") else key[:-4]
            
        if key_is_id and not key_without_id.endswith("_token"):
            key_without_id += "_token"

        if self.__dict__.get("special_tokens_map", None) is not None and any(
            name in self.__dict__["special_tokens_map"] for name in [key, key_without_id]
        ):
            if key_is_id:
                if value is not None:
                    value = (
                        self.convert_ids_to_tokens(value)
                        if key != "additional_special_tokens"
                        else [self.convert_ids_to_tokens(val) for val in value]
                    )
                key = key_without_id

            if key != "additional_special_tokens" and not isinstance(value, str) and value is not None:
                raise ValueError(f"Cannot set a non-string value as the {key}")
            self.special_tokens_map[key] = value
        else:
            super().__setattr__(key, value)
    
    def __getattr__(self, key):
        # can get tokens/ids
        key_without_id = key
        key_is_id = key.endswith("_id") or key.endswith("_ids")
        if key_is_id:
            key_without_id = key[:-3] if not key.endswith("_ids") else key[:-4]

        if key_is_id and not key_without_id.endswith("_token"):
            key_without_id += "_token"

        if self.__dict__.get("special_tokens_map", None) is not None and any(
            name in self.__dict__["special_tokens_map"] for name in [key, key_without_id]
        ):
            _special_tokens_map = self.__dict__["special_tokens_map"]
            if not key_is_id:
                if _special_tokens_map[key] is None:
                    return None
                value = _special_tokens_map[key]
                return str(value) if key != "additional_special_tokens" else [str(tok) for tok in value]
            else:
                attr_as_tokens = getattr(self, key_without_id)
                return self.convert_tokens_to_ids(attr_as_tokens) if attr_as_tokens is not None else None
        elif self.__dict__.get("special_tokens_map", None) is not None and any(
            name in self.__dict__["special_tokens_map"]["additional_special_tokens"] for name in [key, key_without_id]
        ):
            _additional_special_tokens = self.__dict__["special_tokens_map"]["additional_special_tokens"]
            if key_is_id:
                for token in _additional_special_tokens:
                    if token == key_without_id:
                        return self.convert_tokens_to_ids(token)
            else:
                return key

        if key not in self.__dict__:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")
        else:
            return super().__getattr__(key)
    
    def convert_tokens_to_ids(self, tokens: Union[str, Iterable[str]]) -> Union[int, List[int]]:
        raise NotImplementedError

class BaseTokenizer(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def encode(self, text: str, **kwargs: Dict[str, Any]) -> List[int]:
        """
        Given a string, return the encoded list of token ids.

        Args:
            text (str): The text to encode.
            **kwargs (Dict[str, Any]): kwargs.

        Returns:
            List[int]: The encoded list of token ids.
        """
        pass
    
    def decode(
        self,
        token_ids: Union[int, List[int], "hetu.Tensor", "np.ndarray"],
        **kwargs: Dict[str, Any],
    ) -> str:
        """
        Given a list of token ids, return the decoded text, optionally including special tokens.

        Args:
            token_ids (List[int]): The list of token ids to decode.
            **kwargs (Dict[str, Any]): kwargs.

        Returns:
            str: The decoded text.
        """
        # Convert to python objects first
        token_ids = to_py_obj(token_ids)
        return self._decode(token_ids, **kwargs)
    
    @abstractmethod
    def _decode(
        self,
        token_ids: Union[int, List[int]],
        **kwargs: Dict[str, Any],
    ) -> str:
        pass

__all__ = ["PaddingStrategy", "TruncationStrategy", "SpecialToken", "BaseTokenizer", "SPECIAL_TOKENS_ATTRIBUTE"]