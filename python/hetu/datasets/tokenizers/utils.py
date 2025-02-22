import os
import copy
import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Union, Optional


def _vocab_size_with_padding(
    orig_vocab_size,
    make_vocab_size_divisible_by,
    tp_degree,
    rank
):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = make_vocab_size_divisible_by * tp_degree
    while (after % multiple) != 0:
        after += 1
    if rank == 0:
        print(' > padded vocab (size: {}) with {} dummy tokens '
              '(new size: {})'.format(
                  orig_vocab_size, after - orig_vocab_size, after), flush=True)
    return after

class BaseTokenizer(ABC):
    def __init__(self, **kwargs):
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
    
    @abstractmethod
    def decode(self, token_ids: List[int], **kwargs: Dict[str, Any]) -> str:
        """
        Given a list of token ids, return the decoded text, optionally including special tokens.

        Args:
            token_ids (List[int]): The list of token ids to decode.
            **kwargs (Dict[str, Any]): kwargs.

        Returns:
            str: The decoded text.
        """
        pass

class ModelTokenizer(BaseTokenizer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key in kwargs:
            if hasattr(self, key) and callable(getattr(self, key)):
                raise AttributeError(f"{key} conflicts with the method {key} in {self.__class__.__name__}")
        
        self.init_kwargs = copy.deepcopy(kwargs)
        self.name_or_path = kwargs.pop("name_or_path", "")
        self._processor_class = kwargs.pop("processor_class", None)
        
        self.extra_special_tokens = kwargs.pop("extra_special_tokens", {})
        self._set_model_specific_special_tokens(special_tokens=self.extra_special_tokens)
    
    def apply_prompt_template():
        pass
    
    def get_prompt_template():
        pass
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ):
        subfolder = kwargs.pop("subfolder", None)
        
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        vocab_files = {}
        is_local = os.path.isdir(pretrained_model_name_or_path)
        

    def encode():
        pass
    
    def decode():
        pass

    def add_special_tokens():
        pass
    
    def add_tokens():
        pass
    
    def tokenize():
        pass
    
    def convert_ids_to_tokens():
        pass
    
    def convert_tokens_to_ids():
        pass
    
    @property
    def vocab_size(self):
        pass
    
    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to index.

        `tokenizer.get_vocab()[token]` is equivalent to `tokenizer.convert_tokens_to_ids(token)` when `token` is in the
        vocab.

        Returns:
            `Dict[str, int]`: The vocabulary.
        """
        raise NotImplementedError()
    
    def save_vocabulary():
        pass