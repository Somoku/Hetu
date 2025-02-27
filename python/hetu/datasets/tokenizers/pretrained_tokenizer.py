import os
from typing import List, Union, Optional

class PreTrainedTokenizer(object):
    
    tokenizer_class: str = None
    
    # tiktoken
    pattern: str = None
    special_tokens: List[str] = None
    
    def __init__(self, **kwargs):
        self._tokenizer = kwargs.get("tokenizer", None)
        if kwargs.get("tokenizer_class", None) is not None:
            self.tokenizer_class = kwargs.get("tokenizer_class")
        if kwargs.get("pattern", None) is not None:
            self.pattern = kwargs.get("pattern")
        if kwargs.get("special_tokens", None) is not None:
            self.special_tokens = kwargs.get("special_tokens")
    
    @property
    def backend_tokenizer(self):
        return self._tokenizer

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ):
        subfolder = kwargs.get("subfolder", "")
        
        pattern = cls.pattern
        if pattern is None:
            pattern = kwargs.get("pattern", None)
        special_tokens = cls.special_tokens
        if special_tokens is None:
            special_tokens = kwargs.get("special_tokens", None)
        
        if len(subfolder) > 0:
            pretrained_model_name_or_path = str(os.path.join(pretrained_model_name_or_path, subfolder))
        else:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        
        if cls.tokenizer_class is not None:
            tokenizer_class = cls.tokenizer_class
        else:
            tokenizer_class = kwargs.get("tokenizer_class", None)
        
        # prioritize HuggingFaceTokenizer
        if not kwargs.get("tokenizer_class", None) and os.path.isfile(os.path.join(pretrained_model_name_or_path, "tokenizer.json")):
            tokenizer_class = "hf"
        
        if tokenizer_class is None:
            if os.path.isfile(os.path.join(pretrained_model_name_or_path, "vocab.json")):
                tokenizer_class = "gpt2"
                assert os.path.isfile(os.path.join(pretrained_model_name_or_path, "merges.txt")), \
                    "Cannot find `merges.txt` file, which is required by GPT2Tokenizer"
            else:
                raise ValueError(f"Cannot determine tokenizer class from {pretrained_model_name_or_path}, please specify `tokenizer_class`")
        
        kwargs.update({"tokenizer_class": tokenizer_class})
        
        if tokenizer_class == "hf":
            from .hf_tokenizer import HuggingFaceTokenizer
            
            _tokenizer = HuggingFaceTokenizer.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir, **kwargs)
        elif tokenizer_class == "sentencepiece":
            from .sentencepiece_tokenizer import SentencePieceTokenizer
            
            if os.path.isfile(pretrained_model_name_or_path):
                model_file = pretrained_model_name_or_path
            elif os.path.isdir(pretrained_model_name_or_path):
                model_file = None
                for file in os.listdir(pretrained_model_name_or_path):
                    if file.endswith(".model"):
                        model_file = os.path.join(pretrained_model_name_or_path, file)
                        break
                if model_file is None:
                    raise ValueError(f"Cannot find model file in {pretrained_model_name_or_path}")
            else:
                raise ValueError(f"Unknown pretrained_model_name_or_path {pretrained_model_name_or_path}")

            _tokenizer = SentencePieceTokenizer(model_file, **kwargs)
        elif tokenizer_class == "tiktoken":
            from .tiktoken_tokenizer import TikTokenizer
            
            if os.path.isfile(pretrained_model_name_or_path):
                model_file = pretrained_model_name_or_path
            elif os.path.isdir(pretrained_model_name_or_path):
                model_file = None
                for file in os.listdir(pretrained_model_name_or_path):
                    if file.endswith(".model"):
                        model_file = os.path.join(pretrained_model_name_or_path, file)
                        break
                if model_file is None:
                    raise ValueError(f"Cannot find model file in {pretrained_model_name_or_path}")
            else:
                raise ValueError(f"Unknown pretrained_model_name_or_path {pretrained_model_name_or_path}")

            _tokenizer = TikTokenizer(model_file, pattern, special_tokens, **kwargs)
        elif tokenizer_class == "gpt2":
            from .gpt2_tokenizer import GPT2Tokenizer
            
            _tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir, **kwargs)
        else:
            raise ValueError(f"Unknown tokenizer class {tokenizer_class}")
        
        instance = cls(_tokenizer, **kwargs)
        
        return instance
    
    def __getattr__(self, name):
        self_attr = self.__dict__.get(name, None)
        if self_attr is not None:
            return self_attr
        attr = getattr(self._tokenizer, name)
        if attr is None:
            raise AttributeError(f"{self._tokenizer.__class__.__name__} object has no attribute '{name}'")
        return attr

__all__ = ["PreTrainedTokenizer"]