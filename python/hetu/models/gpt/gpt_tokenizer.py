from hetu.datasets.tokenizers.pretrained_tokenizer import PreTrainedTokenizer

class GPTTokenizer(PreTrainedTokenizer):
    tokenizer_class = "gpt2"
    
    def __init__(self):
        super().__init__()

__all__ = ["GPTTokenizer"]