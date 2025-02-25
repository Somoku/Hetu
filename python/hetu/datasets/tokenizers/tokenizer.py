from .gpt2_tokenizer import GPT2BPETokenizer
from .sentencepiece_tokenizer import SentencePieceTokenizer
from .tiktoken_tokenizer import TikTokenizer, PATTERN_TIKTOKEN, PATTERN_TIKTOKEN_V2, PATTERN_TIKTOKEN_CL100K
from .hf_tokenizer import HuggingFaceTokenizer

pattern_dict = {
    "v1": PATTERN_TIKTOKEN,
    "v2": PATTERN_TIKTOKEN_V2,
    "cl100k": PATTERN_TIKTOKEN_CL100K,
}

def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * \
        args.tensor_model_parallel_size
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print(' > padded vocab (size: {}) with {} dummy tokens '
              '(new size: {})'.format(
                  orig_vocab_size, after - orig_vocab_size, after), flush=True)
    return after

def build_tokenizer(args, **kwargs):
    """Initialize tokenizer."""
    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.tokenizer_type),
              flush=True)

    # Select and instantiate the tokenizer.
    if args.tokenizer_type == 'GPT2BPETokenizer':
        assert args.vocab_file is not None
        assert args.merge_file is not None
        tokenizer = GPT2BPETokenizer(args.vocab_file, args.merge_file)
    elif args.tokenizer_type == 'SentencePieceTokenizer':
        assert args.tokenizer_model is not None
        tokenizer = SentencePieceTokenizer(
            args.tokenizer_model, vocab_extra_ids=args.vocab_extra_ids
        )
    elif args.tokenizer_type == 'HuggingFaceTokenizer':
        if args.tokenizer_model is not None:
            kwargs.update({"tokenizer_file": args.tokenizer_model})
        tokenizer = HuggingFaceTokenizer(**kwargs)
    elif args.tokenizer_type == 'TikTokenizer':
        assert args.tokenizer_model is not None
        assert args.tiktoken_pattern is not None
        assert args.tiktoken_pattern in {"v1", "v2", "cl100k"}

        pattern = pattern_dict[args.tiktoken_pattern]
        tokenizer = TikTokenizer(
            path=args.tokenizer_model,
            pattern=pattern,
            special_tokens=args.tiktoken_special_tokens,
        )
    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(args.tokenizer_type))
    
    # Add vocab size.
    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size,
                                                          args)

    return tokenizer

__all__ = ["build_tokenizer"]