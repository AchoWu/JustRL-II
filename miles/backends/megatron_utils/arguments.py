import logging
import os

from megatron.training.arguments import parse_args, validate_args

# Megatron moved the vocab padding helper: it used to live in
# megatron/training/tokenizer/tokenizer.py as _vocab_size_with_padding(vocab_size, args);
# newer miles-main dropped that whole tokenizer/ package and exposes
# calculate_padded_vocab_size(vocab_size, make_vocab_size_divisible_by,
# tensor_model_parallel_size) from megatron/training/vocab_utils.py instead.
# The arithmetic is identical (ceil(vocab / (divisible_by * tp)) * multiple); only the
# argument plumbing changed. Support both so the recipe survives either checkout.
try:
    from megatron.training.vocab_utils import calculate_padded_vocab_size

    def _vocab_size_with_padding(vocab_size, args):
        return calculate_padded_vocab_size(
            vocab_size,
            args.make_vocab_size_divisible_by,
            args.tensor_model_parallel_size,
        )

except ImportError:  # older Megatron checkouts
    from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding

__all__ = ["validate_args", "parse_args", "set_default_megatron_args"]

logger = logging.getLogger(__name__)


def set_default_megatron_args(args):
    # always use zero optimizer
    args.use_distributed_optimizer = True
    # TODO: maybe change this after megatron has good fp8 support
    args.bf16 = not args.fp16
    # placeholders
    if args.seq_length is None:
        args.seq_length = 4096
    args.max_position_embeddings = args.seq_length
    # Notice(Jiajun): new megatron has removed this argument and use dp_reshardable instead of fully_shard
    if os.getenv("DEPRECATED_MEGATRON_COMPATIBLE", "0") == "1":
        args.dist_ckpt_save_pre_mcore_014 = True
    # compatible for megatron
    if hasattr(args, "rope_type") and args.rope_type is None:
        args.rope_type = "yarn" if args.multi_latent_attention else "rope"

    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)

    if not args.tokenizer_model and not args.tokenizer_type:
        logger.info("--tokenizer-model not set, use --hf-checkpoint as tokenizer model.")
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"

    return args
