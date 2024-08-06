"""Factory for creating instances of classes based on a string identifier."""

from transformers import BertTokenizerFast, DebertaV2TokenizerFast, PreTrainedTokenizerFast


def create_tokenizer(tokenizer_name: str, from_pretrained: bool = False, *args, **kwargs) -> PreTrainedTokenizerFast:
    """Create a tokenizer instance from a name and arguments.

    Args:
        tokenizer_name (str): The class name of the tokenizer to create.

    Raises:
        ValueError: If no tokenizer is found with the given name.

    Returns:
        PreTrainedTokenizerFast: An instance of the tokenizer class.
    """
    
    tokenizer_classes = {
        "BERT": BertTokenizerFast,
        "DeBERTa": DebertaV2TokenizerFast,
    }
    
    tokenizer_class = tokenizer_classes.get(tokenizer_name)
    if tokenizer_class is None:
        raise ValueError(f"No tokenizer found with name {tokenizer_name}")
    
    if from_pretrained:
        return tokenizer_class.from_pretrained(*args, **kwargs)
    else:
        return tokenizer_class(*args, **kwargs)  # TODO no usecase for non pretrained tokenizer ATM