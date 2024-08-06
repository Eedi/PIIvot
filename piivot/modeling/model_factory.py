"""Factory for creating instances of classes based on a string identifier."""
import torch

from transformers import BertForTokenClassification, DebertaV2ForTokenClassification 


def create_model(name: str, from_pretrained: bool = False, *args, **kwargs) -> torch.nn.Module:
    """Create an model instance from a name and arguments.

    Args:
        name (str): The class name of the model to create.

    Raises:
        ValueError: If no model is found with the given name.

    Returns:
        torch.nn.Module: An instance of the model class.
    """
    model_classes = {
        "BERT": BertForTokenClassification,
        "DeBERTa": DebertaV2ForTokenClassification
    }

    model_class = model_classes.get(name)
    if model_class is None:
        raise ValueError(f"No model found with name {name}")
    
    if from_pretrained:
        return model_class.from_pretrained(*args, **kwargs)
    else:
        return model_class(*args, **kwargs)