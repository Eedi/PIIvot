"""Factory for creating instances of classes based on a string identifier."""

from torch.utils.data import Dataset
from .BERTDialogueDataset import BERTDialogueDataset, MultiSentenceBERTDialogueDataset


def create_dataset(dataset_name: str, *args, **kwargs) -> Dataset:
    """Create an dataset instance from a name and arguments.

    Args:
        dataset_name (str): The class name of the dataset to create.

    Raises:
        ValueError: If no dataset is found with the given name.

    Returns:
        torch.utils.data.Dataset: An instance of the dataset class.
    """
    dataset_classes = {
        "BERTDialogue": BERTDialogueDataset,
        "MultiSentenceBERTDialogue": MultiSentenceBERTDialogueDataset,
    }

    dataset_class = dataset_classes.get(dataset_name)
    if dataset_class is None:
        raise ValueError(f"No dataset found with name {dataset_name}")
    return dataset_class(*args, **kwargs)