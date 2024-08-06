"""Factory for creating instances of classes based on a string identifier."""

import torch.optim as optim


def create_optimizer(optimizer_name: str, *args, **kwargs) -> optim.Optimizer:
    """Create an optimizer instance from a name and arguments.

    Args:
        optimizer_name (str): The class name of the optimizer to create.

    Raises:
        ValueError: If no optimizer is found with the given name.

    Returns:
        torch.optim.Optimizer: An instance of the Optimizer class.
    """
    optimizer_classes = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
    }

    optimizer_class = optimizer_classes.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError(f"No optimizer found with name {optimizer_name}")
    return optimizer_class(*args, **kwargs)