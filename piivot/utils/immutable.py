"""Create an immutable object which should be used instead of polluting global."""


class Immutable:
    def __init__(self):
        """Initialize a dictionary to store the attributes of the object.

        Use all caps for variable names to indicate that they are constants.

        e.g. immutable.DEBUG __not__ immutable.debug
        """
        self._dict = {}

    def __getattr__(self, name: str) -> any:
        """Get the value of an attribute if it exists in the dictionary.

        Args:
            name (str): The name of the attribute to get.

        Raises:
            AttributeError: If the attribute does not exist in the dictionary.

        Returns:
            any: The value of the attribute.
        """
        if name in self._dict:
            return self._dict[name]
        else:
            raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name: str, value: any):
        """Set the value of an attribute if it does not already exist in the dictionary.

        Args:
            name (str): The name of the attribute to set.
            value (any): The value to set the attribute to.

        Raises:
            ValueError: If the attribute already exists in the dictionary.
        """
        if name == "_dict":
            # Call the superclass __setattr__ method to avoid infinite recursion.
            super().__setattr__(name, value)
        elif name in self._dict:
            if self._dict[name] != value:
                raise ValueError(
                    f"Value for attribute {name} is already set and cannot be changed."
                )
        else:
            self._dict[name] = value


global_immutable = Immutable()