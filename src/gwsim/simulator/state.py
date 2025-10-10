"""
A descriptor class to handle a state attribute.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar, overload

T = TypeVar("T")


class StateAttribute(Generic[T]):
    """A state attribute."""

    def __init__(
        self,
        default: T | None = None,
        default_factory: Callable[[], T] | None = None,
        post_set_hook: Callable[[Any, T], None] | None = None,
    ) -> None:
        """A state attribute.

        Args:
            default (T | None, optional): The default value of the attribute. Defaults to None.
            default_factory (Callable[[], T] | None, optional): A factory to create the default value.
                This is useful when you want to have a list as a state attribute, and
                do not want to share the list across instances. Defaults to None.
            post_set_hook (Callable[[Any, T], None] | None): Called after the value of the attribute is set.
        """
        self.default = default
        self.default_factory = default_factory
        self.post_set_hook = post_set_hook
        self.name = None

    def __set_name__(self, owner: type, name: str) -> None:
        """Set the name of the attribute.

        Args:
            owner (type): Owner of the attribute.
            name (str): Name.
        """
        self.name = name
        if not hasattr(owner, "_state_attributes"):
            owner._state_attributes = []
        if name not in owner._state_attributes:
            owner._state_attributes.append(name)

    @overload
    def __get__(self, instance: None, owner: type) -> StateAttribute[T]: ...

    @overload
    def __get__(self, instance: Any, owner: type) -> T: ...

    def __get__(self, instance: Any, owner: type) -> T | StateAttribute[T]:
        """Get the value of the attribute.

        Args:
            instance (Any): Instance of the owner.
            owner (type): Placeholder.

        Returns:
            T | StateAttribute: Value of the attribute.
        """
        if instance is None:
            return self
        if self.name not in instance.__dict__:
            if self.default_factory is not None:
                instance.__dict__[self.name] = self.default_factory()
            else:
                instance.__dict__[self.name] = self.default
        return instance.__dict__[self.name]

    def __set__(self, instance: Any, value: T) -> None:
        """Set the value of the attribute.

        Args:
            instance (Any): An instance of the owner.
            value (T): Value to set.
        """
        instance.__dict__[self.name] = value
        if self.post_set_hook is not None:
            self.post_set_hook(instance, value)
