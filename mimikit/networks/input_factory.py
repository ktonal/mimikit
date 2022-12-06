import abc
from typing import Set, Dict, Iterable, Type, Any, Generic, TypeVar

T = TypeVar("T")


class Configurable(Generic[T]):
    type: Type
    config: Dict[str, Any]


class InputConfig(abc.ABC):
    feature: Configurable["Feature"]

    module: Configurable["nn.Module"]

    as_condition: bool = False


class InputFactory(abc.ABC, Generic[T]):

    @property
    @abc.abstractmethod
    def supported_features(self) -> Set[Configurable["Feature"]]:
        ...

    @abc.abstractmethod
    def supported_modules(self, feature: Configurable["Feature"]
                          ) -> Set[Configurable["nn.Module"]]:
        ...

    @abc.abstractmethod
    def items(self) -> Iterable[str, InputConfig]:
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def __setitem__(self, key: str, value: InputConfig) -> None:
        ...

    @abc.abstractmethod
    def __getitem__(self, item: str) -> InputConfig:
        ...

    @abc.abstractmethod
    def __delitem__(self, item: str) -> None:
        ...

    @abc.abstractmethod
    def build(self, network: T):
        ...

    @property
    @abc.abstractmethod
    def features(self) -> Dict[str, "Feature"]:
        ...

    @abc.abstractmethod
    def modules(self) -> Dict[str, "nn.Module"]:
        ...

