import dataclasses as dtc
from typing import Union, Any, Optional


__all__ = [
    "Sample",
    "Frame",
    "Step",
    "Second",
    "Unit",
    "ItemSpec"
]


class _Unit:
    _order = ("Sample", "Frame", "Second", "Step")

    def __lt__(self, other):
        return self._order.index(type(self).__name__) < self._order.index(type(other).__name__)


@dtc.dataclass
class Sample(_Unit):
    sr: Optional[int]

    def __hash__(self):
        return hash(repr(self))


@dtc.dataclass
class Frame(_Unit):
    frame_size: int
    hop_length: int
    padding: Optional[Any] = None

    def __hash__(self):
        return hash(repr(self))


@dtc.dataclass
class Second(_Unit):
    sr: Optional[int]

    def __hash__(self):
        return hash(repr(self))


@dtc.dataclass
class Step(_Unit):
    def __hash__(self):
        return hash(repr(self))


Unit = Union[Sample, Frame, Second, Step]


def convert(
        x: Union[int, float],
        from_unit: Unit,
        to_unit: Unit,
        as_length: bool,
):
    def _get_extra(f: Frame):
        if as_length:
            return (f.frame_size - f.hop_length) * int(not bool(f.padding))
        return 0

    def _get_sr(u: Unit, v: Unit):
        sr = {x.sr for x in (u, v) if x.sr is not None}
        assert len(sr) == 1, f"couldn't find a single sr: {u}, {v}"
        return sr.pop()

    from_ = type(from_unit)
    to_ = type(to_unit)

    if from_ is Sample:
        if to_ is Frame:
            fs, hl = to_unit.frame_size, to_unit.hop_length
            x -= _get_extra(to_unit)
            return int(x // hl)
        elif to_ is Second:
            return x / _get_sr(from_unit, to_unit)
        else:
            return x

    elif from_ is Frame:
        fs, hl = from_unit.frame_size, from_unit.hop_length
        if to_ is Sample:
            return int(x * hl) + _get_extra(from_unit)
        elif to_ is Second:
            return (x * hl + _get_extra(from_unit)) / to_unit.sr
        else:
            return x

    elif from_ is Second:
        if to_ is Frame:
            sr = from_unit.sr
            return (int(x * sr) - _get_extra(to_unit)) // to_unit.hop_length
        elif to_ is Sample:
            return int(x * _get_sr(to_unit, from_unit))
        elif to_ is Step:
            raise TypeError(f"can not convert seconds to steps")

    elif from_ is Step:
        if to_ is Step:
            raise TypeError(f"can not convert steps to seconds")
        return x


@dtc.dataclass
class ItemSpec:
    shift: Union[int, float] = 0
    length: Union[int, float] = 0
    stride: Union[int, float] = 1
    unit: Unit = Step()

    def __add__(self, other):
        if not isinstance(other, ItemSpec):
            raise TypeError(f"Expected other to be of type ItemSpec."
                            f" Got {type(other)}")
        if isinstance(self.unit, type(other.unit)) and self.unit != other.unit:
            raise ValueError(f"Can not add unit of the same type parametrized differently:\n"
                             f" {self.unit} and {other.unit}")

        target_unit = min(self.unit, other.unit)
        if target_unit == self.unit:
            if other.unit != self.unit:
                a, b = self, other.to(target_unit)
            else:
                a, b = self, other
        else:
            a, b = self.to(target_unit), other
        return ItemSpec(
            a.shift + b.shift,
            a.length + b.length,
            max(a.stride, b.stride),
            target_unit
        )

    def to(self, unit: Unit):
        return ItemSpec(
            shift=convert(self.shift, self.unit, unit, as_length=False),
            length=convert(self.length, self.unit, unit, as_length=True),
            stride=self.stride,
            unit=unit
        )
