import numpy as np
from numbers import Number
from .pbind import Pattern, EOP, patify


class Warp:

    def __init__(self, min_val=0.0, max_val=1.0, warp='lin'):
        self.min_val = min_val
        self.max_val = max_val
        self.warp = None
        self.set_warp(warp)
        self.update(min_val, max_val, warp)

    def set_warp(self, warp):
        if warp is not self.warp:
            if isinstance(warp, Number):
                warpclz = CurveWarp
            else:
                warpclz = self.warps[warp]
            self.map = warpclz.map.__get__(self)
            self.unmap = warpclz.unmap.__get__(self)
            if hasattr(warpclz, 'update'):
                self.update = warpclz.update.__get__(self)
            self.warp = warp

    def update(self, minval, maxval, warp):
        pass


class Pwarp(Pattern):

    def __init__(self, min_val=0.0, max_val=1.0, warp='lin', pattern=None):
        self.min_val = min_val
        self.max_val = max_val
        self.warp = warp
        self.pattern = pattern

    def embedInStream(self, rout):
        min_val_str = patify(self.min_val).asStream()
        max_val_str = patify(self.max_val).asStream()
        warp_str = patify(self.warp).asStream()
        stream = patify(self.pattern).asStream()
        
        inval = rout.inval
        min_val = min_val_str.next(inval)
        max_val = max_val_str.next(inval)
        warp = warp_str.next(inval)
        if EOP in [min_val, max_val, warp]:
            yield EOP
        warper = Warp(min_val, max_val, warp)

        while True:
            val = stream.next(inval)
            if val is EOP:
                yield EOP
            yield warper.map(val)
            inval = rout.inval
            min_val = min_val_str.next(inval)
            max_val = max_val_str.next(inval)
            warp = warp_str.next(inval)
            if EOP in [min_val, max_val, warp]:
                yield EOP
            warper.set_warp(warp)
            warper.update(min_val, max_val, warp)


class LinearWarp:

    def map(self, value):
        range = self.max_val - self.min_val
        return value * range + self.min_val

    def unmap(self, value):
        range = self.max_val - self.min_val
        return (value - self.min_val) / range


class ExponentialWarp:

    def map(self, value):
        return ((self.max_val / self.min_val) ** value) * self.min_val

    def unmap(self, value):
        return np.log(value / self.min_val) / np.log(self.max_val / self.min_val)


class CurveWarp:

    def update(self, min_val=0.0, max_val=1.0, curve=-2):
        if abs(curve) < 0.001:
            curve = 0.0005
        self.curve = curve
        self.grow = np.exp(curve)
        self.a = (self.max_val - self.min_val) / (1.0 - self.grow)
        self.b = self.min_val + self.a

    def map(self, value):
        return self.b - self.a * pow(self.grow, value)

    def unmap(self, value):
        return np.log((self.b - value) / self.a) / self.curve


class CosineWarp:
    def map(self, value):
        return super().map(0.5 - (np.cos(np.pi * value) * 0.5))

    def unmap(self, value):
        return np.acos(1.0 - (super().unmap(value) * 2.0)) / np.pi


class SineWarp:
    def map(self, value):
        return super().map(np.sin(0.5 * np.pi * value))

    def unmap(self, value):
        return np.asin(super().unmap(value)) / 0.5 * np.pi


Warp.warps = {'lin': LinearWarp, 'exp': ExponentialWarp, 'sin': SineWarp,
	      'cos': CosineWarp, 'linear': LinearWarp, 'exponential': ExponentialWarp}
