# This is a partial port of SuperCollider's Pattern-Stream-Event system.

import sys
import copy
import random
import inspect
import itertools
import numbers
import numpy as np
import heapq
from itertools import cycle

from .event import Event

inf = float("inf")

EOP = object()
EMBED = object()


def wrap(val, lo, hi):
    """
    Wrap ``val`` into range ``lo`` ... ``hi``
    """
    return lo + np.remainder(val, (hi - lo))


def clip(val, lo, hi):
    """
    Clip ``val`` into range ``lo`` ... ``hi``
    This is just a call to np.clip.  It is only provided to make some snippets of
    code from supercollider work without modifications.
    """
    return np.clip(val, lo, hi)


def fold(val, lo, hi):
    """
    Fold a ``val`` into the range ``lo`` .. ``hi``
    """
    return np.remainder(val - lo, hi - lo) + lo


def choose(lst):
    return random.choice(lst)


def choosefun(lst):
    return lambda: random.choice(lst)


def rrand(a, b):
    if (type(a) == int) and (type(b) == int):
        return random.randint(a, b)
    else:
        return random.uniform(a, b)


def rrandfun(a, b):
    return lambda: random.randint(a, b) if (type(a) == int) and (type(b) == int) else random.uniform(a, b)


def normalize(array):
    """ Normalise an array to sum to 1.0. """
    if sum(array) == 0:
        return array
    array_sum = sum(array)
    return np.array(list(map(lambda n: float(n) / array_sum, array)))


def windex(weights):
    """ Return a random index based on a list of weights, from 0..(len(weights) - 1).
    Assumes that weights is already normalised. """
    n = np.random.uniform(0, 1)
    for i in range(len(weights)):
        if n < weights[i]:
            return i
        n = n - weights[i]


def wnindex(weights):
    """ Returns a random index based on a list of weights. 
    Normalises list of weights before executing. """
    wnorm = normalize(weights)
    return windex(wnorm)


def wchoice(array, weights):
    """ Performs a weighted choice from a list of values (assumes pre-normalised weights) """
    index = windex(weights)
    return array[index]


def wnchoice(array, weights):
    """ Performs a weighted choice from a list of values
    (does not assume pre-normalised weights). """
    index = wnindex(weights)
    return array[index]


class Pattern:

    def asStream(self, override_inval=None):
        return SCRoutine(self, override_inval or Event.default)

    def __iter__(self):
        return iter(SCRoutine(self, Event.default))

    def clip(self, mini, maxi):
        return Pclip(self, mini, maxi)

    def wrap(self, mini, maxi):
        return Pwrap(self, mini, maxi)

    def fold(self, mini, maxi):
        return Pfold(self, mini, maxi)

    def __add__(self, operand):
        """Binary op: add two patterns"""
        return Pbinop('__add__', self, operand)

    def __radd__(self, operand):
        """Binary op: add two patterns"""
        return self.__add__(operand)

    def __mul__(self, operand):
        """Binary op: multiply two patterns"""
        return Pbinop('__mul__', self, operand)

    def __rmul__(self, operand):
        """Binary op: multiply two patterns"""
        return self.__mul__(operand)

    def __div__(self, operand):
        """Binary op: divide two patterns"""
        return Pbinop('__div__', self, operand)

    def __rdiv__(self, operand):
        """Binary op: divide two patterns"""
        return self.__div__(operand)

    def __sub__(self, operand):
        """Binary op: subtract two patterns"""
        return Pbinop('__sub__', self, operand)

    def __rsub__(self, operand):
        """Binary op: subtract two patterns"""
        return self.__sub__(operand)

    def __mod__(self, operand):
        """Modulo"""
        return Pbinop('__mod__', self, operand)

    def __rmod__(self, operand):
        """Modulo (as operand)"""
        return operand.__mod__(self)

    def __rpow__(self, operand):
        """Power (as operand)"""
        return operand.__pow__(self)

    def __pow__(self, operand):
        """Power"""
        return Pbinop('__pow__', self, operand)

    def __rfloordiv__(self, operand):
        """Integer division"""
        return operand.__floordiv__(self)

    def __floordiv__(self, operand):
        """Integer division"""
        return Pbinop('__floordiv__', self, operand)

    def __abs__(self, operand):
        return Punop('__abs__', self)

    def __trunc__(self, operand):
        return Punop('__trunc__', self)

    def __ceil__(self, operand):
        return Punop('__ceil__', self)

    def __floor__(self, operand):
        return Punop('__floor__', self)

    def __invert__(self, operand):
        return Punop('__invert__', self)

    def __int__(self, operand):
        return Punop('__int__', self)

    def __float__(self, operand):
        return Punop('__float__', self)


def patify(val):
    if issubclass(type(val), Pattern):
        return val
    else:
        return Pconst(val)


# function to evaluate repeat and duration arguments
def patvalue(val, inval=None):
    if isinstance(val, numbers.Number):
        return val
    elif isinstance(val, Pattern):
        return val.asStream().next()
    else:
        if len(inspect.getargspec(lambda: 3).args):
            return val(inval)
        else:
            return val()


def embedInStream(rout, obj, val=None):
    if isinstance(obj, Pattern):
        return (EMBED, obj, obj.embedInStream(rout))
    else:
        return obj


class Pconst(Pattern):
    def __init__(self, val):
        Pattern.__init__(self)
        self.val = val

    # this is equivalent to the embedInStream function
    def embedInStream(self, rout):
        while True:
            yield self.val


class Punop(Pattern):

    def __init__(self, op, a):
        Pattern.__init__(self)
        self.op = a
        self.op1 = a

    def embedInStream(self, rout):
        op1str = patify(self.op1).asStream()
        while True:
            inval = rout.inval
            val1 = op1str.next(inval)
            if val1 is EOP:
                yield EOP
            # broadcast arithmetic
            if type(val1) is list:
                yield list([getattr(v1, self.op)() for v1 in val1])
            else:
                yield getattr(val1, self.op)()
    

# adverbs are not (yet) implemented
class Pbinop(Pattern):

    def __init__(self, op, a, b):
        Pattern.__init__(self)
        self.op = a
        self.op1 = a
        self.op2 = b

    def embedInStream(self, rout):
        op1str = patify(self.op1).asStream()
        op2str = patify(self.op2).asStream()
        while True:
            inval = rout.inval
            val1 = op1str.next(inval)
            val2 = op2str.next(inval)
            if (val1 is EOP) or (val2 is EOP):
                yield EOP
                # broadcast arithmetic
            if type(val1) is list:
                len1 = len(val1)
                if type(val2) is list:
                    len2 = len(val2)
                    zip_list = zip(val1, cycle(val2)) if len1 > len2 else zip(cycle(val1), val2)
                    yield list([getattr(v1, self.op)(v2) for v1, v2 in zip_list])
                else:
                    yield list([getattr(v1, self.op)(val2) for v1 in val1])
            elif type(val2) is list:
                yield list([getattr(val1, self.op)(v2) for v2 in val2])
            else:
                yield getattr(val1, self.op)(val2)


class Pclip(Pattern):
    def __init__(self, pattern, mini, maxi):
        Pattern.__init__(self)
        self.pattern = pattern
        self.mini = mini
        self.maxi = maxi

    def embedInStream(self, rout):
        mini_stream = patify(self.mini).asStream()
        maxi_stream = patify(self.maxi).asStream()
        stream = patify(self.pattern).asStream()
        while True:
            inval = rout.inval
            val = stream.next(inval)
            min_val = mini_stream.next(inval)
            max_val = maxi_stream.next(inval)
            if (val is EOP) or (min_val is EOP) or (max_val is EOP):
                yield EOP
            yield clip(val, min_val, max_val)


class Pfold(Pattern):
    def __init__(self, pattern, mini, maxi):
        Pattern.__init__(self)
        self.pattern = pattern
        self.mini = mini
        self.maxi = maxi

    def embedInStream(self, rout):
        mini_stream = patify(self.mini).asStream()
        maxi_stream = patify(self.maxi).asStream()
        stream = patify(self.pattern).asStream()
        while True:
            inval = rout.inval
            val = stream.next(inval)
            min_val = mini_stream.next(inval)
            max_val = maxi_stream.next(inval)
            if (val is EOP) or (min_val is EOP) or (max_val is EOP):
                yield EOP
            yield fold(val, min_val, max_val)


class Pwrap(Pattern):
    def __init__(self, pattern, mini, maxi):
        Pattern.__init__(self)
        self.pattern = pattern
        self.mini = mini
        self.maxi = maxi

    def embedInStream(self, rout):
        mini_stream = patify(self.mini).asStream()
        maxi_stream = patify(self.maxi).asStream()
        stream = patify(self.pattern).asStream()
        while True:
            inval = rout.inval
            val = stream.next(inval)
            min_val = mini_stream.next(inval)
            max_val = maxi_stream.next(inval)
            if (val is EOP) or (min_val is EOP) or (max_val is EOP):
                yield EOP
            yield wrap(val, min_val, max_val)


class SCRoutine:
    def __init__(self, rout, override_inval=None):
        if isinstance(rout, numbers.Number):
            rout = Pconst(rout)
        self.stack = []
        self.rout = rout
        self.inval = None
        self.override_inval = override_inval
        self.gen = rout.embedInStream(self)

    def __iter__(self):
        def iterfunc():
            while True:
                elt = self.next(self.override_inval.copy())
                if elt == EOP:
                    return
                yield elt
        return iterfunc()

    def push(self):
        self.stack.append((self.rout, self.gen))

    def all(self, val=None):
        res = []
        item = self.next(val)
        while item is not None:
            res.append(item)
            item = self.next(val)
        return res

    def __next__(self):
        return self.next(self.override_inval)

    def next(self, val=None):
        self.inval = val
        res = next(self.gen)
        if res == EOP:
            self.inval = val
            if self.stack != []:
                r, g = self.stack.pop()
                self.rout = r
                self.gen = g
                return self.next(val)
            else:
                return EOP
        elif isinstance(res, tuple) and (res[0] == EMBED):
            self.push()
            self.rout = res[1]
            self.gen = res[2]
            return self.next(val)
        else:
            return res


class Pbind(Pattern):
    def __init__(self, *args):
        Pattern.__init__(self)
        self.patternpairs = args

    def embedInStream(self, rout):
        streampairs = list(self.patternpairs)
        endval = len(streampairs)

        for i in range(1, endval, 2):
            streampairs[i] = patify(streampairs[i]).asStream()

        while True:
            inval = rout.inval
            if inval is None:
                return
            event = inval.copy()
            for i in range(0, endval, 2):
                name = streampairs[i]
                stream = streampairs[i + 1]
                streamout = stream.next(event)
                if streamout == EOP:
                    yield streamout
                # support tupled
                if isinstance(name, (list, tuple)):
                    if len(name) > len(streamout):
                        print("the pattern is not providing enough values to assign to the key set:" + name)
                        return
                    # here we would iterate over the name
                    for i, key in enumerate(name):
                        event[key] = streamout[i]
                else:
                    event[name] = streamout
            yield event

