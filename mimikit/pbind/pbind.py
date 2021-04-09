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
from contextvars import ContextVar, copy_context

_rout = ContextVar('rout')
_rout.set(None)

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

    def integrate(self):
        return Pintegrate(self)

    def differentiate(self):
        return Pdifferentiate(self)

    def fin(self, val):
        return Pfin(self, val)

    def finDur(self, val, tol=0.001):
        return Pfindur(self, val, tol)


class Pfin(Pattern):
    def __init__(self, pattern, n):
        Pattern.__init__(self)
        self.n = n
        self.pattern = pattern

    def embedInStream(self, rout):
        stream = patify(self.pattern).asStream()
        inval = rout.inval
        n = patvalue(self.n, inval)
        counter = 0
        while counter < n:
            inval = rout.inval
            val = stream.next(inval)
            if val is EOP:
                yield EOP
            yield val
            counter = counter + 1
        yield EOP

class Pfindur(Pattern):

    def __init__(self, pattern, dur, tolerance=0.001):
        self.pattern = pattern
        self.dur = dur
        self.tolerance = tolerance

    def embedInStream(self, rout):
        elapsed = 0.0
        localdur = patvalue(self.dur, rout.inval)
        stream = patify(self.pattern).asStream()

        while True:
            inevent = stream.next(rout.inval)
            if inevent is EOP:
                yield EOP
            delta = inevent.delta
            nextElapsed = elapsed + delta
            if (nextElapsed + self.tolerance) >= localdur:
                # must always copy an event before altering it.
                # fix delta time and yield to play the event.
                inevent = copy.copy(inevent)
                inevent['delta'] = localdur - elapsed
                yield inevent
                yield EOP
            elapsed = nextElapsed
            yield inevent


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


def embedInStream(rout, obj):
    if isinstance(obj, Pattern):
        return (EMBED, obj, obj.embedInStream(rout))
    else:
        return obj


class Pintegrate(Pattern):

    def __init__(self, pattern, start_val=0):
        Pattern.__init__(self)
        self.start_val = start_val
        self.pattern = pattern

    def embedInStream(self, rout):
        val = patvalue(self.start_val, rout.inval)
        stream = patify(self.pattern).asStream()

        while True:
            item = stream.next(rout.inval)
            if item is EOP:
                yield EOP
            val += item
            yield val


class Pdifferentiate(Pattern):

    def __init__(self, pattern):
        Pattern.__init__(self)
        self.pattern = pattern

    def embedInStream(self, rout):
        stream = patify(self.pattern).asStream()
        prev = stream.next(rout.inval)
        if prev is EOP:
            yield EOP

        while True:
            item = stream.next(rout.inval)
            if item is EOP:
                yield EOP
            yield item - prev
            prev = item


class Pmin(Pattern):
    def __init__(self, *vals):
        Pattern.__init__(self)
        self.vals = vals

    # this is equivalent to the embedInStream function
    def embedInStream(self, rout):
        streams = list([patify(x) for x in self.vals])

        while True:
            vals = list([x.next(rout.inval) for x in streams])
            if EOP in vals:
                yield EOP
            yield min(vals)


class Pmax(Pattern):
    def __init__(self, *vals):
        Pattern.__init__(self)
        self.vals = vals

    # this is equivalent to the embedInStream function
    def embedInStream(self, rout):
        streams = list([patify(x) for x in self.vals])

        while True:
            vals = list([x.next(rout.inval) for x in streams])
            if EOP in vals:
                yield EOP
            yield max(vals)


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
        self.op = op
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
        self.op = op
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

    def __init__(self, pat, override_inval=None):
        if isinstance(pat, numbers.Number):
            pat = Pconst(pat)
        self.stack = []
        self.time = 0.0
        rout = _rout.get()
        if rout:
            self.ctx = rout.ctx
            self.gen = pat.embedInStream(rout)
        else:
            def set_rout():
                _rout.set(self)
            self.ctx = copy_context()
            self.ctx.run(set_rout)
            self.gen = pat.embedInStream(self)
        self.stack = []
        self.override_inval = override_inval
        self.inval = override_inval
        self.time = 0.0

    def __iter__(self):
        def iterfunc():
            while True:
                elt = self.next(self.override_inval.copy())
                if elt is EOP:
                    return
                yield elt
        return iterfunc()

    def push(self):
        self.stack.append(self.gen)

    def all(self, val=None):
        res = []
        val = val or self.override_inval
        item = self.next(val)
        while item is not None:
            res.append(item)
            item = self.next(val)
        return res

    def __next__(self):
        return self.next(self.override_inval)

    def next(self, val=None):
        rout = _rout.get()
        if not rout:
            self.inval = val
            res = self.ctx.run(next, self.gen)
        else:
            rout.inval = val
            res = next(self.gen)
        if res is EOP:
            if self.stack != []:
                self.gen = self.stack.pop()
                return self.next(val)
            else:
                return EOP
        elif isinstance(res, tuple) and (res[0] is EMBED):
            self.push()
            self.gen = res[2]
            return self.next(val)
        else:
            self.time += getattr(res, 'delta', None) or 0.0
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
                if streamout is EOP:
                    yield streamout
                # support tupled
                if isinstance(name, (list, tuple)):
                    if len(name) > len(streamout):
                        print("the pattern is not providing enough values to assign to the key set:" + name)
                        return
                    for i, key in enumerate(name):
                        event[key] = streamout[i]
                else:
                    event[name] = streamout
            yield event

