# This is a partial port of SuperCollider's Pattern-Stream-Event system.

import sys
import copy
import random
import inspect
import itertools
import numbers
import numpy as np
from copy import copy
import heapq

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
    def __init__(self):
        self.nextarg = False

    def asStream(self):
        return SCRoutine(self)

    def __iter__(self):
        return SCRoutine(self)

    def clip(self, mini, maxi):
        return Pclip(self, mini, maxi)

    def wrap(self, mini, maxi):
        return Pwrap(self, mini, maxi)

    def fold(self, mini, maxi):
        return Pfold(self, mini, maxi)

    def __add__(self, operand):
        """Binary op: add two patterns"""
        return Padd(self, operand)

    def __radd__(self, operand):
        """Binary op: add two patterns"""
        return self.__add__(operand)

    def __mul__(self, operand):
        """Binary op: multiply two patterns"""
        return Pmul(self, operand)

    def __rmul__(self, operand):
        """Binary op: multiply two patterns"""
        return self.__mul__(operand)

    def __div__(self, operand):
        """Binary op: divide two patterns"""
        return Pdiv(self, operand)

    def __rdiv__(self, operand):
        """Binary op: divide two patterns"""
        return self.__div__(operand)

    def __sub__(self, operand):
        """Binary op: subtract two patterns"""
        return Psub(self, operand)

    def __rsub__(self, operand):
        """Binary op: subtract two patterns"""
        return self.__sub__(operand)

    def __mod__(self, operand):
        """Modulo"""
        return Pmod(self, operand)

    def __rmod__(self, operand):
        """Modulo (as operand)"""
        return operand.__mod__(self)

    def __rpow__(self, operand):
        """Power (as operand)"""
        return operand.__pow__(self)

    def __pow__(self, operand):
        """Power"""
        return Ppow(self, operand)

    def __rfloordiv__(self, operand):
        """Integer division"""
        return operand.__floordiv__(self)

    def __floordiv__(self, operand):
        """Integer division"""
        return Pfloordiv(self, operand)


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


class Pseq(Pattern):
    def __init__(self, lst, repeats=1, offset=0):
        Pattern.__init__(self)
        self.lst = lst
        self.offset = offset
        self.repeats = repeats

    # this is equivalent to the embedInStream function
    def embedInStream(self, rout):
        reps = patvalue(self.repeats)
        offsetValue = patvalue(self.offset)
        lsize = len(self.lst)
        counter = 0
        while counter < reps:
            for i in range(lsize):
                item = self.lst[(i + offsetValue) % lsize]
                yield embedInStream(rout, item)
            counter = counter + 1
        yield EOP


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
                elt = self.next(self.override_inval)
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
        end_of_stream = False
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

    def asStream(self, override_inval=None):
        return SCRoutine(self, override_inval)

    def embedInStream(self, rout):
        streampairs = list(self.patternpairs)
        endval = len(streampairs)

        for i in range(1, endval, 2):
            streampairs[i] = patify(streampairs[i]).asStream()

        while True:
            inval = rout.inval
            print('pbind', inval)
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


class Pfunc(Pattern):
    def __init__(self, func, repeats=inf):
        Pattern.__init__(self)
        self.func = func
        self.repeats = repeats

    def embedInStream(self, rout):
        counter = 0
        while counter < self.repeats:
            if self.func.__code__.co_argcount == 0:
                yield self.func()
            else:
                yield self.func(rout.inval)
            counter += 1
        yield EOP


class Plazy(Pattern):
    def __init__(self, func):
        Pattern.__init__(self)
        self.func = func

    def embedInStream(self, rout):
        if self.func.__code__.co_argcount == 0:
            yield embedInStream(rout, self.func())
        else:
            yield embedInStream(rout, self.func(rout.inval))
        yield EOP


class Pkey(Pattern):
    def __init__(self, key, default=None):
        Pattern.__init__(self)
        self.key = key
        self.default = default

    def embedInStream(self, rout):
        yield rout.inval.get(self.key, self.default)
        yield EOP


class Pinsteps(Pattern):
    def __init__(self, pattern, steps):
        Pattern.__init__(self)
        self.pattern = pattern
        self.steps = steps

    def embedInStream(self, rout):
        stream = patify(self.pattern).asStream()
        step_stream = patify(self.steps).asStream()
        last_event = stream.next(rout.inval)
        if last_event is EOP:
            yield EOP
        yield last_event
        if isinstance(last_event, list):
            is_list = True
            last_event = np.asarray(last_event).astype(np.float64)
        else:
            is_list = False
        while True:
            out_event = stream.next(rout.inval)
            steps = step_stream.next(rout.inval)
            if out_event is EOP or steps is EOP:
                yield EOP
            if is_list or isinstance(out_event, list):
                is_list = True
                out_event = np.asarray(out_event).astype(np.float64)
            incr = (out_event - last_event) / steps
            for _ in range(steps):
                last_event += incr
                yield last_event
 

class Pseq(Pattern):
    def __init__(self, lst, repeats=1, offset=0):
        Pattern.__init__(self)
        self.lst = lst
        self.offset = offset
        self.repeats = repeats

    # this is equivalent to the embedInStream function
    def embedInStream(self, rout):
        counter = 0
        reps = patvalue(self.repeats)
        lsize = len(self.lst)
        offsetValue = patvalue(self.offset)
        while counter < reps:
            for i in range(lsize):
                item = self.lst[(i + offsetValue) % lsize]
                yield embedInStream(rout, item)
            counter = counter + 1
        yield EOP


class Ptuple(Pattern):
    def __init__(self, lst, repeats=1):
        Pattern.__init__(self)
        self.lst = lst
        self.repeats = repeats

    def embedInStream(self, rout):
        counter = 0
        reps = patvalue(self.repeats)
        while counter < reps:
            streams = [patify(item).asStream() for item in self.lst]
            saw_nil = False
            while not saw_nil:
                tup = []
                inval = rout.inval
                for s in streams:
                    outval = s.next(inval)
                    if outval is EOP:
                        saw_nil = True
                        break
                    tup.append(outval)
                if not saw_nil:
                    yield tup
            counter = counter + 1
        yield EOP


class Pn(Pattern):
    def __init__(self, val, repeats=1):
        Pattern.__init__(self)
        self.val = val
        self.repeats = repeats

    # this is equivalent to the embedInStream function
    def embedInStream(self, rout):
        reps = patvalue(self.repeats)
        for _ in range(reps):
            yield embedInStream(rout, self.val)
        yield EOP


class Pser(Pattern):
    def __init__(self, lst, repeats=1, offset=0):
        Pattern.__init__(self)
        self.lst = lst
        self.offset = offset
        self.repeats = repeats

    # this is equivalent to the embedInStream function
    def embedInStream(self, rout):
        counter = 0
        reps = patvalue(self.repeats)
        lsize = len(self.lst)
        offsetValue = patvalue(self.offset)
        while counter < reps:
            item = self.lst[(counter + offsetValue) % lsize]
            yield embedInStream(rout, item)
            counter = counter + 1
        yield EOP


class Prand(Pattern):
    def __init__(self, lst, repeats=1):
        Pattern.__init__(self)
        self.lst = lst
        self.repeats = repeats

    # this is equivalent to the embedInStream function
    def embedInStream(self, rout):
        counter = 0
        inval = rout.inval
        reps = patvalue(self.repeats, inval)
        # lsize = len(self.lst)
        while counter < reps:
            item = random.choice(self.lst)
            yield embedInStream(rout, item)
            counter = counter + 1
            inval = rout.inval
        yield EOP


class Pxrand(Pattern):
    def __init__(self, lst, repeats=1):
        Pattern.__init__(self)
        self.lst = lst
        self.repeats = repeats

    # this is equivalent to the embedInStream function
    def embedInStream(self, rout):
        counter = 0
        prev = None
        inval = rout.inval
        reps = patvalue(self.repeats, inval)
        # lsize = len(self.lst)
        while counter < reps:
            item = random.choice(self.lst)
            while prev == item:
                item = random.choice(self.lst)
            yield embedInStream(rout, item)
            prev = item
            counter = counter + 1
            inval = rout.inval
        yield EOP


class Pwrand(Pattern):
    def __init__(self, lst, weights, repeats=1):
        Pattern.__init__(self)
        self.lst = lst
        self.weights = weights
        self.repeats = repeats

    # this is equivalent to the embedInStream function
    def embedInStream(self, rout):
        counter = 0
        inval = rout.inval
        reps = patvalue(self.repeats, inval)
        while counter < reps:
            item = wchoice(self.lst, self.weights)
            yield embedInStream(rout, item)
            counter = counter + 1
            inval = rout.inval
        yield EOP


class Pconst(Pattern):
    def __init__(self, val):
        Pattern.__init__(self)
        self.val = val

    # this is equivalent to the embedInStream function
    def embedInStream(self, rout):
        while True:
            yield self.val


# adverbs are not (yet) implemented
class Pbinop(Pattern):

    op = '__add__'

    def __init__(self, a, b):
        Pattern.__init__(self)
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
            yield getattr(val1, cls.op)(val2)


class Pfloordiv(Pbinop):
    op = '__floordiv__'


class Padd(Pbinop):
    op = '__add__'


class Pmul(Pbinop):
    op = '__mul__'


class Pdiv(Pbinop):
    op = '__div__'


class Pmod(Pbinop):
    op = '__mod__'


class Psub(Pbinop):
    op = '__sub__'


class Ppow(Pbinop):
    op = '__pow__'


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



class FilterPattern(Pattern):
    def __init__(self, pattern):
        Pattern.__init__(self)
        self.pattern = pattern


class Pfin(FilterPattern):
    def __init__(self, pattern, n):
        FilterPattern.__init__(self, pattern)
        self.n = n

    def embedInStream(self, rout):
        stream = patify(self.pattern).asStream()
        inval = rout.inval
        n = patvalue(self.n, inval)
        counter = 0
        while counter < n:
            val = stream.next(inval)
            if val is EOP:
                yield EOP
            yield val
            counter = counter + 1


class Pstutter(FilterPattern):
    def __init__(self, n, pattern):
        FilterPattern.__init__(self, pattern)
        self.n = n

    def embedInStream(self, rout):
        stream = patify(self.pattern).asStream()
        nstr = patify(self.n).asStream()
        while True:
            inval = rout.inval
            num = nstr.next(inval)
            val = stream.next(inval)
            if (num is EOP) or (val is EOP):
                yield EOP
            for k in range(num):
                yield val


class Pcollect(FilterPattern):
    def __init__(self, func, pattern):
        FilterPattern.__init__(self, pattern)
        self.func = func

    def embedInStream(self, rout):
        stream = patify(self.pattern).asStream()
        while True:
            inval = rout.inval
            val = stream.next(inval)
            if val is EOP:
                yield EOP
            yield self.func(val)


class Pwhite(Pattern):
    def __init__(self, lo=0.0, hi=1.0, repeats=inf):
        Pattern.__init__(self)
        self.lo = lo
        self.hi = hi
        self.repeats = repeats

    # this is equivalent to the embedInStream function
    def embedInStream(self, rout):
        counter = 0
        reps = patvalue(self.repeats)
        lostr = patify(self.lo).asStream()
        histr = patify(self.hi).asStream()
        while counter < reps:
            inval = rout.inval
            loval = lostr.next(inval)
            hival = histr.next(inval)
            if (loval is EOP) or (hival is EOP):
                yield EOP
            if (type(loval) == int) and (type(hival) == int):
                yield random.randint(loval, hival)
            else:
                yield random.uniform(loval, hival)
            counter = counter + 1
        yield EOP


class Pbrown(Pattern):
    def __init__(self, lo=0.0, hi=1.0, step=0.125, repeats=inf):
        Pattern.__init__(self)
        self.lo = lo
        self.hi = hi
        self.step = step
        self.repeats = repeats

    # this is equivalent to the embedInStream function
    def embedInStream(self, rout):
        counter = 0
        reps = patvalue(self.repeats)
        lostr = patify(self.lo).asStream()
        histr = patify(self.hi).asStream()
        stepstr = patify(self.step).asStream()
        inval = rout.inval
        loval = lostr.next(inval)
        hival = histr.next(inval)
        stepval = stepstr.next(inval)
        if (loval is EOP) or (hival is EOP) or (stepval is EOP):
            yield EOP
        if (type(loval) == int) and (type(hival) == int) and (type(stepval) == int):
            curval = random.randint(loval, hival)
        else:
            curval = random.uniform(loval, hival)
        yield curval
        while counter < reps:
            inval = rout.inval
            loval = lostr.next(inval)
            hival = histr.next(inval)
            stepval = stepstr.next(inval)
            if (loval is EOP) or (hival is EOP) or (stepval is EOP):
                yield EOP
            if (type(loval) == int) and (type(hival) == int) and (type(stepval) == int):
                curval = fold(curval + random.randint(-stepval, stepval), loval, hival)
                yield curval
            else:
                curval = fold(curval + random.uniform(-stepval, stepval), loval, hival)
                yield curval
            counter = counter + 1
        yield EOP


class Pshuf(Pattern):
    def __init__(self, lst, repeats=1):
        Pattern.__init__(self)
        self.lst = lst
        self.repeats = repeats

    def embedInStream(self, rout):
        counter = 0
        reps = patvalue(self.repeats)
        lsize = len(self.lst)
        ll = list(self.lst)
        random.shuffle(ll)
        while counter < reps:
            for i in range(lsize):
                item = ll[i]
                yield embedInStream(rout, item)
            counter = counter + 1
        yield EOP


class Pindex(Pattern):
    def __init__(self, lst, index):
        Pattern.__init__(self)
        self.lst = lst
        self.index = index

    def embedInStream(self, rout):
        lsize = len(self.lst)
        inval = rout.inval
        idxstr = patify(self.index).asStream()
        while True:
            idx = idxstr.next(inval)
            if idx is EOP:
                yield EOP
                yield inval
            yield embedInStream(rout, self.lst[idx % lsize])


class Pseries(Pattern):
    def __init__(self, start, step, length):
        Pattern.__init__(self)
        self.start = start
        self.step = step
        self.length = length

    def embedInStream(self, rout):
        inval = rout.inval
        cur = patvalue(self.start, inval)
        stepstr = patify(self.step).asStream()
        leng = patvalue(self.length, inval)
        for _ in range(leng):
            yield cur
            inval = rout.inval
            stp = stepstr.next(inval)
            if stp is EOP:
                yield EOP
            cur = cur + stp
        yield EOP


class Pgeom(Pattern):
    def __init__(self, start, step, length):
        Pattern.__init__(self)
        self.start = start
        self.step = step
        self.length = length

    def embedInStream(self, rout):
        inval = rout.inval
        cur = patvalue(self.start, inval)
        stepstr = patify(self.step).asStream()
        leng = patvalue(self.length, inval)
        for _ in range(leng):
            yield cur
            inval = rout.inval
            stp = stepstr.next(inval)
            if stp is EOP:
                yield EOP
            cur = cur * stp
        yield EOP


class Prepscheme(FilterPattern):
    def __init__(self, groupsize, scheme, pattern, reset=False):
        FilterPattern.__init__(self, pattern)
        self.groupsize = groupsize
        self.scheme = scheme
        self.reset = reset
        
    def embedInStream(self, rout):
        stream = patify(self.pattern).asStream()
        gsizestream = patify(self.groupsize).asStream()
        gschemestream = patify(self.groupscheme).asStream()
        resetstream = patify(self.reset).asStream()
        groups = [None] * 16

        while True:
            event = rout.inval
            gscheme = gschemestream.next(event)
            resetval = resetstream.next(event)
            if resetval:
                groups = [None] * 16
            while gscheme.isNil or (gscheme == 'take'):
                groups = [None] * 16
                if gscheme is EOP:
                    gschemestream = patify(self.groupscheme).asStream()
                gscheme = gschemestream.next(event)
            group = groups[gscheme]
            gsize = gsizestream.next(event)
            if gsize is EOP:
                yield EOP
            for k in range(gsize):
                if k >= len(group):
                    val = stream.next(event)
                    if val is EOP:
                        yield EOP
                    group = group.append(val)
                    groups[gscheme] = group
                    yield val
                    event = rout.inval
                else:
                    yield groups.at(gscheme).at(k)
                    event = rout.inval


class Ppatlace(Pseq):
    def embedInStream(self, rout):
        consecutiveNils = 0
        streamList = list([patify(item).asStream() for item in self.lst])
        offsetValue = patvalue(self.offset)
        localRepeats = patvalue(self.repeats)
        length = len(self.lst)
        index = repeat = 0
        while repeat < localRepeats and consecutiveNils < length:
            inval = rout.inval
            item = streamList[(offsetValue + index) % length]
            item = item.next(inval)
            if item is not EOP:
                consecutiveNils = 0
                inval = embedInStream(rout, item)
            else:
                consecutiveNils = consecutiveNils + 1
            index = index + 1
            if index == length:
                index = 0
                repeat = repeat + 1
        yield EOP


class Pchain(Pattern):

    def __init__(self, *patterns):
        self.patterns = patterns

    def embedInStream { arg inval;
        streams = reversed[ patify(pattern).asStream() for pattern in patterns ]
        while True:
            inevent = copy(rout.inval)
            for stream in streams:
                inevent = stream.next(inevent);
                if inevent is EOP:
                    yield EOP
            yield(inevent);


class Ppar(Pattern):
    
    def __init__(self, lst, repeats=1):
        self.lst = lst
        self.repeats = repeats

    def initStreams(self, priorityQ):
        for pattern in self.lst:
            priorityQ.append((0.0, patify(pattern).asStream()))
        heapq.heapify(priorityQ)

    def embedInStream(self, inval):
        priorityQ = []
        
        repeats.value(inval).do({ arg j;
            now = 0.0;

        this.initStreams(priorityQ);

        # if first event not at time zero
        if (priorityQ.notEmpty and: { (nexttime = priorityQ.topPriority) > 0.0 }) {
            outval = Event.silent(nexttime, inval);
            inval = outval.yield;
            now = nexttime;


        while { priorityQ.notEmpty } {
            stream = priorityQ.pop;
            outval = stream.next(inval).asEvent;
            if (outval.isNil) {
                nexttime = priorityQ.topPriority;
                if (nexttime.notNil, {
                    # that child stream ended, so rest until next one
                    outval = Event.silent(nexttime - now, inval);
                    inval = outval.yield;
                    now = nexttime;
                else:
                    priorityQ.clear;
                # requeue stream
                priorityQ.put(now + outval.delta, stream);
                nexttime = priorityQ.topPriority;
                outval.put(\delta, nexttime - now);

                inval = outval.yield;
                # inval ?? { this.purgeQueue(priorityQ); ^nil.yield };
                now = nexttime;
        yield EOP

# Ppar Pindexlace Pdftsm Plocperm integrate differentiate Pbeta exponential distribution
