# This is a partial port of SuperCollider's Pattern-Stream-Event system.

import sys
import copy
import random
import inspect
import itertools
import numbers
import numpy as np

inf = float("inf")


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
    return np.array(list(map(lambda n: float(n) / sum(array), array)))


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


# The routine has a generator
# a list of frames (parents)

class SCRoutine:
    def __init__(self, rout, override_inval=None):
        if isinstance(rout, numbers.Number):
            rout = Pconst(rout)
        self.stack = []
        self.rout = rout
        self.inval = None
        self.override_inval = override_inval
        self.gen = rout.generator(self)

    def __iter__(self):
        return self

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
        if res is None:
            # faked return statement
            self.inval = next(self.gen)
            if self.stack != []:
                r, g = self.stack.pop()
                self.rout = r
                self.gen = g
                return self.next(val)
            else:
                return None
        elif isinstance(res, tuple) and (res[0] == '_EMBD_'):
            self.push()
            self.rout = res[1]
            self.gen = res[2]
            return self.next(val)
        return res


class Pattern:
    def __init__(self):
        self.nextarg = False

    def asStream(self):
        return SCRoutine(self)

    def __add__(self, operand):
        """Binary op: add two patterns"""
        # operand = copy.deepcopy(operand) if isinstance(operand, pattern) else PConst(operand)
        # return PAdd(copy.deepcopy(self), operand)
        # we actually want to retain references to our constituent patterns
        # in case the user later changes parameters of one
        # operand = Pattern.pattern(operand)
        return Padd(self, operand)

    def __radd__(self, operand):
        """Binary op: add two patterns"""
        return self.__add__(operand)

    def __mul__(self, operand):
        """Binary op: multiply two patterns"""
        return Pmul(self, operand)

    def __rmul__(self, operand):
        """Binary op: add two patterns"""
        return self.__mul__(operand)

    def __div__(self, operand):
        """Binary op: divide two patterns"""
        return Pdiv(self, operand)

    def __rdiv__(self, operand):
        """Binary op: add two patterns"""
        return self.__div__(operand)

    def __sub__(self, operand):
        """Binary op: add two patterns"""
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


def patify(val):
    if issubclass(type(val), Pattern):
        return val
    else:
        return Pconst(val)


# function to evaluate repeat and duration arguments
def patvalue(val, inval=None):
    if isinstance(val, numbers.Number):
        return val
    else:
        if len(inspect.getargspec(lambda: 3).args):
            return val(inval)
        else:
            return val()


def embedInStream(rout, obj, val=None):
    if isinstance(obj, Pattern):
        return ('_EMBD_', obj, obj.generator(rout))
    else:
        return obj


class Pseq(Pattern):
    def __init__(self, lst, repeats=1, offset=0):
        Pattern.__init__(self)
        self.lst = lst
        self.offset = offset
        self.repeats = repeats

    # this is equivalent to the embedInStream function
    def generator(self, rout):
        counter = 0
        reps = patvalue(self.repeats)
        lsize = len(self.lst)
        offsetValue = patvalue(self.offset)
        while counter < reps:
            for i in range(lsize):
                item = self.lst[(i + offsetValue) % lsize]
                yield embedInStream(rout, item)
            counter = counter + 1
        yield None
        yield rout.inval


class Ptuple(Pattern):
    def __init__(self, lst, repeats=1):
        Pattern.__init__(self)
        self.lst = lst
        self.repeats = repeats

    def generator(self, rout):
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
                    if outval is None:
                        saw_nil = True
                        break
                    tup.append(outval)
                if not saw_nil:
                    yield tup
            counter = counter + 1
        yield None
        yield rout.inval


class Pn(Pattern):
    def __init__(self, val, repeats=1):
        Pattern.__init__(self)
        self.val = val
        self.repeats = repeats

    # this is equivalent to the embedInStream function
    def generator(self, rout):
        reps = patvalue(self.repeats)
        for _ in range(reps):
            yield embedInStream(rout, self.val)
        yield None
        yield rout.inval


class Pser(Pattern):
    def __init__(self, lst, repeats=1, offset=0):
        Pattern.__init__(self)
        self.lst = lst
        self.offset = offset
        self.repeats = repeats

    # this is equivalent to the embedInStream function
    def generator(self, rout):
        counter = 0
        reps = patvalue(self.repeats)
        lsize = len(self.lst)
        offsetValue = patvalue(self.offset)
        while counter < reps:
            item = self.lst[(counter + offsetValue) % lsize]
            yield embedInStream(rout, item)
            counter = counter + 1
        yield None
        yield rout.inval

        
class Prand(Pattern):
    def __init__(self, lst, repeats=1):
        Pattern.__init__(self)
        self.lst = lst
        self.repeats = repeats

    # this is equivalent to the embedInStream function
    def generator(self, rout):
        counter = 0
        inval = rout.inval
        reps = patvalue(self.repeats, inval)
        # lsize = len(self.lst)
        while counter < reps:
            item = random.choice(self.lst)
            yield embedInStream(rout, item)
            counter = counter + 1
            inval = rout.inval
        yield None
        yield rout.inval


class Pwrand(Pattern):
    def __init__(self, lst, weights, repeats=1):
        Pattern.__init__(self)
        self.lst = lst
        self.weights = weights
        self.repeats = repeats

    # this is equivalent to the embedInStream function
    def generator(self, rout):
        counter = 0
        inval = rout.inval
        reps = patvalue(self.repeats, inval)
        while counter < reps:
            item = wchoice(self.lst, self.weights)
            yield embedInStream(rout, item)
            counter = counter + 1
            inval = rout.inval
        yield None
        yield inval

        
class Pconst(Pattern):
    def __init__(self, val):
        Pattern.__init__(self)
        self.val = val

    # this is equivalent to the embedInStream function
    def generator(self, rout):
        while True:
            yield self.val


# adverbs are not (yet) implemented
class Pbinop(Pattern):
    def __init__(self, a, b):
        Pattern.__init__(self)
        self.op1 = a
        self.op2 = b


class Padd(Pbinop):
    def __init__(self, a, b):
        Pbinop.__init__(self, a, b)
        pass

    def generator(self, rout):
        op1str = patify(self.op1).asStream()
        op2str = patify(self.op2).asStream()
        while True:
            inval = rout.inval
            val1 = op1str.next(inval)
            val2 = op2str.next(inval)
            if (val1 == None) or (val2 == None):
                yield None
                yield inval
                return
            yield val1 + val2


class Pmul(Pbinop):
    def __init__(self, a, b):
        Pbinop.__init__(self, a, b)
        pass

    def generator(self, rout):
        op1str = patify(self.op1).asStream()
        op2str = patify(self.op2).asStream()
        while True:
            inval = rout.inval
            val1 = op1str.next(inval)
            val2 = op2str.next(inval)
            if (val1 == None) or (val2 == None):
                yield None
                yield inval
                return
            yield val1 * val2

            
class Pdiv(Pbinop):
    def __init__(self, a, b):
        Pbinop.__init__(self, a, b)
        pass

    def generator(self, rout):
        op1str = patify(self.op1).asStream()
        op2str = patify(self.op2).asStream()
        while True:
            inval = rout.inval
            val1 = op1str.next(inval)
            val2 = op2str.next(inval)
            if (val1 == None) or (val2 == None):
                yield None
                yield inval
                return
            yield val1 / val2


class Psub(Pbinop):
    def __init__(self, a, b):
        Pbinop.__init__(self, a, b)
        pass

    def generator(self, rout):
        op1str = patify(self.op1).asStream()
        op2str = patify(self.op2).asStream()
        while True:
            inval = rout.inval
            val1 = op1str.next(inval)
            val2 = op2str.next(inval)
            if (val1 == None) or (val2 == None):
                yield None
                yield inval
                return
            yield val1 - val2


class Ppow(Pbinop):
    def __init__(self, a, b):
        Pbinop.__init__(self, a, b)
        pass
        
    def generator(self, rout):
        op1str = patify(self.op1).asStream()
        op2str = patify(self.op2).asStream()
        while True:
            inval = rout.inval
            val1 = op1str.next(inval)
            val2 = op2str.next(inval)
            if (val1 == None) or (val2 == None):
                yield None
                yield inval
                return
            yield val1 ** val2


class Pmod(Pbinop):
    def __init__(self, a, b):
        Pbinop.__init__(self, a, b)
        pass

    def generator(self, rout):
        op1str = patify(self.op1).asStream()
        op2str = patify(self.op2).asStream()
        while True:
            inval = rout.inval
            val1 = op1str.next(inval)
            val2 = op2str.next(inval)
            if (val1 == None) or (val2 == None):
                yield None
                yield inval
                return
            yield val1 % val2


class FilterPattern(Pattern):
    def __init__(self, pattern):
        Pattern.__init__(self)
        self.pattern = pattern


class Pfin(FilterPattern):
    def __init__(self, pattern, n):
        FilterPattern.__init__(self, pattern)
        self.n = n

    def generator(self, rout):
        stream = patify(self.pattern).asStream()
        inval = rout.inval
        n = patvalue(self.n, inval)
        counter = 0
        while counter < n:
            val = stream.next(inval)
            if (val == None):
                yield None
                yield inval
                return
            yield val
            counter = counter + 1


class Pstutter(FilterPattern):
    def __init__(self, n, pattern):
        FilterPattern.__init__(self, pattern)
        self.n = n

    def generator(self, rout):
        stream = patify(self.pattern).asStream()
        nstr = patify(self.n).asStream()
        while True:
            inval = rout.inval
            num = nstr.next(inval)
            val = stream.next(inval)
            if (num == None) or (val == None):
                yield None
                yield inval
                return
            for k in range(num):
                yield val


class Pcollect(FilterPattern):
    def __init__(self, func, pattern):
        FilterPattern.__init__(self, pattern)
        self.func = func

    def generator(self, rout):
        stream = patify(self.pattern).asStream()
        while True:
            inval = rout.inval
            val = stream.next(inval)
            if val == None:
                yield None
                yield inval
                return
            yield self.func(val)


class Pwhite(Pattern):
    def __init__(self, lo=0.0, hi=1.0, repeats=inf):
        Pattern.__init__(self)
        self.lo = lo
        self.hi = hi
        self.repeats = repeats

    # this is equivalent to the embedInStream function
    def generator(self, rout):
        counter = 0
        reps = patvalue(self.repeats)
        lostr = patify(self.lo).asStream()
        histr = patify(self.hi).asStream()
        while counter < reps:
            inval = rout.inval
            loval = lostr.next(inval)
            hival = histr.next(inval)
            if (loval == None) or (hival == None):
                yield None
                yield inval
            if (type(loval) == int) and (type(hival) == int):
                yield random.randint(loval, hival)
            else:
                yield random.uniform(loval, hival)
            counter = counter + 1
        yield None
        yield rout.inval


class Pbrown(Pattern):
    def __init__(self, lo=0.0, hi=1.0, step=0.125, repeats=inf):
        Pattern.__init__(self)
        self.lo = lo
        self.hi = hi
        self.step = step
        self.repeats = repeats

    # this is equivalent to the embedInStream function
    def generator(self, rout):
        counter = 0
        reps = patvalue(self.repeats)
        lostr = patify(self.lo).asStream()
        histr = patify(self.hi).asStream()
        stepstr = patify(self.step).asStream()
        inval = rout.inval
        loval = lostr.next(inval)
        hival = histr.next(inval)
        stepval = stepstr.next(inval)
        if (loval is None) or (hival is None) or (stepval is None):
            yield None
            yield inval
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
            if (loval is None) or (hival is None) or (stepval is None):
                yield None
                yield inval
            if (type(loval) == int) and (type(hival) == int) and (type(stepval) == int):
                curval = fold(curval + random.randint(-stepval, stepval), loval, hival)
                yield curval
            else:
                curval = fold(curval + random.uniform(-stepval, stepval), loval, hival)
                yield curval
            counter = counter + 1
        yield None
        yield rout.inval


class Pshuf(Pattern):
    def __init__(self, lst, repeats=1):
        Pattern.__init__(self)
        self.lst = lst
        self.repeats = repeats

    def generator(self, rout):
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
        yield None
        yield rout.inval


class Pindex(Pattern):
    def __init__(self, lst, index):
        Pattern.__init__(self)
        self.lst = lst
        self.index = index

    def generator(self, rout):
        lsize = len(self.lst)
        inval = rout.inval
        idxstr = patify(self.index).asStream()
        while True:
            idx = idxstr.next(inval)
            if idx == None:
                yield None
                yield inval
            yield embedInStream(rout, self.lst[idx % lsize])

            
class Pseries(Pattern):
    def __init__(self, start, step, length):
        Pattern.__init__(self)
        self.start = start
        self.step = step
        self.length = length

    def generator(self, rout):
        inval = rout.inval
        cur = patvalue(self.start, inval)
        stepstr = patify(self.step).asStream()
        leng = patvalue(self.length, inval)
        counter = 0
        while counter < leng:
            yield cur
            inval = rout.inval
            stp = stepstr.next(inval)
            if stp == None:
                yield None
                yield inval
                return
            cur = cur + stp
            counter = counter + 1
        yield None
        yield inval


# base class for event - subtype for special purpose
class Event:
    default_parent = {}

    def __init__(self, dict, parent=None):
        if parent is None:
            self.parent = self.default_parent
        else:
            self.parent = parent
        self.map = dict

    def __repr__(self):
        return str(self.map)

    def __missing__(self, key):
        raise KeyError(key)

    def __getitem__(self, key):
        try:
            return self.map[key]
        except:
            return self.parent.get(key)

    def __setitem__(self, key, value):
        self.map[key] = value

    def value(self, key):
        try:
            res = self.map[key]
        except:
            res = self.parent[key]
        if callable(res):
            return res(self)
        else:
            return res

    def copy(self):
        cp = copy.copy(self)
        cp.map = copy.copy(self.map)
        return cp


class Pbind(Pattern):
    def __init__(self, *args):
        Pattern.__init__(self)
        self.patternpairs = args

    def asStream(self, override_inval=None):
        return SCRoutine(self, override_inval)

    def generator(self, rout):
        streampairs = list(self.patternpairs)
        endval = len(streampairs)

        for i in range(1, endval, 2):
            streampairs[i] = patify(streampairs[i]).asStream()

        while True:
            inval = rout.inval
            if inval is None:
                yield None
                return
            event = inval.copy()
            for i in range(0, endval, 2):
                name = streampairs[i]
                stream = streampairs[i + 1]
                streamout = stream.next(event)
                if streamout is None:
                    yield None
                    yield None
                    return
                # support tupled
                if isinstance(name, (list, tuple)):
                    if (name.size > streamout.size):
                        print("the pattern is not providing enough values to assign to the key set:" + name)
                        yield None
                        return
                    # here we would iterate over the name
                    print("list type keys are not supported (yet)")
                else:
                    event[name] = streamout
            yield event




# give it a unique switch that tries to keep the group values
# unique - this might not be possible if the element stream
# does not provide enough different values
# class Prepscheme2(Pattern):
# 	var <>groupsize, <>groupscheme, <>reset;
# 	*new { arg groupsize, groupscheme, pattern, reset=false;
# 		^super.new(pattern).groupsize_(groupsize).groupscheme_(groupscheme).reset_(reset);
# 	}
# 	storeArgs { ^[pattern,groupsize, groupscheme, reset] }

#         def generator(self, rout):
# 	    stream = patify(pattern).asStream()
# 	    gsizestream = patify(groupsize).asStream()
# 	    gschemestream = patify(groupscheme).asStream()
# 	    resetstream = patify(reset).asStream()
# 	    groups = [None] * 16            

#             while True:
#                 gscheme = gschemestream.next(event)
#                 resetval = gschemestream.next(event)
# 		if resetval:
# 		    groups = [None] * 16
# 		while isinstance(gscheme, basestring) and (isinstane(gscheme[0], string)):
# 		    sym = gscheme.at(0);
 
#                     if sym == 'take':
#                         for k in range(1, len(gscheme)):
#                             groups[gscheme[k]] = None
#                     elif sym == 'reset':
#                         groups = [None] * 16
#                     elif sym == 'set':
#                         groups[gscheme[1]] = groups[gscheme[2]]
#                     elif sym == 'swap':
# 			tmp = groups[gscheme[1]]; 
# 			groups[gscheme[1]] = groups[gscheme[2]];
# 			groups[gscheme[2]] = tmp;
#                     elif sym == 'sort':
#                         tmp = [x for x in groups if x is not None]
# 			if gscheme[1] > 0:
# 			    groups = sorted(tmp)
#                         else:
# 			    groups = tmp.sort({ arg a,b;  a.wrapAt(0) > b.wrapAt(0) })
					
# 			groups.postln;
# 			groups = groups.extend(16)
#                     elif sym == 'sortfunc':
# 			tmp = groups.reject(_.isNil);
# 			//tmp.postln;
# 			groups = tmp.sort(gscheme[1]);
# 			groups = groups.extend(16)

# 		    elif sym == 'scramble': 
# 			tmp = groups.reject(_.isNil);
# 			groups = tmp.scramble ++ Array.newClear(16-tmp.size);					
				
# 		    elif sym == 'permute' 
# 			tmp = groups.reject(_.isNil);
# 			groups = tmp.permute(gscheme[1]) ++ Array.newClear(16-tmp.size);
				
# 			gscheme = gschemestream.next(event);
			
# 			group = groups.at(gscheme);
# 			gsize = gsizestream.next(event);
# 			if (gsize.isNil) { ^event };
# 			gsize.do({ arg k;
# 				if (k >= group.size) {
# 					val = stream.next(event);
# 					if (val.isNil) { ^event };
# 					group = group.add(val);
# 					groups.put(gscheme, group);
# 					event = val.yield;
# 				} {
# 					event = groups.at(gscheme).at(k).yield;
# 				};
# 			});
# 		};		
# 	}
# }

# pb = Pbind('type', 'model',   'model', Prand([0,1,2], inf), 'dur', Pseq([0.5, 1.0], inf))
# pst = pb.asStream()
# pst.next(Event({}))

#pb = Pbind('type', 'model',   'model', Prand([0,1,2], inf), 'dur', Pseq([0.5, 1.0], inf))
#pst = pb.asStream({})
#pst.next(Event({}))

