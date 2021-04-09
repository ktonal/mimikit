# This is a partial port of SuperCollider's Pattern-Stream-Event system.

import sys
import copy
import random
import inspect
import itertools
import numbers
import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import Any

from .event import Event
from .pbind import EOP, inf, EMBED, Pattern, SCRoutine, patvalue, embedInStream, patify
from .pbind import wchoice, fold, wrap, clip, choose, windex


class Pseq(Pattern):
    def __init__(self, lst, repeats=1, offset=0):
        Pattern.__init__(self)
        self.lst = lst
        self.offset = offset
        self.repeats = repeats

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
    def __init__(self, key, repeats=inf, default=None):
        Pattern.__init__(self)
        self.key = key
        self.default = default
        self.repeats = repeats

    def embedInStream(self, rout):
        reps = patvalue(self.repeats)
        counter = 0
        while counter < reps:
            yield rout.inval.get(self.key, self.default)
            counter += 1
        yield EOP


class Pinsteps(Pattern):
    def __init__(self, steps, pattern):
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


class FilterPattern(Pattern):
    def __init__(self, pattern):
        Pattern.__init__(self)
        self.pattern = pattern


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


class Preject(FilterPattern):
    def __init__(self, func, pattern):
        FilterPattern.__init__(self, pattern)
        self.func = func

    def embedInStream(self, rout):
        stream = patify(self.pattern).asStream()

        while True:
            val = stream.next(rout.inval)
            if val is EOP:
                yield EOP
            out = self.func(val)
            if not out:
                yield val


class Pselect(FilterPattern):
    def __init__(self, func, pattern):
        FilterPattern.__init__(self, pattern)
        self.func = func

    def embedInStream(self, rout):
        stream = patify(self.pattern).asStream()

        while True:
            val = stream.next(rout.inval)
            if val is EOP:
                yield EOP
            out = self.func(val)
            if out:
                yield val


class Pwhite(Pattern):
    def __init__(self, lo=0.0, hi=1.0, repeats=inf):
        Pattern.__init__(self)
        self.lo = lo
        self.hi = hi
        self.repeats = repeats

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


class Pgauss(Pattern):
    def __init__(self, mu=0.0, sigma=0.5, repeats=inf):
        Pattern.__init__(self)
        self.mu = mu
        self.sigma = sigma
        self.repeats = repeats

    def embedInStream(self, rout):
        counter = 0
        reps = patvalue(self.repeats)
        mu_str = patify(self.mu).asStream()
        sigma_str = patify(self.sigma).asStream()
        while counter < reps:
            inval = rout.inval
            mu = mu_str.next(inval)
            sigma = sigma_str.next(inval)
            if (mu is EOP) or (sigma is EOP):
                yield EOP
            yield np.random.normal(mu, sigma)
            counter = counter + 1
        yield EOP


class Pgamma(Pattern):
    def __init__(self, shape=0.1, scale=1.0, repeats=inf):
        Pattern.__init__(self)
        self.shape = shape
        self.scale = scale
        self.repeats = repeats

    def embedInStream(self, rout):
        counter = 0
        reps = patvalue(self.repeats)
        shape_str = patify(self.shape).asStream()
        scale_str = patify(self.scale).asStream()
        while counter < reps:
            inval = rout.inval
            shape = shape_str.next(inval)
            scale = scale_str.next(inval)
            if (scale is EOP) or (shape is EOP):
                yield EOP
            yield np.random.gamma(shape, scale)
            counter = counter + 1
        yield EOP


class Ppoisson(Pattern):
    def __init__(self, lam=1.0, repeats=inf):
        Pattern.__init__(self)
        self.lam = lam
        self.repeats = repeats

    def embedInStream(self, rout):
        counter = 0
        reps = patvalue(self.repeats)
        lam_str = patify(self.lam).asStream()

        while counter < reps:
            inval = rout.inval
            lam = lam_str.next(inval)
            if lam is EOP:
                yield EOP
            yield np.random.poisson(lam)
            counter = counter + 1
        yield EOP


class Pcauchy(Pattern):
    def __init__(self, mean=0.0, spread=1.0, repeats=inf):
        self.mean = mean
        self.spread = spread
        self.repeats = repeats

    def embedInStream(self, rout):
        meanStr = patify(self.mean).asStream()
        spreadStr = patify(self.spread).asStream()
        reps = patvalue(self.repeats)
        count = 0

        while count < reps:
            inval = rout.inval
            ran = 0.5
            meanVal = meanStr.next(inval)
            spreadVal = spreadStr.next(inval)
            if meanVal is EOP or spreadVal is EOP:
                yield EOP
            while ran == 0.5:
                ran = random.uniform(0.0, 1.0)
            yield (spreadVal * np.tan(ran * np.pi)) + meanVal
            count += 1
        yield EOP


class Pbrown(Pattern):
    def __init__(self, lo=0.0, hi=1.0, step=0.125, repeats=inf):
        Pattern.__init__(self)
        self.lo = lo
        self.hi = hi
        self.step = step
        self.repeats = repeats

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
        counter = 0

        while counter < leng:
            yield cur
            inval = rout.inval
            stp = stepstr.next(inval)
            if stp is EOP:
                yield EOP
            cur = cur + stp
            counter += 1
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
                    yield groups[gscheme][k]
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
                yield item
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

    def embedInStream(self, rout):
        streams = list(reversed([patify(pattern).asStream() for pattern in self.patterns]))
        while True:
            inevent = copy.copy(rout.inval)
            for stream in streams:
                inevent = stream.next(inevent)
                if inevent is EOP:
                    yield EOP
            yield(inevent)


class Pbeta(Pattern):

    def __init__(self, lo=0.0, hi=1.0, prob1=1, prob2=1, repeats=inf):
        self.lo = lo
        self.hi = hi
        self.prob1 = prob1
        self.prob2 = prob2
        self.repeats = repeats

    def embedInStream(self, rout):
        loStr = patify(self.lo).asStream()
        hiStr = patify(self.hi).asStream()
        prob1Str = patify(self.prob1).asStream()
        prob2Str = patify(self.prob2).asStream()
        counter = 0
        reps = patvalue(self.repeats)

        while counter < reps:
            sum = 2
            inval = rout.inval
            rprob1 = prob1Str.next(inval)
            rprob2 = prob2Str.next(inval)
            if rprob1 is EOP or rprob2 is EOP:
                yield EOP
            rprob1 = 1.0 / rprob1
            rprob2 = 1.0 / rprob2
            loVal = loStr.next(inval)
            hiVal = hiStr.next(inval)
            if loVal is EOP or hiVal is EOP:
                yield EOP

            while sum > 1:
                temp = random.uniform(0.0, 1.0) ** rprob1
                sum = temp + (random.uniform(0.0, 1.0) ** rprob2)
            yield (temp / sum) * (hiVal - loVal) + loVal
            counter += 1
        yield EOP


class Prandlace(Pattern):

    def __init__(self, lst, repeats=1):
        self.lst = lst
        self.repeats = repeats

    def embedInStream(self, rout):
        streams = list([patify(pat).asStream() for pat in self.lst])
        reps = patvalue(self.repeats)
        maxidx = len(self.lst) - 1
        counter = 0

        while counter < reps:
            val = rout.inval
            idx = random.randint(0, maxidx)
            item = streams[idx]
            inval = item.next(val)
            if inval is EOP:
                streams[idx] = patify(self.lst[idx].asStream())
                item = streams[idx]
                inval = item.next(val)
            yield inval
            counter += 1
        yield EOP


class Pindexlace(Pattern):

    def __init__(self, lst, idx, repeats=1):
        self.lst = lst
        self.idx = idx
        self.repeats = repeats

    def embedInStream(self, rout):
        streams = list([patify(pat).asStream() for pat in self.lst])
        reps = patvalue(self.repeats)
        idx_str = patify(self.idx).asStream()
        length = len(self.lst)
        counter = 0

        while counter < reps:
            while True:
                val = rout.inval
                idx = idx_str.next(val)
                if idx is EOP:
                    yield EOP
                item = streams[idx % length]
                inval = item.next(val)
                if inval is EOP:
                    streams[idx] = patify(self.lst[idx].asStream())
                    item = streams[idx]
                    inval = item.next(val)
                yield inval
            counter += 1
        yield EOP


class Pwrandlace(Pattern):

    def __init__(self, lst, weights, repeats=1):
        self.lst = lst
        self.repeats = repeats
        self.weights = weights

    def embedInStream(self, rout):
        streams = list([patify(pat).asStream() for pat in self.lst])
        wStr = patify(self.weights).asStream()
        reps = patvalue(self.repeats)
        counter = 0

        while counter < reps:
            val = rout.inval
            wval = wStr.next(val)
            if wval is EOP:
                yield EOP
            idx = windex(wval)
            item = streams[idx]
            inval = item.next(val)
            if inval is EOP:
                streams[idx] = patify(self.lst[idx].asStream())
                item = streams[idx]
                inval = item.next(val)
            yield inval
            counter += 1
        yield EOP


@dataclass(order=True)
class PrioritizedEvent:
    priority: int
    item: Any = field(compare=False)


class Ppar(Pattern):

    def __init__(self, lst, repeats=1):
        self.lst = lst
        self.repeats = repeats

    def initStreams(self, priorityQ):
        for pattern in self.lst:
            priorityQ.append(PrioritizedEvent(0.0, patify(pattern).asStream()))
        heapq.heapify(priorityQ)

    def embedInStream(self, rout):
        priorityQ = []
        counter = 0
        reps = patvalue(self.repeats)

        while counter < reps:
            now = 0.0

            self.initStreams(priorityQ)
            inval = rout.inval
            # if first event not at time zero
            if len(priorityQ) > 0:
                nexttime = priorityQ[0].priority
                if nexttime > 0.0:
                    outval = Event.silent(nexttime, inval)
                    yield outval
                    now = nexttime

            while len(priorityQ) > 0:
                inval = rout.inval
                stream = heapq.heappop(priorityQ).item
                outval = stream.next(inval)
                if outval is EOP:
                    if len(priorityQ) < 1:
                        priorityQ = []
                    else:
                        nexttime = priorityQ[0].priority
                        # that child stream ended, so rest until next one
                        outval = Event.silent(nexttime - now, inval)
                        yield outval
                        now = nexttime
                else:
                    # requeue stream
                    heapq.heappush(priorityQ, PrioritizedEvent(now + outval.delta, stream))
                    nexttime = priorityQ[0].priority                    
                    outval['delta'] = nexttime - now
                    yield outval
                    # inval ?? { this.purgeQueue(priorityQ); ^nil.yield };
                    now = nexttime
            counter += 1
        yield EOP


class Ptime(Pattern):

    def __init__(self, repeats=inf):
        self.repeats = repeats

    def embedInStream(self, rout):
        start = rout.time
        reps = patvalue(self.repeats, rout)
        counter = 0
        while counter < reps:
            yield rout.time - start
            counter += 1
        yield EOP
