import pytest
from mimikit.pbind.patterns import *
from mimikit.pbind import Pbind, Pmin, Pmax
import numpy as np


def test_simple():
    assert [0, 1, 2, 3, 0, 1, 2, 3] == list([x for x in Pseq([0, 1, 2, 3], 2)])
    assert [0, 1, 2, 3, 0, 1] == list([x for x in Pser([0, 1, 2, 3], 6)])
    assert len(list([x for x in Prand([0, 1, 2, 3], 6)])) == 6
    assert len(list([x for x in Prand([0, 1, 2, 3], 6)])) == 6
    assert len(list([x for x in Pxrand([0, 1, 2, 3], 6)])) == 6
    assert len(list([x for x in Pwrand([0, 1, 2], [0.8, 0.1, 0.1], 6)])) == 6
    assert len(list([x for x in Pn(Prand([0, 1], 2), 2)])) == 4
    assert [0,0, 1, 1, 2, 2] == list([x for x in Pstutter(2, Pseq([0,1,2]))])
    assert [0.0, 1.0] == list([x for x in Pcollect(np.sin, Pseq([0, np.pi/2]))])
    assert [0, 3] == list([x for x in Preject(lambda x: x % 3 != 0, Pseq([0, 1, 2, 3, 4, 5]))])
    assert [1, 2, 4, 5] == list([x for x in Pselect(lambda x: x % 3 != 0, Pseq([0, 1, 2, 3, 4, 5]))])
    assert [[0, 1], [4, 1], [6, 3]] == list([x for x in Ptuple([Pseq([0, 4, 6]), Pseq([1, 1, 3])])])
    assert len(list([x for x in Pwhite(0.0, 1.0, 5)])) == 5
    assert len(list([x for x in Pgauss(0.0, 0.5, 5)])) == 5
    assert len(list([x for x in Pgamma(0.1, 0.5, 5)])) == 5
    assert len(list([x for x in Ppoisson(1.0, 5)])) == 5
    assert len(list([x for x in Pcauchy(1.0, 2.0,  5)])) == 5
    assert len(list([x for x in Pbrown(-1.0, 1.0, 0.25, 5)])) == 5
    assert len(list([x for x in Pshuf([0,1,2,3,4,5], 2)])) == 12
    assert [30, 30, 20, 10, 10] == list([x for x in Pindex([10, 20, 30], Pseq([2, 2, 1, 3, 0]))])
    assert [0, 2, 4, 6] == list([x for x in Pseries(0, 2, 4)])
    assert [1, 2, 4, 8] == list([x for x in Pgeom(1, 2, 4)])
    assert len(list([x for x in Pbeta(0.0, 1.0, 1.0, 1.0, 4)])) == 4
    assert [3, 0, 3, 1, 3, 2, 4, 3] == list([x for x in Ppatlace([Pseq([3, 3, 3, 4]), Pseries(0, 1, 4)], 4)])


def test_nest_and_arithmetic():
    assert [5, 10, 6, 7, 8, 10, 10, 3] == [x for x in Pindex([Pseq([1, 2]) * 5, 5 + Pseq([1, 2]), 8, 1 * Pseq([Pn(3, 2) + 5, 1]) + 2], Pseq([0, 1, 2, 3]))]


def test_pbind():
    res = [{'freq': 100.0, 'param': 10},
           {'freq': 210, 'param': 10},
           {'freq': 120.0, 'param': 10},
           {'freq': 230, 'param': 10}]
    assert res == [x for x in Pbind('freq', Pseq([100.0, 200], 2) + Pseries(0, 10, 4), 'param', 10)]

    # linked parameter groups
    res = [x for x in Pbind(['freq', 'dur'], Pseq([Ptuple((Prand([2, 10]), Prand([200, 1]))),
                                                   Pseq([[40, 40], [100, 3]])], 2),
                            'param', 10)]
    assert len(res) == 6 and res[1]['freq'] == 40 and res[1]['dur'] == 40


def test_key_access():
    res = [x for x in Pbind('dur', Prand([1.0, 0.5, 0.25], 8), 'sustain', Pkey('dur'))]
    assert all(x['dur'] == x['sustain'] for x in res)
    res = [x for x in Pbind('dur', Pseq([1.0, 1.0, 0.25]),
                            'sustain', Pseq([0.5, Pkey('dur', 1)], inf))]
    assert [0.5, 1.0, 0.5] == list(map(lambda x: x['sustain'], res))
    res = [x for x in Pbind(['dur', 'p2'], Prand([[1.0, 0.0], [0.5, 2], [0.25, 0.5]], 8),
                            'param', Pfunc(lambda x: np.cos(x['dur'])))]
    assert all(x['param'] == np.cos(x['dur']) for x in res)


def test_pchain():
    correct = [{'lower': 0, 'something': 0, 'other': 5},
               {'lower': 0.25, 'something': -1, 'other': 5},
               {'lower': 0.5, 'something': 1, 'other': 5},
               {'lower': 0.75, 'something': 0, 'other': 5},
               {'lower': 1.0, 'something': -1, 'other': 5},
               {'lower': 1.25, 'something': 1, 'other': 5},
               {'lower': 1.5, 'something': 0, 'other': 5},
               {'lower': 1.75, 'something': -1, 'other': 5},
               {'lower': 2.0, 'something': 1, 'other': 5}]
    res = [x for x in Pchain(Pbind('something', Pseq([0, -1, 1], 3), 'other', 5),
                             Pbind('lower', Pseries(0, 0.25, 12)))]
    assert  correct == res
    correct = [{'lower': 0, 'something': 0, 'other': 5},
               {'lower': 0.25, 'something': -1, 'other': 5}]
    res = [x for x in Pchain(Pbind('something', Pseq([0, -1, 1], 3), 'other', 5),
                             Pbind('lower', Pseries(0, 0.25, 2)))]
    assert  correct == res
    # key access from a chained pattern
    correct = [{'lower': 0, 'something': 0, 'other': 0},
               {'lower': 0.25, 'something': -1, 'other': 0.25}]
    res = [x for x in Pchain(Pbind('something', Pseq([0, -1, 1], 3), 'other', Pkey('lower')),
                             Pbind('lower', Pseries(0, 0.25, 2)))]
    assert  correct == res


def test_time():
    correct = [{'dur': 0.5, 'time': 0.0},
               {'dur': 0.5, 'time': 0.5},
               {'dur': 0.25, 'time': 1.0},
               {'dur': 0.25, 'time': 1.25},
               {'dur': 0.5, 'time': 1.5},
               {'dur': 0.5, 'time': 2.0},
               {'dur': 0.25, 'time': 2.5},
               {'dur': 0.25, 'time': 2.75},
               {'dur': 0.5, 'time': 3.0},
               {'dur': 0.5, 'time': 3.5},
               {'dur': 0.25, 'time': 4.0},
               {'dur': 0.25, 'time': 4.25}]
    res = [x for x in Pbind('dur', Pseq([0.5, 0.5, 0.25, 0.25], 3), 'time', Ptime())]
    assert res == correct
    correct = [{'dur': 0.5, 'time': 0.0},
               {'dur': 0.5, 'time': 0.5},
               {'dur': 0.25, 'time': 1.0},
               {'dur': 0.25, 'time': 1.25},
               {'dur': 0.5, 'time': 0.0},
               {'dur': 0.5, 'time': 0.5},
               {'dur': 0.25, 'time': 1.0},
               {'dur': 0.25, 'time': 1.25}]
    res = [x for x in Pseq([Pbind('dur', Pseq([0.5, 0.5, 0.25, 0.25]), 'time', Ptime()),
                            Pbind('dur', Pseq([0.5, 0.5, 0.25, 0.25]), 'time', Ptime())])]
    assert res == correct
    correct = [{'dur': 0.5, 'time': 0.0},
               {'dur': 0.5, 'time': 0.5},
               {'dur': 0.25, 'time': 1.0},
               {'dur': 0.25, 'time': 1.25},
               {'dur': 0.5, 'time': 1.5},
               {'dur': 0.5, 'time': 2.0},
               {'dur': 0.25, 'time': 2.5},
               {'dur': 0.25, 'time': 2.75}]
    res = [x for x in Pchain(Pbind('time', Ptime()),
                             Pseq([Pbind('dur', Pseq([0.5, 0.5, 0.25, 0.25])),
                                   Pbind('dur', Pseq([0.5, 0.5, 0.25, 0.25]))]))]
    assert res == correct


def test_fin():
    res = [x for x in Pbind('dur', Prand([0.5, 1.0], 8)).finDur(2.0)]
    assert sum(map(lambda x: x.delta, res)) == 2.0
    res = [x for x in Pbind('dur', Pseq([0.5, 1.0], 8)).fin(2)]
    assert len(res) == 2


# TODO:
# to do Pset, Pmul, Padd, Psub, Pdiv, loop, sin, cos, tan, tanh, exp, log, log10
# Ppatlace
# Prandlace
# Pindexlace
# Pwrandlace
# Ppar
# Pmsm
# Prepscheme
# integrate
# differentiate
# Pmin
# Pmax
# clip
# fold
# wrap
# Plazy


