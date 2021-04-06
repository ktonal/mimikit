import copy


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


# base class for event - subtype for special purpose
class Event:
    default_parent = {}

    def __init__(self, dict=None, parent=None):
        self.parent = parent or self.default_parent
        self.map = dict or {}

    def __repr__(self):
        return str(self.map)

    def __missing__(self, key):
        raise KeyError(key)

    def __getitem__(self, key):
        res = self.map.get(key)
        if res is None:
            return self.parent.get(key)
        else:
            return res

    def get(self, key, default=None):
        res = self.map.get(key)
        if res is None:
            return self.parent.get(key, default)
        else:
            return res

    def __setitem__(self, key, value):
        self.map[key] = value

    def value(self, key):
        res = self.map.get(key)
        if res is None:
            res = self.parent.get(key)
        if callable(res):
            return res(self)
        else:
            return res

    def keys(self):
        return self.map.keys()

    def update(self, event):
        self.map.update(event)

    @property
    def delta(self):
        delta = self.get('delta')
        return delta or self.get('dur')

    def copy(self):
        cp = copy.copy(self)
        cp.map = copy.copy(self.map)
        return cp

    @classproperty
    def default(cls):
        return Event()

    @classmethod
    def silent(cls, dur, inval=None):
        vals = {'dur': dur, 'delta': dur, 'isRest': True}
        if inval:
            inval.update(vals)
            return Event(inval)
        return Event(vals)
