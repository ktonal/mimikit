import copy


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


# base class for event - subtype for special purpose
class Event(dict):
    default_parent = {}

    def __init__(self, dictionary=None, parent=None):
        dict.__init__(self, **(dictionary or {}))
        self.parent = parent or self.default_parent

    def __missing__(self, key):
        raise KeyError(key)

    def __getitem__(self, key):
        res = dict.get(self, key)
        return res if res is not None else dict.get(self.parent, key)

    def get(self, key, default=None):
        res = dict.get(self, key)
        return res if res is not None else dict.get(self.parent, key)

    def value(self, key):
        res = self.get(key)
        if callable(res):
            return res(self)
        else:
            return res

    @property
    def delta(self):
        return self.get('delta') or self.get('dur')

    def copy(self):
        return copy.copy(self)

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

