import operator
import torch.nn as nn
from itertools import chain
from functools import reduce
from types import MethodType
import re
from inspect import signature, Parameter

__all__ = [
    'HOC',
    "HOM",
    "hom",
    "flatten",
    "Sum",
    "Maybe",
    "Switch",
    "PPrintCtx",
    "combine",
    "Map",
    "Reduce",
    "get_input_signature"
]


def is_step(obj):
    if isinstance(obj, tuple):
        if obj == tuple() or (len(obj) == 2 and callable(obj[0]) and isinstance(obj[1], str)):
            return True
    return False


def sig2tuple(sig):
    if sig is not None:
        i, o = sig.split("->")
        i, o = i.strip(" "), o.strip(" ")
        return tuple(x.split('=')[0] if '=' in x else x.strip(',').strip("*")
                     for x in i.split(", ")), tuple(o.split(", "))
    else:
        return ("_",), ("_",)


def tuple2sig(i, o):
    i, o = ", ".join(i), ", ".join(o)
    return f"{i} -> {o}"


is_optional = re.compile(r"(\*\*)|[=]")


class Signature(str):
    def __init__(self, signature_str):
        super(Signature, self).__init__()
        self.in_, self.out_ = sig2tuple(signature_str)
        self.in_sig, self.out_sig = signature_str.split(" -> ")
        self.args = tuple(i for i, si in zip(self.in_, self.in_sig.split(", "))
                          if re.search(is_optional, si) is None)
        self.full_kwargs = tuple(si for si in self.in_sig.split(", ")
                                  if re.search(is_optional, si) is not None)
        self.kwargs_names = tuple(si.split("=")[0] if '=' in si else si
                                  for si in self.full_kwargs)
        self.default = {i: eval(si.split("=")[1]) if '=' in si else {}
                        for i, si in zip(self.in_, self.in_sig.split(", "))
                        if re.search(is_optional, si) is not None}

    def asdict(self, args_val=None):
        args = {k: args_val for k in self.in_}
        args.update(self.default)
        return args


def always_tuple(obj):
    if isinstance(obj, tuple):
        return obj
    return obj,


def eval_step(clb, sig, ctx):
    key_error = tuple(i for i in sig.in_ if i not in ctx and i != "_")
    if any(key_error):
        raise KeyError(f"Input keys '{key_error[0]}' not found in context at step ({clb}, {sig})")
    inpt = tuple(ctx[i] if i != "_" else ctx for i in sig.in_)
    try:
        rv = clb(*inpt)
    except Exception as e:
        raise RuntimeError(f"step ({clb}, {sig}) raised a {type(e).__qualname__} exception")
    if sig.out_ == ("_",):
        return rv
    if len(sig.out_) == 1 and isinstance(rv, tuple) and len(rv) > 1:
        rv = (rv,)
    # update and filter out voids
    ctx.update({o: v for v, o in zip(always_tuple(rv), sig.out_) if o != "$"})
    return ctx


def get_out_(ctx, out_):
    if out_ == ("_",):
        return ctx
    if len(out_) == 1:
        return ctx[out_[0]]
    return tuple(ctx[o] for o in out_)


class HOCBase:
    __fname__ = "__call__"

    @staticmethod
    def _wrap_clb(clb):
        # useful in the hom subclass to wrap functions in modules
        return clb

    def __init__(self, sig_str, *steps):
        super().__init__()
        self.signatures = []
        # we first store the steps...
        if any(steps):
            self.extend(steps)
        # ...then we compile
        self.s = Signature(sig_str)
        self.recompile()

    def recompile(self, sig_str=None):
        if sig_str is not None:
            self.s = Signature(sig_str)
        dct = {}
        exec(f"def {self.__fname__}(self, {self.s.in_sig}): ctx = locals(); return self.call(ctx)", dct)
        self.__dict__.update({self.__fname__: MethodType(dct[self.__fname__], self)})
        return self

    def call(self, ctx):
        for clb, sig in self:
            ctx = eval_step(clb, sig, ctx)
        return get_out_(ctx, self.s.out_)

    def __iter__(self):
        for clb, sig in zip(super().__iter__(), self.signatures):
            yield clb, sig

    def append(self, item):
        clb, sig = item
        super().append(self._wrap_clb(clb))
        self.signatures.append(Signature(sig))

    def extend(self, lst):
        clbs, sigs = zip(*lst)
        super().extend(map(self._wrap_clb, clbs))
        self.signatures.extend(map(Signature, sigs))

    def insert(self, index, step):
        clb, sig = step
        super().insert(index, self._wrap_clb(clb))
        self.signatures.insert(index, Signature(sig))

    def __getitem__(self, item):
        clb = super().__getitem__(item)
        sig = self.signatures[item]
        return clb, sig

    def __setitem__(self, item, value):
        if is_step(value):
            clb, sig = value
            super().__setitem__(item, self._wrap_clb(clb))
            self.signatures[item] = Signature(sig)
        elif isinstance(item, slice):
            clbs, sigs = zip(*value)
            items_to_set = list(range(len(self)))[item]
            for it in items_to_set:
                super().__setitem__(it, self._wrap_clb(clbs[it]))
            self.signatures[item] = list(map(Signature, sigs))
        else:
            raise ValueError(f"Cannot set item {item} to value {value}")

    def __add__(self, other):
        return self.extend(other)

    def __iadd__(self, other):
        return self.extend(other)


HOC = type("HOC", (HOCBase, list),
           {"__call__": lambda self, *args, **kwargs: self.__call__(*args, **kwargs)})


class WrapperModule(nn.Module):
    def __init__(self, clb):
        super().__init__()
        self.clb = clb

    def forward(self, *args, **kwargs):
        return self.clb(*args, **kwargs)

    def __str__(self):
        return str(self.clb)

    def __repr__(self):
        if isinstance(self.clb, MethodType) and isinstance(self.clb.__self__, nn.Module):
            # we need this to break infinite recursion
            return repr(self.clb.__func__)
        return repr(self.clb)


def _wrap_clb(clb):
    return clb if isinstance(clb, nn.Module) else WrapperModule(clb)


# The Higher-Order Module class
HOM = type(
    "HOM", (HOCBase, nn.ModuleList),
    {"__fname__": "forward",
     "_wrap_clb": staticmethod(_wrap_clb),
     # method to return a copy of the base (meta?) type.
     # handy if a hom needs to extend its graph without losing its attributes.
     "elevate": lambda self: HOM(self.s, *self),
     "device": property(lambda self: next(self.parameters()).device)
     })


def hom(name, sig, *steps):
    cls = type(name, (HOM,), {})
    return cls(sig, *steps)


def flatten(steps):
    """helper to flatten hierarchies of steps"""
    return [s for s in chain(*map(lambda s: (s,) if is_step(s) else flatten(s), steps))]


def Sum(sig):
    return lambda *args: sum(args), sig


def PPrintCtx():
    return print, "_ -> $"


def Maybe(cond, *steps):
    """
    make a step optional by returning step if cond is True else ()
    must be spread with `*` into the outer steps.
    """
    return steps if cond else tuple()


def Pack(signature):
    """put some inputs into one output variable as tuple"""
    return lambda *args: args, signature


def Unpack(signature):
    return lambda arg: (*arg,), signature


def Map(clb, sig):
    return lambda *args: tuple(map(clb, args)), sig


def Reduce(clb, sig):
    return lambda *args: reduce(clb, args), sig


def get_input_signature(clb, pos_args_suffix=""):
    if isinstance(clb, HOCBase):
        return clb.s.args, clb.s.full_kwargs
    pos, kw = [], []
    for k, v in signature(clb).parameters.items():
        if v.default == Parameter.empty and k not in ('self', 'kwargs'):
            pos += [k + pos_args_suffix]
        if v.default != Parameter.empty:
            kw += ["=".join([k, str(v.default)])]
    return ', '.join(pos) if pos else "", ', '.join(kw) if kw else ""


def combine(mode=None, reducer=None, *modules):
    """
    combine `modules` into single HOM with a basic graph controlled by `mode` and `reducer`

    - if mode == ">-" : each module gets its own input and all their outputs are reduced with `reducer`
    - if mode == "-<" : all modules receive the same input and all outputs are returned
    - if mode == "==" : each module gets its own input and all outputs are returned
    """
    in_pos, in_kw, mod_ins, mod_outs = [], [], [], []
    for i, mod in enumerate(modules):
        clb = mod.forward if isinstance(mod, nn.Module) else mod
        pos, kw = get_input_signature(clb, pos_args_suffix=str(i))
        in_pos += [pos]
        in_kw += [kw]
        mod_ins += [", ".join([pos, kw])]
    in_vars = in_pos + list(set(in_kw))
    mod_outs = tuple(f'_output_{i}' for i in range(len(modules)))
    rdc_ins = ""
    if mode == "==":
        if reducer is not None:
            raise ValueError("reducer must be None if mode is '='")
        in_ = ", ".join([v for v in in_vars if v])
        out_ = ", ".join(mod_outs)
    elif mode == ">-" or (mode is None and reducer is not None):
        rdc_ins = ", ".join(mod_outs)
        in_ = ", ".join(in_vars)
        out_ = '_output_'
        if reducer is None:
            reducer = operator.add
    elif mode == "-<" and reducer is None:
        in_ = f"_input_, {','.join(list(set(in_kw)))}"
        mod_ins = tuple(', '.join(['_input_', kw]) for kw in in_kw)
        mod_outs = tuple(f'_output_{i}' for i in range(len(modules)))
        rdc_ins = ""
        out_ = ", ".join(mod_outs)
    else:
        raise ValueError
    return HOM(f"{in_} -> {out_}",
               *[(clb, f"{vi} -> {vo}")
                 for clb, vi, vo in zip(modules, mod_ins, mod_outs)],
               *Maybe(reducer is not None,
                      Reduce(reducer, f"{rdc_ins} -> {out_}"))
               )


class Switch(HOCBase, nn.ModuleDict):
    _wrap_clb = staticmethod(_wrap_clb)

    __fname__ = "forward"

    def __init__(self, sig_str, cond, dct):
        super().__init__(sig_str)
        self.cond = cond
        self.signatures = {}
        self.update(dct)
        self.s = sig_str
        self.recompile()

    def update(self, dct):
        return super().update({k: self._wrap_clb(v) for k, v in dct.items()})

    def call(self, ctx):
        eval_k = str(self.cond(ctx))
        clb, sig = self[eval_k]
        ctx = eval_step(clb, sig, ctx)
        return get_out_(ctx, sig.out_)

    def __setitem__(self, item, value):
        clb, sig = value
        nn.ModuleDict.__setitem__(self, str(item), self._wrap_clb(clb))
        self.signatures[str(item)] = value


if __name__ == '__main__':
    import torch

    # Small demo :
    # HOM constructor takes a signature ("in -> out")
    # and steps, aka updates on the context, as tuples : (module or callable, "in -> out")
    # for instance :
    h = HOM(
        "x, y -> z",  # in/out signature
        # steps
        (nn.Linear(32, 32), "x -> x"),
        (nn.Linear(32, 32), "y -> y"),
        Sum("x, y -> z"),
        # works with ANY callable
        (lambda z: z.mean(dim=1), "z -> z")
    )

    # Now, h registered all passed modules and their params and its forward method has been 'compiled'.
    # Internally, forward always wraps h.call(ctx: dict), so that you could do :
    z = h.call({"x": torch.randn(32, 32), "y": torch.randn(32, 32)})
    # this is equivalent to h(x, y)

    # or if you want to define the class of the instance on the fly :
    h = hom("MyHOM", "x, y -> z", ...)

    # Signatures
    # - input signature can use the python idioms like "x, *others, training=False" etc.
    # - at all other places, variables are strings separated by comas.
    # there's 2 special characters :
    # 1/ to get or set the full context anywhere use the special string "_"
    # 2/ to NOT put a returned value in the context (void) use "$"

    # HOM subclasses nn.ModuleList, this means, you can :
    h.append((lambda z: z ** 2, "z -> z"))
    # and this modifies the forward pass!!

    # if you need to readjust the in/out signature, do :
    h.recompile("x, y -> y, z")

    # to make a step optional (based on the 'compilation' context!) use *Maybe(bool, *steps) in the list of steps
    with_relu = True
    g = HOM(
        "x, y -> z",  # in/out sig
        # steps
        (nn.Linear(32, 32), "x -> x"),
        (nn.Linear(32, 32), "y -> y"),
        Sum("x, y -> z"),
        *Maybe(with_relu,
               (nn.ReLU(), "z -> z")),
        # and also, you can of course access self! Like the real forward() ;)
        (lambda self: print(type(self)), "self -> $")
    )

    # And now, the best of the best :
    # `Switch` is a nn.ModuleDict that lets you program control flow (of course, on the fly)

    ch = Switch("x -> x",  # in/out
                lambda ctx: ctx['x'] > .5,  # takes the full context (dict) as input and returns the key to be evaluated
                {
                    True: (lambda x: 'hello', "x -> x"),
                    False: (lambda x: "Bye", 'x -> x')
                })

    assert ch(1) == "hello"
    # the "Bye" branch never got evaluated... :)

    # cool features ideas : h.trace() for debugging
    # and generally follow the dev of torch.fx !!
