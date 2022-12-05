from typing import Any, Callable, Optional

import ipywidgets.widgets as W
import dataclasses as dtc


__all__ = [
    "Param",
    "ConfigView"
]


@dtc.dataclass()
class Param:
    name: str
    widget: W.Widget
    compute: Optional[Callable[[Any, Any], Any]] = None
    inverse_transform: Optional[Callable[[Any, Any], Any]] = None


class ConfigView:
    def __init__(self, config: Any, *params):
        self.config = config
        self._callbacks = []
        for i, param in enumerate(params):
            if param.name[0] != "_":  # starting with "_" -> no effect on config
                # link to config value
                def observer(ev, p=param):
                    compute = p.compute
                    v = ev["new"] if isinstance(ev, dict) else ev
                    val = v if compute is None else compute(config, v)
                    setattr(self.config, p.name, val)
                    self.callback()

                param.widget.observe(observer, "value")
        self.params = params

    def as_widget(self, container_cls, **kwargs):
        return container_cls(children=self.widgets, **kwargs)

    @property
    def widgets(self):
        return [p.widget for p in self.params]

    def apply(self):
        for p in self.params:
            v = p.compute(self.config, p.widget.value) if p.compute is not None else p.widget.value
            setattr(self.config, p.name, v)
        return self

    def callback(self):
        for cb in self._callbacks:
            cb(self.config)
        return self

    def observe(self, callback, _):
        self._callbacks.append(callback)
        return self

    def __repr__(self):
        return self.config.__repr__()