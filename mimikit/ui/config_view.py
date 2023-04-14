from typing import Any, Callable, Optional, Tuple, Union

import ipywidgets.widgets as W
import dataclasses as dtc


__all__ = [
    "Param",
    "ConfigView"
]

from ipywidgets import GridspecLayout


@dtc.dataclass()
class Param:
    name: str
    # label: W.Widget
    widget: W.Widget
    setter: Optional[Callable[[Any, Any], Any]] = None
    inverse_transform: Optional[Callable[[Any, Any], Any]] = None
    position: Optional[Tuple[Union[int, slice], Union[int, slice]]] = None


class ConfigView:
    def __init__(self, config: Any, *params, grid_spec=None):
        self.config = config
        self._callbacks = []
        if grid_spec is not None:
            self.grid = GridspecLayout(*grid_spec, grid_gap='8px 8px')
        else:
            self.grid = GridspecLayout(len(params), 1, grid_gap='4px 8px')
        for i, param in enumerate(params):
            if param.name[0] != "_":  # starting with "_" -> no effect on config
                # link to config value
                def observer(ev, p=param):
                    setter = p.setter
                    v = ev["new"] if isinstance(ev, dict) else ev
                    val = v if setter is None else setter(config, v)
                    setattr(self.config, p.name, val)
                    self.callback()

                param.widget.observe(observer, "value")
            if param.position is not None:
                self.grid[param.position] = param.widget
            else:
                self.grid[i, 0] = param.widget
        self.params = params

    def as_widget(self, container_cls, **kwargs):
        return container_cls(children=(self.grid,), **kwargs)

    @property
    def widgets(self):
        return [p.widget for p in self.params]

    def apply(self):
        for p in self.params:
            v = p.setter(self.config, p.widget.value) if p.setter is not None else p.widget.value
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