from abc import ABCMeta, abstractmethod

class SelectorABC(metaclass=ABCMeta):
    def __init__(self):
        self._on = True
    def __str__(self):
        return "unnamed selector"
    def __call__(self, x, cutflow=None):
        if self._on:
            if len(x) == 0:
                if cutflow != None:
                    cutflow[str(self)] += len(x)
                return x
            x = self.apply(x)
            if cutflow != None:
                cutflow[str(self)] += len(x)
            return x
        else:
            if cutflow != None:
                cutflow[str(self) + " (off)"] += len(x)
            return x
    def on(self):
        self._on = True
    def off(self):
        self._on = False
    
    @abstractmethod
    def apply(self, x):
        raise NotImplementedError
    
    @property
    def status(self):
        return self._on
        
class SelectorList():
    def __init__(self, selector_list):
        self._selector_list = selector_list
    def add(self, selector):
        self._selector_list.append(selector)
    def on(self):
        [selector.on() for selector in self._selector_list]
    def off(self):
        [selector.off() for selector in self._selector_list]
    def set_status(self, status_list):
        assert len(status_list) == len(self._selector_list), \
                "expect status list of length {}, get {}".format(len(self._selector_list), len(status_list))
        [selector.on() if status else selector.off() for selector, status in zip(self._selector_list, status_list)]
    
    @property
    def status(self):
        return [selector.status() for selector in self._selector_list]
    
    def apply(self, x):
        for selector in self._selector_list:
            x = selector.apply(x)
        return x
    def __call__(self, x, cutflow=None):
        for selector in self._selector_list:
            x = selector(x, cutflow)
        return x