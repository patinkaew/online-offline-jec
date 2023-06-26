from abc import ABCMeta, abstractmethod

class SelectorABC(metaclass=ABCMeta):
    def __init__(self, enable=True):
        self._enable = enable if isinstance(enable, bool) else False if enable is None else True
    def __str__(self):
        return "unnamed selector"
    def count(self, events): # return count for cutflow
        return len(events)
    def __call__(self, events, cutflow=None):
        if self._enable:
            if len(events) == 0:
                if cutflow != None:
                    cutflow[str(self)] += self.count(events)
                return events
            
            events = self.apply(events)
            if isinstance(events, tuple):
                events, _rest = events
                if cutflow != None:
                    cutflow[str(self)] += self.count(events)
                return events, _rest
            else:
                if cutflow != None:
                    cutflow[str(self)] += self.count(events)
                return events
            
        else:
            if cutflow != None:
                cutflow[str(self) + " (disabled)"] += self.count(events)
            return events
        
    def enable(self):
        self._enable = True
    def disable(self):
        self._enable = False
    
    @abstractmethod
    def apply(self, events):
        raise NotImplementedError
    
    @property
    def status(self):
        return self._enable
        
# class SelectorList():
#     def __init__(self, selector_list):
#         self._selector_list = selector_list
#     def add(self, selector):
#         self._selector_list.append(selector)
#     def on(self):
#         [selector.on() for selector in self._selector_list]
#     def off(self):
#         [selector.off() for selector in self._selector_list]
#     def set_status(self, status_list):
#         assert len(status_list) == len(self._selector_list), \
#                 "expect status list of length {}, get {}".format(len(self._selector_list), len(status_list))
#         [selector.on() if status else selector.off() for selector, status in zip(self._selector_list, status_list)]
    
#     @property
#     def status(self):
#         return [selector.status() for selector in self._selector_list]
    
#     def apply(self, events):
#         for selector in self._selector_list:
#             events = selector.apply(events)
#         return x
#     def __call__(self, events, cutflow=None):
#         for selector in self._selector_list:
#             events = selector(events, cutflow)
#         return events