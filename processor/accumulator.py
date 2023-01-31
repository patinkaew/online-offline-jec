import numpy as np
from coffea.lumi_tools import LumiList

class LumiAccumulator(LumiList): # extention of coffea lumi list
    def __init__(self, runs=None, lumis=None, auto_unique=False):
        super().__init__(runs, lumis)
        self.auto_unique = auto_unique
    
    def unique(self):
        if len(self.array) > 0:
            self.array = np.unique(self.array, axis=0)
            
    def __add__(self, other):
        lumi_acc = LumiAccumulator()
        lumi_acc.array = np.r_[self.array, other.array]
        if self.auto_unique:
            lumi_acc.unique()
        return lumi_acc
    
    def __iadd__(self, other):
        super().__iadd__(other)
        if self.auto_unique:
            self.unique()
            
    __radd__ = __iadd__