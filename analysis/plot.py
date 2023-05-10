import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm

import awkward as ak
import uproot
import hist
import mplhep as hep
import coffea
from coffea import util as cutil
from coffea.lumi_tools import LumiData

import string
import warnings
from functools import partial
import inspect

import itertools
from scipy import optimize
from scipy import stats

def plot_cutflow(out, hep_args=None, legend_title=None, filename=None):
    cutflow = out["cutflow"]
    fig, ax = plt.subplots()
    x = np.arange(len(cutflow))
    y = cutflow.values()
    xlabels = cutflow.keys()
    ax.step(x, y, "|-", where="post")
    ax.set_xticks(x, xlabels, rotation = "vertical")
    for i, v in enumerate(y):
        ax.text(x[i]+0.05, v + 100, str(v), fontsize=10)
        ax.axvline(x[i], linestyle="dashed", alpha=0.1)
    
    format_plot(fig, ax, legend_title=legend_title, legend_loc="upper right", 
                hep_args=hep_args, hep_magic=False,
                filename=filename)