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

#########################
### Histogram helpers ###
#########################

# get axis index corresponding to given axis name
def get_axis_index(h, name):
    for idx, axis in enumerate(h.axes):
        if axis.name == name:
            return idx

# get bin index corresponding to given category name 
# axis must be string category type
def get_string_category_bin_index(h, axis, name): 
    for idx, cat in enumerate(h.axes[axis]):
        if cat == name:
            return idx
# create a new string category axis with one gien category        
def make_single_string_category_axis(category, name, label=None):
    assert isinstance(category, str)
    label = label if label is not None else name
    return hist.axis.StrCategory([category], name=name, label=label)

# add a new string category axis to histogram
# this is generally used to add unique label to histogram (provided by this new axis)
# then, it can be stacked / merged with other histograms later
def add_histogram_axis(h, axis_index, new_axis):
    assert isinstance(new_axis, hist.axis.StrCategory), "New axis must be string category axis"
    assert len(new_axis) == 1, "New axis must have exactly one category"
    new_axes = list(h.axes)
    new_axes.insert(axis_index, new_axis)
    new_hist = hist.Hist(*new_axes, storage=h.storage_type())
    new_hist[{new_axis.name: new_axis[0]}] = h.view(flow=True)
    return new_hist

# merge two histograms using existing axis
# merging axis must be a string category
def merge_histograms_existing_axis(*hs, axis="dataset"):
    if len(hs) == 1: # if only one histogram is given, just return
        return hs[0]
    # checking
    assert all([isinstance(h.axes[axis], hist.axis.StrCategory) for h in hs]), \
            "Merge axis must be string category axis"
    assert all([len(h.axes) == len(hs[0].axes) for h in hs]), "Other axes must match"
    assert all([ax == hs[i].axes[ax.name] for h in hs for ax in h.axes if ax.name != axis for i in range(len(hs))]), \
           "Other axes must match"
    
    # build new axes
    merged_categories = [cat for h in hs for cat in h.axes[axis]]
    merged_axis = hist.axis.StrCategory(merged_categories, name=axis, label=hs[0].axes[axis].label, growth=True)
    merged_axes = list(hs[0].axes)
    merged_axes[get_axis_index(hs[0], axis)] = merged_axis # insert new axis
    
    # create merged histogram
    merged_hist = hist.Hist(*merged_axes, storage=hs[0].storage_type())
    
    # fill merged histogram
    for h in hs:
        for cat in h.axes[axis]: 
            merged_hist[{axis: cat}] = h[{axis:cat}].view(flow=True)
        
    return merged_hist

# merge two histograms by creating new axis
# h_dict = {name: hist}, name is used as category in merged axis 
# new axis will be a string category
def merge_histograms_new_axis(h_dict, name="untitled", label=None, index=0):
    if len(h_dict) == 1: # if only one histogram is given, just return
        return list(h_dict.values())[0]
    
    label = label if label is not None else name
    h_dict_items = h_dict.items()
    cats, hs = list(zip(*h_dict_items))
    assert all([len(h.axes) == len(hs[0].axes) for h in hs]), "All axes must match"
    assert all([ax == hs[i].axes[ax.name] for h in hs for ax in h.axes for i in range(len(hs))]), \
           "All axes must match"
    # build new axis
    new_axis = hist.axis.StrCategory(cats, name=name, label=label, growth=True)
    merged_axes = list(hs[0].axes)
    merged_axes.insert(index, new_axis)
    
    # create merged histogram
    merged_hist = hist.Hist(*merged_axes, storage=hs[0].storage_type())
    
    # fill merged histogram
    for h in hs:
        for cat, h in h_dict_items: 
            merged_hist[{name: cat}] = h.view(flow=True)
    return merged_hist

# merge histograms from stack of histograms
def merge_histograms_from_stack(h_stack, name="untitled", label=None, index=0):
    h_dict = {h.name:h for h in h_stack}
    return merge_histograms_new_axis(h_dict, name=name, label=label, index=index)

# merge histograms
# this is one stop function which will call appropriate merge histogram functions based on given arguments
def merge_histograms(*hs, axis="dataset", label=None, index=0):
    if len(hs) == 1 and isinstance(hs[0], dict):
        return merge_histograms_new_axis(hs[0], name=axis, label=label, index=index)
    elif len(hs) == 1 and isinstance(hs[0], hist.stack.Stack):
        return merge_histograms_from_stack(hs[0], name=axis, label=label, index=index)
    else:
        if axis in hs[0].axes.name:
            return merge_histograms_existing_axis(*hs, axis=axis)
        else:
            # create h_dict with default dict keys
            h_dict = {str(i):h for i, h in enumerate(hs)}
            return merge_histograms_new_axis(h_dict, name=axis, label=label, index=index)

# merge three response histograms together
# this is used in very old design of processor
# currently, label raw,original,corrected are grouped as correction_level
# which is represented as additional string category axis
def merge_response_histograms(out, dataset):
    response_name=["raw_pt_response", "original_pt_response", "corrected_pt_response"]
    h_resp_list = [out[resp][{"dataset":dataset}] for resp in response_name]
    h_resp_dict = {h.label:h for h in h_resp_list}
    h_resp_stack = hist.Stack.from_dict(h_resp_dict)
    return merge_histograms(h_resp_stack, axis="response", label="Response")

# integrate out all axes excluding in reduced_axes
def reduce_histogram(h, reduced_axes):
    assert all([axis in h.axes.name for axis in reduced_axes]) # reduced_axes must be the subset of original axes
    reduction_dict = {axis:sum for axis in h.axes.name if axis not in reduced_axes}
    return h[reduction_dict]

# integrate given axis
# this handles more complicated integrating histogram bins
# if upper (lower) is None, will integrate to last (first) bins
# e.g. if both upper = lower = None, integrate all bins
# mirror = reflection about zero (this is useful for integrate eta)
# mirror = True will integrate (-upper, -lower) and (lower, upper)
def integrate_histogram(h, axis, lower=0, upper=None, mirror=False): # experimental, please use properly
    if upper is None and lower is None: # upper = lower = None, integrate all
            return h[{axis: sum}]
    elif upper is None and lower is not None: # upper = None, lower != None
        if mirror: # mirror=True, integrate (-infty, -lower) + (lower, infty)
            return h[{axis: sum}] - h[{axis: slice(-lower * 1j, lower*1j, sum)}]
        else: # mirror=False, integrate (lower, infty)
            return h[{axis: slice(lower*1j, np.inf*1j, sum)}]
    elif upper is not None and lower is None: # upper != None, lower = None, integrate (-infty, upper)
        # mirror does not make sense in this case
        return h[{axis: slice(-np.inf*1j, upper, sum)}]
    else: # upper != None, lower != None
        upper, lower = (upper, lower) if upper > lower else (lower, upper)
        rslt = h[{axis: slice(lower * 1j, upper*1j,sum)}]
        if not mirror:
            return rslt
        else:
            return rslt + h[{axis: slice(-upper * 1j, -lower*1j,sum)}]

# common histogram preprocessing
# this combine selecting and integrating
# None = do not integrate given axis
# sum = integrate out axis
# range = (lower, upper, mirror) which is passed to integrate_histogram
def preprocess_histogram(h, dataset=None, correction_level=None, jet_type=None, 
                          eta_range=None, phi_range=None, pt_range=None):
    # build selection dict
    selection_dict = dict()
    if dataset:
        selection_dict["dataset"] = dataset
    if correction_level:
        selection_dict["correction_level"] = correction_level
    if jet_type:
        selection_dict["jet_type"] = jet_type
    if eta_range == sum:
        selection_dict["jet_eta"] = sum
    if phi_range == sum:
        selection_dict["jet_phi"] = sum
    if pt_range == sum:
        selection_dict["jet_pt"] = sum
    
    # apply selection dict
    h = h[selection_dict]
    
    # integrate
    if eta_range and eta_range != sum:
        if len(eta_range) == 2:
            eta_range = eta_range + (True,) # mirror integrate
        h = integrate_histogram(h, "jet_eta", *eta_range)
    if phi_range and phi_range != sum:
        if len(phi_range) == 2: 
            phi_range = phi_range + (False,)
        h = integrate_histogram(h, "jet_phi", *phi_range)
    if pt_range and pt_range != sum:
        if len(pt_range) == 2:
            pt_range = pt_range + (False,)
        h = integrate_histogram(h, "jet_pt", *pt_range)
        
    return h

def repeat_value_from_freq(values, freqs):
    return list(itertools.chain.from_iterable([itertools.repeat(elem, n) for elem, n in zip(values, freqs)]))

def build_points_from_histogram(h):
    assert len(h.axes.name) == 2, "Histogram must be two dimensional"
    w = h.counts()
    x = h.axes[0].centers
    y = h.axes[1].centers
    values = itertools.product(x, y)
    points = np.array(repeat_value_from_freq(values, w.reshape(-1)))
    return points

def format_correction_level(off_correction_level, on_correction_level=None):
    if not on_correction_level:
        on_correction_level = off_correction_level
        if len(off_correction_level.split(":")) == 2: # already in correct format
            return off_correction_level
    return "off={}:on={}".format(off_correction_level, on_correction_level)

def extract_correction_level(correction_level):
    off_correction_level, on_correction_level = correction_level.split(":")
    return off_correction_level.split("=")[1], on_correction_level.split("=")[1]
    
##############################
### Statistics and Fitting ###
##############################
def compute_stats(h, axis=-1, compute_median=True, compute_mode=False, approx_median=False, approx_mode=False):
    # retrieve axis and bin information
    if isinstance(axis, str):
        axis = get_axis_index(h, axis)
    h_axis = h.axes[axis]
    centers = h_axis.centers
    edges = h_axis.edges
    widths = h_axis.widths
    
    # retrieve counts (np.array)
    freqs = h.counts()
    freqs = np.swapaxes(freqs, axis, -1) # swap to last
    total = np.sum(freqs, axis=-1)
    cmf = np.cumsum(freqs, axis=-1)
    
    # compute mean
    ave = np.sum(centers * freqs, axis=-1) / total
    var = np.sum((centers**2) * freqs, axis=-1) / total - (ave ** 2)
    var = np.where(total > 0, var, 0)
    stdev = np.sqrt(var)
    mean_error = np.where(total > 0, stdev / np.sqrt(total), 0)
    
    # compute median
    median = None
    if compute_median:
        med_bin_idx = np.sum(cmf < np.expand_dims(total/2, axis=-1), axis=-1)
        if approx_median:
            median = centers[med_bin_idx] # approx with center
            median = np.where(total > 0, median, np.nan)
        else:
            med_cmf_before = np.take_along_axis(cmf, np.expand_dims(med_bin_idx-1, axis=-1), axis=-1).squeeze()
            med_freq = np.take_along_axis(freqs, np.expand_dims(med_bin_idx, axis=-1), axis=-1).squeeze()
            med_freq = np.where(med_freq==0, np.nan, med_freq)
            median = edges[med_bin_idx] + (total/2 - med_cmf_before) * widths[med_bin_idx] / med_freq
            
    median_error = 1.2533 * mean_error
    
    # compute mode
    mode = None
    if compute_mode:
        mod_bin_idx = np.argmax(freqs, axis=dep_ax)
        if approx_mode:
            mode = centers[mod_bin_idx] # approx with center
            mode = np.where(total > 0, mode, np.nan)
        if not approx_mode: # experimental
            mod_freq = np.take_along_axis(freqs, np.expand_dims(mod_bin_idx, axis=-1), axis=-1).squeeze()
            diff_low = mod_freq - np.take_along_axis(freqs, np.expand_dims(mod_bin_idx-1, axis=-1), axis=-1).squeeze()
            diff_high = mod_freq - np.take_along_axis(freqs, np.expand_dims(mod_bin_idx+1, axis=-1), axis=-1).squeeze()
            mode = edges[mod_bin_idx] + (diff_low / (diff_low + diff_high)) * widths[med_bin_idx]
    
    freqs = np.swapaxes(freqs, axis, -1) # swap back
            
    return {"centers": centers, "freqs": freqs, "mean": ave, \
            "stdev": stdev, "var": var, "mean_error": mean_error, \
            "median": median, "median_error": median_error, \
            "mode": mode}

################
### Plotting ###
################

# strike text
def strike(text):
    result = ""
    for c in text:
        result = result + c + "\u0336"
    return result

# return range text as str
def format_range_text(lower, upper, var, absolute=False, omit_zero=True):
    if lower > upper:
        lower, upper = upper, lower
    if absolute:
        var = "|{}|".format(var)
    if omit_zero and lower == 0:
        return "{} < {}".format(var, upper)
    else:
        return "{} < {} < {}".format(lower, var, upper)

# quick pre-configured for eta range 
def format_eta_range_text(lower, upper, jet_name="jet", absolute=True, omit_zero=False):
    return format_range_text(lower, upper, "$\eta^{%s}$"%(jet_name), absolute=absolute, omit_zero=omit_zero)

# quick pre-configured for pt range 
def format_pt_range_text(lower, upper, jet_name="jet", omit_zero=True):
    return format_range_text(lower, upper, "$p_T^{%s}$"%(jet_name), omit_zero=omit_zero)

# quick pre-configured for phi range 
def format_phi_range_text(lower, upper, jet_name="jet", omit_zero=False):
    return format_range_text(lower, upper, "$\phi^{%s}$"%(jet_name), omit_zero=omit_zero)

# save figure to file
def save_figure(fig, filename=None, dpi=100):
    if filename:
        fig.savefig(filename, bbox_inches="tight", dpi=dpi)

# wrapper to call appropriate matplotlib functions
def format_plt_plot(ax,
                    xscale=None, yscale=None,
                    xlim=None, ylim=None,
                    legend_title=None, legend_loc=0, legend_args=None):
    
    # axis scaling
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
    
    # setting axis limit
    # left=None, right=None, *, emit=True, auto=False, xmin=None, xmax=None
    if xlim:
        if isinstance(xlim, set) or isinstance(xlim, tuple):
            ax.set_xlim(*xlim)
        if isinstance(xlim, dict):
            ax.set_xlim(**xlim)
    if ylim:
        if isinstance(ylim, set) or isinstance(ylim, tuple):
            ax.set_ylim(*ylim)
        if isinstance(ylim, dict):
            ax.set_ylim(**xlim)
    
    # formatting legend
    if legend_title:
        ax.legend(title=legend_title, loc=legend_loc)
    if legend_args: # other arguments, see matplotlib document for reference
        if legend_title and "title" not in legend_args:
            legend_args["title"] = legend_title
        if legend_loc and "loc" not in legend_args:
            legend_args["loc"] = legend_loc
        ax.legend(**legend_args)

# wrapper to call appropriate mplhep functions
def format_hep_plot(ax,
                    hep_args=None, with_cms_name=True,
                    hep_magic=False):
    if hep_args:
        if with_cms_name:
            hep.cms.label(**hep_args)
        else: # experimental, x, y, and fontsize may need to be adjusted manually
            if hep_args["data"]:
                ax.set_title("private", style="italic", ha="left", x=0, y=1.005, fontsize=28)
            else:
                ax.set_title("Simulation private", style="italic", ha="left", x=0, y=1.005, fontsize=28)
                
    if hep_magic:
        try:
            hep.mpl_magic(ax=ax)
        except:
            warnings.warn("mplhep magic fails! trying yscale_legend")
            try:
                hep.plot.yscale_legend(ax=ax)
            except:
                warnings.warn("mplhep yscale also fails!")

# combine both wrappers and save figure
# this is generally called at the end
def format_plot(fig, ax,
                xscale=None, yscale=None,
                xlim=None, ylim=None,
                legend_title=None, legend_loc=0, legend_args=None, 
                hep_args=None, with_cms_name=True,
                hep_magic=False,
                filename=None, dpi=100):
    
    format_plt_plot(ax, xscale=xscale, yscale=yscale, xlim=xlim, ylim=ylim, 
                    legend_title=legend_title, legend_loc=legend_loc, legend_args=legend_args)
    
    if hep_args or hep_magic:
        format_hep_plot(ax, hep_args=hep_args, with_cms_name=with_cms_name, hep_magic=hep_magic)
        
    save_figure(fig, filename, dpi)