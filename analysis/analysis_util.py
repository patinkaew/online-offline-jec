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
###### Luminosity #######
#########################

# goldenjson: https://twiki.cern.ch/twiki/bin/view/CMS/CertificationOfCollisions22
# lumidata: https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun3
# brilcalc lumi --normtag /cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json -u /fb -i ../lumimask/Cert_Collisions2022_355100_362760_Golden.json -b "STABLE BEAMS" --byls --output-style csv > lumi2022.csv
def compute_integrated_luminosity(out, lumi_csv_path, dataset=None, lumi=None):
    from coffea.lumi_tools import LumiData
    dataset = out["configurations"]["IO"]["dataset_names"] if dataset is None else dataset
    if eval(out["configurations"]["Processor"]["is_data"]): # data
        lumidata = LumiData(lumi_csv_path)
        out["processed_lumi"][dataset]["lumi_list"].unique() # apply unique
        lumi = lumidata.get_lumi(out["processed_lumi"][dataset]["lumi_list"])
        out["processed_lumi"][dataset]["lumi"] = lumi
    else: # mc
        # use the same lumi from data otherwise will sum all
        if lumi is None:
            lumidata = LumiData(lumi_csv_path)
            lumi = np.sum(lumidata._lumidata[:, 2])
        out["processed_lumi"] = {dataset:{"lumi_list": None, "lumi": lumi}}
    return lumi

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
def preprocess_histogram(h, dataset=None, jet_type=None, #correction_level=None, 
                          eta_range=None, phi_range=None, pt_range=None):
    # build selection dict
    selection_dict = dict()
    if dataset:
        selection_dict["dataset"] = dataset
    #if correction_level:
    #    selection_dict["correction_level"] = correction_level
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
def profile_mean(h, axis):
    return h.profile(axis)

def profile_median(h, axis):
    #if h.kind != bh.Kind.COUNT:
    #    raise TypeError("Profile requires a COUNT histogram")

    axes = list(h.axes)
    iaxis = axis if isinstance(axis, int) else h._name_to_index(axis)
    axes.pop(iaxis)

    values = h.values()
    tmp_variances = h.variances()
    variances = tmp_variances if tmp_variances is not None else values
    centers = h.axes[iaxis].centers

    count = np.sum(values, axis=iaxis)

    num = np.tensordot(values, centers, ([iaxis], [0]))
    num_err = np.sqrt(np.tensordot(variances, centers**2, ([iaxis], [0])))

    den = np.sum(values, axis=iaxis)
    den_err = np.sqrt(np.sum(variances, axis=iaxis))
    
    centers = h.axes[iaxis].centers
    edges = h.axes[iaxis].edges
    widths = h.axes[iaxis].widths
    
    freqs = h.values()
    total = np.sum(freqs, axis=iaxis, keepdims=True)
    cmf = np.cumsum(freqs, axis=iaxis)
    med_bin_idx = np.sum(cmf < total/2, axis=iaxis, keepdims=True)
    med_cmf_before = np.take_along_axis(cmf, med_bin_idx-1, axis=iaxis)
    med_freq = np.take_along_axis(freqs, med_bin_idx, axis=iaxis)
    med_freq = np.where(med_freq==0, np.nan, med_freq)
    median = edges[med_bin_idx] + (total/2 - med_cmf_before) * widths[med_bin_idx] / med_freq
    median = median.squeeze()
    
    # approximate with 1.2533 * mean_variances
    with np.errstate(invalid="ignore"):
        new_values = median
        new_variances = (num_err / den) ** 2 - (den_err * num / den**2) ** 2
        new_variances *= (1.2533**2)

    retval = h.__class__(*axes, storage=hist.storage.Mean())
    retval[...] = np.stack([count, new_values, count * new_variances], axis=-1)
    
    return retval

def compute_stats(h, axis, compute_median=True, approx_median=False):
    iaxis = axis if isinstance(axis, int) else h._name_to_index(axis)
    centers = h.axes[iaxis].centers
    edges = h.axes[iaxis].edges
    widths = h.axes[iaxis].widths
    
    freqs = h.values()
    total = np.sum(freqs, axis=iaxis, keepdims=True)
    cmf = np.cumsum(freqs, axis=iaxis)
    
    # compute mean
    h_pf = h.profile(iaxis)
    ave = h_pf.values()
    var = h_pf.variances()
    stdev = np.sqrt(var)
    mean_error = stdev #np.where(total.squeeze() > 0, stdev / np.sqrt(total).squeeze(), 0)
    
    # compute median
    median = None
    if compute_median:
        med_bin_idx = np.sum(cmf < total/2, axis=iaxis, keepdims=True)
        if approx_median:
            median = centers[med_bin_idx] # approx with center
            median = np.where(total > 0, median, np.nan)
        else:
            med_cmf_before = np.take_along_axis(cmf, med_bin_idx-1, axis=iaxis)
            med_freq = np.take_along_axis(freqs, med_bin_idx, axis=iaxis)
            med_freq = np.where(med_freq==0, np.nan, med_freq)
            median = edges[med_bin_idx] + (total/2 - med_cmf_before) * widths[med_bin_idx] / med_freq
    median = median.squeeze()
    median_error = 1.2533 * mean_error
    
    return {"centers": centers, "freqs": freqs, "mean": ave, \
            "stdev": stdev, "var": var, "mean_error": mean_error, \
            "median": median, "median_error": median_error}
    
def compute_stats_old(h, axis=-1, compute_median=True, compute_mode=False, approx_median=False, approx_mode=False):
    # retrieve axis and bin information
    if isinstance(axis, str):
        axis = get_axis_index(h, axis)
    h_axis = h.axes[axis]
    centers = h_axis.centers
    edges = h_axis.edges
    widths = h_axis.widths
    
    # retrieve counts (np.array)
    freqs = h.values()
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