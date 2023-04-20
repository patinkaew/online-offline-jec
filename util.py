import os
import shutil
import string
import re
import json
import inspect
import configparser

import numpy as np
import uproot
import awkward as ak
import hist

def mkdir_if_not_exist(path):
    if (len(path) > 0) and (not os.path.exists(path)):
        os.makedirs(path)
        
def compress_folder(folder_path, out_filename, filetype="zip"):
    if os.path.exists(folder_path):
        shutil.make_archive(out_filename, filetype, folder_path)
    else:
        print("folder: {} does not exist!".format(folder_path))
        
def has_whitespace(s):
    return any([c in s for c in string.whitespace])

def count_whitespace(s):
    return sum([c in s for c in string.whitespace])

def extract_filename(path, remove_format=True):
    start_idx = path.rfind("/") + 1
    end_idx = -1
    if remove_format:
        end_idx = path.rfind(".")
    if start_idx != -1:
        return path[start_idx:end_idx]
    else:
        return path[:end_idx]
    
def json2dict(filename):
    with open(filename) as file:
        dictionary = json.load(file)
    return dictionary

def dict2json(dictionary, filename, indent=4):
    with open(filename, "w") as file:
        json.dump(dictionary, file, indent=indent)
    return filename
    
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def print_dict_json(dictionary, title):
    print("="*50)
    print(title)
    print(json.dumps(dictionary, indent = 4, cls=NpEncoder))
    print("="*50)

def create_default_config_file(processor_class, config_file):
    assert not os.path.exists(config_file), "config file: {} already exist".format(config_file)
    with open(config_file, "w") as f:
        f.write("[Processor]\n")
        for parameter in inspect.signature(processor_class).parameters.values():
            default = parameter.default if parameter.default != inspect._empty else ""
            f.write("{} = {}\n".format(parameter.name, default))
        
        f.write("\n[Runner]\n")
        f.write("executor = iterative\n")
        f.write("num_workers = 1\n")
        f.write("chunksize = 100_000\n")
        f.write("maxchunks = None")
        
# update config file to contain new required fields with default values
def update_config_file(default_config_dict, config_file, replace=False):
    new_configs = configparser.ConfigParser()
    # config file does not exist, create new with default values
    if not os.path.exists(config_file): 
        for section in default_config_dict.keys():
            new_configs[section] = {}
            for name, default_value in default_config_dict[section]:
                new_configs[section][name] = str(default_value)              
    # config file exists, update to contain new required fields with default values
    else:
        if not replace:
            shutil.copy(config_file, os.path.splitext(config_file)[0] + "_orig.cfg") # save original

        orig_configs = configparser.ConfigParser()
        orig_configs.read(config_file)

        for section in default_config_dict.keys():
                new_configs[section] = {}
                for name, default_value in default_config_dict[section]:
                    if section in orig_configs.keys():
                        value = orig_configs[section].get(name, default_value)
                    else:
                        value = default_value
                    new_configs[section][name] = str(value)
    with open(config_file, "w") as f:
        new_configs.write(f)
        
# functions for manipulating trigger flags
# def find_available_trigger_flags(events, flag_prefix="PFJet"):
#     return [flag for flag in events["HLT"].fields if re.match("{}\d+".format(flag_prefix), flag)]

# def make_trigger_flag(value, flag_prefix="PFJet"):
#     return flag_prefix + str(value)

# def extract_trigger_value(flag):
#     idx = sum([not c.isdigit() for c in flag])
#     return (flag[:idx], int(flag[idx:])) if idx != len(flag) else None

class TriggerUtil():
    def find_available_trigger_flags(self, events, flag_prefix="PFJet"):
        return [flag for flag in events["HLT"].fields if re.match("{}\d+".format(flag_prefix), flag)]

    def make_trigger_flag(self, value, flag_prefix="PFJet"):
        return flag_prefix + str(value)

    def extract_trigger_value(self, flag):
        idx = sum([not c.isdigit() for c in flag])
        return (flag[:idx], int(flag[idx:])) if idx != len(flag) else None

# definitions of common use trigger cuts
# def trigger_cut_base(events, trigger_value, flag_prefix="PFJet", trigger_values=None, comparison_operation=None):
#     if trigger_values is None or comparison_operation is None:
#         # alternatively, setting trigger_values = [trigger_value] will give the same result
#         return events.HLT[make_trigger_flag(trigger_value, flag_prefix)]
#     mask = events.HLT[make_trigger_flag(trigger_value, flag_prefix)]
#     for value in trigger_values:
#         if comparison_operation(value, trigger_value):
#             mask = mask & ~events.HLT[make_trigger_flag(value, flag_prefix)]
#     return mask

# def trigger_cut_single(events, trigger_value, flag_prefix="PFJet"):
#     # can also use trigger_cut_base, but we can reduce one function call here
#     return events.HLT[make_trigger_flag(trigger_value, flag_prefix)]  

# def trigger_cut_lower_not(events, trigger_value, flag_prefix="PFJet", trigger_values=None):
#     return trigger_cut_base(events, trigger_value, flag_prefix, trigger_values, 
#                             lambda value, trigger_value: value <= trigger_value)

# def trigger_cut_upper_not(events, trigger_value, flag_prefix="PFJet", trigger_values=None):
#     return trigger_cut_base(events, trigger_value, flag_prefix, trigger_values, 
#                             lambda value, trigger_value: value >= trigger_value)

# def trigger_cut_only(events, trigger_value, flag_prefix="PFJet", trigger_values=None):
#     return trigger_cut_base(events, trigger_value, flag_prefix, trigger_values, 
#                             lambda value, trigger_value: value != trigger_value)

# this implements the above trigger cut functions in a class
# this allows trigger cut functions to be used as a interface
class TriggerCut(TriggerUtil):
    def trigger_cut_base(self, events, trigger_value, flag_prefix="PFJet", trigger_values=None, comparison_operation=None):
        if trigger_values is None or comparison_operation is None:
            # alternatively, setting trigger_values = [trigger_value] will give the same result
            return events.HLT[self.make_trigger_flag(trigger_value, flag_prefix)]
        mask = events.HLT[self.make_trigger_flag(trigger_value, flag_prefix)]
        for value in trigger_values:
            if comparison_operation(value, trigger_value):
                mask = np.logical_and(mask, np.logical_not(events.HLT[self.make_trigger_flag(value, flag_prefix)]))
        return mask

    def trigger_cut_single(self, events, trigger_value, flag_prefix="PFJet"):
        # can also use trigger_cut_base, but we can reduce one function call here
        return events.HLT[self.make_trigger_flag(trigger_value, flag_prefix)]  

    def trigger_cut_lower_not(self, events, trigger_value, flag_prefix="PFJet", trigger_values=None):
        return self.trigger_cut_base(events, trigger_value, flag_prefix, trigger_values, 
                                     lambda value, trigger_value: value < trigger_value)

    def trigger_cut_upper_not(self, events, trigger_value, flag_prefix="PFJet", trigger_values=None):
        return self.trigger_cut_base(events, trigger_value, flag_prefix, trigger_values, 
                                     lambda value, trigger_value: value > trigger_value)

    def trigger_cut_only(self, events, trigger_value, flag_prefix="PFJet", trigger_values=None):
        return self.trigger_cut_base(events, trigger_value, flag_prefix, trigger_values, 
                                     lambda value, trigger_value: value != trigger_value)
    
    def apply_trigger_cut(self, events, trigger_value, trigger_cut, flag_prefix="PFJet", trigger_values=None):
        if trigger_cut == "None" or trigger_cut == None:
            return events
        elif trigger_cut == "single":
            return events[self.trigger_cut_single(events, trigger_value, flag_prefix)]
        elif trigger_cut == "only":
            return events[self.trigger_cut_only(events, trigger_value, flag_prefix, trigger_values)]
        elif trigger_cut == "lower_not":
            return events[self.trigger_cut_lower_not(events, trigger_value, flag_prefix, trigger_values)]
        elif trigger_cut == "upper_not":
            return events[self.trigger_cut_upper_not(events, trigger_value, flag_prefix, trigger_values)]
        else:
            raise ValueError("Invalid types of trigger_cut")

# dijet tag and probe
# def trigger_dijet_tag_probe(jets, trigger_value, twosides=True):
#     if len(jets) == 0:
#         return jets
    
#     tag_index = 0
#     probe_index = 1
#     assert all(ak.num(jets) >= 2) # at least 2 jets
        
#     def make_tag_probe_mask(tag_index, probe_index):
#         tag = jets[:, tag_index]
#         probe = jets[:, probe_index]
#         tag_cut = (tag.pt > trigger_value)
#         opposite_cut = (np.abs(tag.phi - probe.phi) > 2.7)
#         close_pt_cut = (np.abs(tag.pt - probe.pt) < 0.7 * (tag.pt + probe.pt))
#                         #0.5 * ak.max((tag.pt, probe.pt), axis=0))
#         # alpha_cut = (2*jets[:, 2].pt < 1.0 * (tag.pt + probe.pt)) TODO
#         tag_probe_selection = tag_cut & opposite_cut & close_pt_cut

#         mask = ak.zeros_like(jets.pt) # all zeros mask
#         count = ak.num(jets.pt)
#         probe_index_flatten = np.concatenate(([0], np.cumsum(count)[:-1])) + probe_index
#         mask_flatten = np.asarray(ak.flatten(mask), dtype=np.bool_) # flatten to np and unflatten to ak
#         mask_flatten[probe_index_flatten] = tag_probe_selection
#         mask = ak.unflatten(mask_flatten, count)
#         return mask

#     mask = make_tag_probe_mask(tag_index, probe_index)

#     if twosides:
#         mask = mask | make_tag_probe_mask(probe_index, tag_index)
    
#     return jets[mask]

# this implements the above trigger_dijet_tag_probe in a class
# this allows trigger cut functions to be used as a interface
class TriggerDijetTagProbe():
    def apply_trigger_dijet_tag_probe(self, jets, trigger_value, twosides=True):
        if len(jets) == 0:
            return jets

        tag_index = 0
        probe_index = 1
        assert all(ak.num(jets) >= 2) # at least 2 jets

        def make_tag_probe_mask(tag_index, probe_index):
            tag = jets[:, tag_index]
            probe = jets[:, probe_index]
            tag_cut = (tag.pt > trigger_value)
            opposite_cut = (np.abs(tag.phi - probe.phi) > 2.7)
            close_pt_cut = (np.abs(tag.pt - probe.pt) < 0.7 * (tag.pt + probe.pt))
                            #0.5 * ak.max((tag.pt, probe.pt), axis=0))
            # alpha_cut = (2*jets[:, 2].pt < 1.0 * (tag.pt + probe.pt)) TODO
            tag_probe_selection = tag_cut & opposite_cut & close_pt_cut

            mask = ak.zeros_like(jets.pt) # all zeros mask
            count = ak.num(jets.pt)
            probe_index_flatten = np.concatenate(([0], np.cumsum(count)[:-1])) + probe_index
            mask_flatten = np.asarray(ak.flatten(mask), dtype=np.bool_) # flatten to np and unflatten to ak
            mask_flatten[probe_index_flatten] = tag_probe_selection
            mask = ak.unflatten(mask_flatten, count)
            return mask

        mask = make_tag_probe_mask(tag_index, probe_index)

        if twosides:
            mask = mask | make_tag_probe_mask(probe_index, tag_index)

        return jets[mask]
    