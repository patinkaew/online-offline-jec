import numpy as np
import awkward as ak
import hist

from coffea import processor
from coffea.lumi_tools import LumiMask, LumiData, LumiList

from processor.selector import *
from processor.selectorbase import SelectorList
from processor.accumulator import LumiAccumulator

from collections import defaultdict
import itertools
import warnings

import time

class SimpleProcessor(processor.ProcessorABC):
    def __init__(self):
        pass
    def process(self, events):
        print("nevents: ", len(events))
        return {"nevents":len(events)}
    def postprocess(self, accumulator):
        return accumulator

class OHProcessor(processor.ProcessorABC):
    def __init__(self, 
                 off_jet_name, off_jet_label=None, # offline jet
                 on_jet_name="TrigObjJMEAK4", on_jet_label=None, # online jet
                 lumi_json_path=None, # path to gloden json file (good run certification)
                 lumi_csv_path=None, # path to output lumi csv from brilcalc
                 save_processed_lumi=True, # save processed lumi section (only data)
                 compute_processed_lumi=True, # compute integrated luminosity at post-processing (only data)
                 
                 # event-level selections
                 flag_filters=None,  # event_level, apply flag filters, e.g. METfilters
                 min_off_jet=1, min_on_jet=1, # event-level, select events with at least n jets
                 MET_type="MET", max_MET=None, # event-level, select max MET and/or max MET/sumET
                 max_MET_sumET=None, min_MET=45, 
                 trigger_min_pt=0, # event-level, trigger cut
                 trigger_type=None, trigger_flag_prefix="PFJet", trigger_all_pts=None,
                 
                 # jet-level selections
                 off_jet_Id=None, on_jet_Id=None, # jet-level, jet id cut
                 off_jet_veto_map_json_path=None, on_jet_veto_map_json_path=None, # jet-level, jet veto map cut
                 off_jet_veto_map_correction_name=None, on_jet_veto_map_correction_name=None,
                 off_jet_veto_map_year=None, on_jet_veto_map_year=None, 
                 off_jet_veto_map_type="jetvetomap", on_jet_veto_map_type="jetvetomap",
                 off_jet_weight_filelist=None, on_jet_weight_filelist=None, # weight file for JEC
                 off_rho_name=None, on_rho_name=None, # rho to use in JEC
                 off_jet_tag_probe=True, on_jet_tag_probe=True, # whether to apply tag and probe
                 off_jet_tag_min_pt=0, on_jet_tag_min_pt=0, # tag min pt to apply during tag and probe
                 off_jet_max_alpha=1.0, on_jet_max_alpha=1.0, # max alpha during tag and probe
                 
                 max_deltaR=0.2, # for deltaR matching
                 max_leading_jet=2, # select up to n leading jets to fill histograms
                 
                 # histograms
                 is_data=True, # data or simulation
                 #storage=hist.storage.Weight(), # storage type for Hist histograms
                 mix_correction_level=False,
                 pt_binning="log",
                 eta_binning="coarse",
                 # phi_binning=None,
                 fill_gen=False, # (only MC)
                 hist_to_fill="all",
                 
                 verbose=0):
        
        # which online and offline to use
        # name is used to retrieve 
        # label is used for histograms + plots
        self.off_jet_name = off_jet_name
        off_jet_label = off_jet_label if off_jet_label != None else off_jet_name
        self.off_jet_label = off_jet_label
        self.on_jet_name = on_jet_name
        on_jet_label = on_jet_label if on_jet_label != None else on_jet_name
        self.on_jet_label = on_jet_label
        
        # luminosity
        self.lumimask = LumiMask(lumi_json_path) if lumi_json_path else None
        self.save_processed_lumi = save_processed_lumi if not compute_processed_lumi else True
        self.compute_processed_lumi = compute_processed_lumi
        if is_data and compute_processed_lumi:
            assert lumi_csv_path != None, "Require brilcalc output csv to compute processed integrated luminosity."
        self.lumi_csv_path = lumi_csv_path
        
        # processing pipeline
        # event-level selections
        # good event cuts
        self.min_npvgood = MinNPVGood(min_NPVGood=0)
        self.max_pv_z = MaxPV_z(max_PV_z=24)
        self.max_pv_rxy = MaxPV_rxy(max_PV_rxy=2)
        
        # flag_filters
        flag_filters = flag_filters if flag_filters else []
        flag_filters = [flag_filters] if isinstance(flag_filters, str) else flag_filters
        self.flag_filters = SelectorList([FlagFilter(flag_filter) for flag_filter in flag_filters])
        
        # minimum number of jets
        # if tag and probe will be applied, need at least 2
        min_off_jet = min_off_jet if not off_jet_tag_probe else max(min_off_jet, 2) 
        min_on_jet = min_on_jet if not on_jet_tag_probe else max(min_on_jet, 2) 
        self.min_off_jet = MinPhysicsObject(off_jet_name, min_off_jet, name=off_jet_label)
        self.min_on_jet = MinPhysicsObject(on_jet_name, min_on_jet, name=on_jet_label)
        
        # MET cut
        self.max_MET = MaxMET(max_MET, MET_type)
        self.max_MET_sumET = MaxMET_sumET(max_MET_sumET, min_MET, MET_type)
        
        # trigger cut
        self.min_trigger = MinTrigger(trigger_type, trigger_min_pt, trigger_flag_prefix, trigger_all_pts)
        
        # event-level wrapped jet-level selectors
        # intuitively, these selectors act on jet
        # however, cut earlier is better for performance
        # and make it easier to tag and probe
        off_jet_identification = JetIdentification(off_jet_Id, off_jet_label, verbose)
        self.off_jet_Id = EventWrappedPhysicsObjectSelector(off_jet_name, off_jet_identification, discard_empty=True)
        on_jet_identification = JetIdentification(on_jet_Id, on_jet_label, verbose)
        self.on_jet_Id = EventWrappedPhysicsObjectSelector(on_jet_name, on_jet_identification, discard_empty=True)
        
        off_jet_veto_map = JetVetoMap(off_jet_veto_map_json_path, off_jet_veto_map_correction_name, 
                                      off_jet_veto_map_year, off_jet_veto_map_type, off_jet_label)
        self.off_jet_veto_map = EventWrappedPhysicsObjectSelector(off_jet_name, off_jet_veto_map, discard_empty=True)
        
        on_jet_veto_map = JetVetoMap(on_jet_veto_map_json_path, on_jet_veto_map_correction_name, 
                                     on_jet_veto_map_year, on_jet_veto_map_type, on_jet_label)
        self.on_jet_veto_map = EventWrappedPhysicsObjectSelector(on_jet_name, on_jet_veto_map, discard_empty=True)
        
        # jet-level selections
        # apply to offline jet
        self.off_jet_JEC = JECBlock(off_jet_weight_filelist, off_rho_name, name=off_jet_label, verbose=verbose)
        self.off_jet_tagprobe = TriggerDijetTagAndProbe(off_jet_tag_min_pt if off_jet_tag_probe else None, 
                                                        max_alpha=off_jet_max_alpha, swap=True, name=off_jet_label)
        
        # apply to online jet
        self.on_jet_JEC = JECBlock(on_jet_weight_filelist, on_rho_name, name=on_jet_label, verbose=verbose)
        self.on_jet_tagprobe = TriggerDijetTagAndProbe(on_jet_tag_min_pt if on_jet_tag_probe else None, 
                                                       max_alpha=on_jet_max_alpha, swap=True, name=on_jet_label)
        
        # delta R matching
        self.deltaR_matching = DeltaRMatching(max_deltaR=max_deltaR)
        
        # select only n leading jets to fill histograms
        self.max_leading_jet = MaxLeadingObject(max_leading_jet, name="jet")
        
        # histograms
        self.is_data = is_data
        self.storage = hist.storage.Weight() #storage
        self.mix_correction_level = mix_correction_level
        pt_axis_dict = {
                        "log": 
                            lambda num_bins=50, name="pt", label=r"$p_T$": 
                                hist.axis.Regular(num_bins, 1, 10000, transform=hist.axis.transform.log, 
                                                  name=name, label=label),
                        "fine":
                            lambda num_bins=None, name="pt", label=r"$p_T$":
                                hist.axis.Variable(
                                    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 23, 27, 30, 35, 40, 
                                              45, 57, 72, 90, 120, 150, 200, 300, 400, 550, 750, 1000, 1500, 2000, 2500, 3000, 
                                              3500, 4000, 4500, 5000, 10000]), 
                                    name=name, label=label),
                        "coarse":
                            lambda num_bins=None, name="pt", label=r"$p_T$":
                                hist.axis.Variable(
                                    np.array([8, 10, 12, 15, 18, 21, 24, 28, 32, 37, 43, 49, 56, 64, 74, 84, 
                                              97, 114, 133, 153, 174, 196, 220, 245, 272, 300, 362, 430,
                                              507, 592, 686, 790, 905, 1032, 1172, 1327, 1497, 1684, 1890,
                                              #1999, 2000, 2238, 2500, 2787, 3103, 3450,
                                              2116, 2366, 2640, 2941, 3273, 3637, 4037, 4477, 4961, 5492, 6076, 7000]),
                                    name=name, label=label),
                        "linear":
                            lambda num_bins=50, name="jet_pt", label=r"$p_T^{jet}$":
                                hist.axis.Regular(num_bins, 0, 10000, name=name, label=label)
                        }
        assert pt_binning in pt_axis_dict, "Unrecognized pt_binning: {}".format(pt_binning)
        self.pt_binning = pt_binning
        self.get_pt_axis = lambda pt_binning, num_bins=50, name="jet_pt", label=r"$p_T^{jet}$": \
                               pt_axis_dict[pt_binning](num_bins, name, label) # syntactic sugar
        
        eta_axis_dict = {
                         "fine": 
                            lambda name="eta", label=r"$\eta$":
                                hist.axis.Variable(
                                    np.array([-5.191, -4.889,  -4.716,  -4.538,  -4.363,  -4.191,  -4.013,  -3.839,  -3.664,  
                                              -3.489, -3.314,  -3.139,  -2.964,  -2.853,  -2.65,  -2.5,  -2.322,  -2.172,  
                                              -2.043,  -1.93,  -1.83, -1.74,  -1.653,  -1.566,  -1.479,  -1.392,  -1.305,  
                                              -1.218,  -1.131,  -1.044,  -0.957,  -0.879, -0.783,  -0.696,  -0.609,  -0.522,
                                              -0.435,  -0.348,  -0.261,  -0.174,  -0.087,  0,  0.087,  0.174, 0.261,  0.348,
                                              0.435,  0.522,  0.609,  0.696,  0.783,  0.879,  0.957,  1.044,  1.131,  1.218, 
                                              1.305,  1.392,  1.479,  1.566,  1.653,  1.74,  1.83,  1.93,  2.043,  2.172, 
                                              2.322,  2.5,  2.65, 2.853,  2.964,  3.139,  3.314,  3.489, 3.664, 3.839, 4.013, 
                                              4.191,  4.363,  4.538,  4.716,  4.889, 5.191]), 
                                    name=name, label=label),
                         "coarse":
                            lambda name="eta", label=r"$\eta$":
                                hist.axis.Variable(
                                    np.array([-5.0, -3.0, -2.5, -1.3, 0.0, 1.3, 2.5, 3.0, 5.0]), 
                                    name=name, label=label)
                        }
        assert eta_binning in eta_axis_dict, "Unrecognized eta_binning: {}".format(eta_binning)
        self.eta_binning = eta_binning
        self.get_eta_axis = lambda eta_binning, num_bins=50, name="jet_eta", label=r"$\eta^{jet}$": \
                               eta_axis_dict[eta_binning](name, label) # syntactic sugar
        self.fill_gen = fill_gen
        if isinstance(hist_to_fill, str):
            hist_to_fill = {hist_to_fill}
        else:
            hist_to_fill = set(hist_to_fill)
        if "2d" in hist_to_fill or "all" in hist_to_fill:
            hist_to_fill.update(["pt_response", "pt_percent_difference", "comparison"])
            if off_jet_tag_probe or on_jet_tag_probe:
                hist_to_fill.update(["tag_and_probe"])
        if "1d" in hist_to_fill or "all" in hist_to_fill:
            hist_to_fill.update(["jet_pt", "jet_eta", "jet_phi"])
        self.hist_to_fill = hist_to_fill
        
        # printing
        self.verbose = verbose
            
    def process(self, events):
        # bookkeeping for dataset's name
        dataset = events.metadata.get("dataset", "untitled")
        
        # check consistency between is_data and input data
        has_gen = ("GenJet" in events.fields)
        if self.is_data and has_gen: # say data, but has gen
            #if verbose:
            #    warnings.warn("Processor set to process data, but contain gen information. Omit gen histograms.")
            raise ValueError("Processor set to process data, but contain gen information.")
        elif (not self.is_data) and (not has_gen): # say MC, does not have gen
            #if verbose:
            #    warnings.warn("Processor set to process MC, but does not contain gen information. Treat as data.")
            #self.is_data = False
            raise ValueError("Processor set to process MC, but does not contain gen information.")
        
        time_pf = defaultdict(float)
        last_time = time.time()
        # TrigObjJMEAK{4, 8} are not sorted by pt...
        for physics_object_name in ["TrigObjJMEAK4", "TrigObjJMEAK8", "ScoutingJet", "ScoutingFatJet"]:
            # is_sorted = lambda arr: np.all(arr[:-1] <= arr[1:]) # TODO
            if physics_object_name in [self.off_jet_name, self.on_jet_name]:
                if self.verbose > 1:
                    print("sorting: {}".format(physics_object_name))
                sort_index = ak.argsort(events[physics_object_name].pt, ascending=False)
                #print(sort_index)
                #is_already_sorted = all(sort_index[i] > sort_index[i+1] for i in range(len(sort_index) - 1))
#                 if is_already_sorted:
#                     print("{} is already sorted by pt!".format(physics_object_name))
#                 else:
                events[physics_object_name] = (events[physics_object_name])[sort_index]
        elapsed_time = time.time() - last_time
        time_pf["sorting"] += elapsed_time
        last_time = time.time()
        
        # define cutflow
        cutflow = defaultdict(int)
        cutflow["all events"] += len(events)
        
        # apply lumimask
        if self.lumimask:
            events = events[self.lumimask(events.run, events.luminosityBlock)]
            cutflow["lumimask"] += len(events)
        # save processed lumi list
        # lumi_list = list(set(zip(events.run, events.luminosityBlock)))
        if self.is_data and self.save_processed_lumi:
            lumi_list = LumiAccumulator(events.run, events.luminosityBlock, auto_unique=True)
        
        # events-level selections
        # good event cuts
        events = self.min_npvgood(events, cutflow)
        events = self.max_pv_z(events, cutflow)
        events = self.max_pv_rxy(events, cutflow)
        
        # Flag filters
        events = self.flag_filters(events, cutflow)
        
        # MET cuts
        events = self.max_MET(events, cutflow)
        events = self.max_MET_sumET(events, cutflow)
        
        # trigger cut
        events = self.min_trigger(events, cutflow)
        
        # event-level wrapped jet-level selectors
        # jet identification
        events = self.off_jet_Id(events, cutflow)
        events = self.on_jet_Id(events, cutflow)
        # jet veto map
        events = self.off_jet_veto_map(events, cutflow)
        events = self.on_jet_veto_map(events, cutflow)
        
        # minimum numbers of jets
        events = self.min_off_jet(events, cutflow)
        events = self.min_on_jet(events, cutflow)
        
        # jet-level selections
        # retrive offline and online jets
        on_jets = events[self.on_jet_name]
        off_jets = events[self.off_jet_name]
        
        # apply to offline jets
        if self.verbose > 1:
            print("processing offline jets")
        off_jets, off_correction_level_in_use = self.off_jet_JEC(off_jets, events, cutflow)
        if self.off_jet_tagprobe.status and len(off_jets) > 0: # prevent error when indexing 0 or 1
            off_jets_tag, off_jets = self.off_jet_tagprobe(off_jets[:, 0], off_jets[:, 1], off_jets, cutflow)
        
        # apply to online jets
        if self.verbose > 1:
            print("processing online jets")
        on_jets, on_correction_level_in_use = self.on_jet_JEC(on_jets, events, cutflow)
        if self.on_jet_tagprobe.status and len(on_jets) > 0:
            on_jets_tag, on_jets = self.on_jet_tagprobe(on_jets[:, 0], on_jets[:, 1], on_jets, cutflow)
        
        # delta R matching
        matched_off_jets, matched_on_jets = self.deltaR_matching(off_jets, on_jets, cutflow)
        
        # select n leading jets to plot
        matched_off_jets = self.max_leading_jet(matched_off_jets)
        matched_on_jets = self.max_leading_jet(matched_on_jets)
        
        elapsed_time = time.time() - last_time
        time_pf["selection"] += elapsed_time
        last_time = time.time()
        
        # check before filling histogram
        assert len(matched_on_jets) == len(matched_off_jets), "online and offline must have the same length for histogram filling, but get online: {} and offline: {}".format(len(matched_on_jets), len(matched_off_jets))
        
        # out accumulator
        out = {"cutflow": {dataset: cutflow}} # {dataset: cutflow}
        # save luminosity
        if self.is_data and self.save_processed_lumi:
            out["processed_lumi"] = {dataset: {"lumi_list": lumi_list}}
            
        # define axes for output histograms
        # bookkeeping axes
        dataset_axis = hist.axis.StrCategory([], name="dataset", label="Primary Dataset", growth=True)
        
        # correction levels to fill
        correction_level_suffix_dict = {"raw":"Raw", "orig":"Original", "jec":"Corrected"}
        # list of (off_correction_level, on_correction_level)
        # this makes it possible to opt-out correction levels by removing from correction_level_suffix_dict
        off_correction_level_names = [_ for _ in correction_level_suffix_dict.keys() if _ in off_correction_level_in_use]
        on_correction_level_names = [_ for _ in correction_level_suffix_dict.keys() if _ in on_correction_level_in_use]
        if not self.mix_correction_level:
            correction_level_names = [(_, _) for _ in correction_level_suffix_dict.keys() \
                                      if _ in off_correction_level_in_use and _ in on_correction_level_in_use]
        else:
            correction_level_names = itertools.product(off_correction_level_names, on_correction_level_names)
            
        correction_level_dict = { # dict comprehension
                                 (off_name, on_name):
                                     "off={}:on={}"\
                                        .format(correction_level_suffix_dict[off_name], correction_level_suffix_dict[on_name])
                                 for off_name, on_name in correction_level_names
                                }
        
        correction_level_axis = hist.axis.StrCategory(list(correction_level_dict.values()), \
                                                      name="correction_level", label=r"Correction levels", growth=False)
        
        # jet_type_axis is used to specify which jet is a control, e.g. used for x-axis and selecting eta range
        jet_types = [self.off_jet_label, self.on_jet_label]
        if (not self.is_data) and self.fill_gen:
            try:
                matched_off_genjets = matched_off_jets.matched_gen # try finding genjets
                jet_types.append(self.off_jet_label+"_Gen")
            except:
                if self.verbose > 0:
                    warnings.warn("Fail to retrieve matched gen for offline")
            try:
                matched_on_genjets = matched_on_jets.matched_gen # try finding genjets
                jet_types.append(self.on_jet_label+"_Gen")
            except:
                if self.verbose > 0:
                    warnings.warn("Fail to retrieve matched gen for online")
        jet_type_axis = hist.axis.StrCategory(jet_types, name="jet_type", label="Types of Jet", growth=False)
        
        # original variable axes
        jet_pt_axis = self.get_pt_axis(self.pt_binning, num_bins=50)
        jet_eta_axis = self.get_eta_axis(self.eta_binning, num_bins=50)
        jet_phi_axis = hist.axis.Regular(50, -np.pi, np.pi, circular=True, name="jet_phi", label=r"$\phi^{jet}$")
        
        # derived variable axes and build histograms
        if "pt_response" in self.hist_to_fill:
            pt_response_axis = hist.axis.Regular(200, 0, 5, name="pt_response", label=r"$p_T$ response")
            h_pt_response = hist.Hist(dataset_axis, correction_level_axis, jet_type_axis, 
                                      jet_pt_axis, jet_eta_axis, jet_phi_axis,
                                      pt_response_axis, storage=self.storage,
                                      name="pt_response", label=r"$p_T$ response")
            out["pt_response"] = h_pt_response
            
        if "pt_percent_difference" in self.hist_to_fill:
            pt_percent_diff_axis = hist.axis.Regular(200, -2, 2, name="pt_percent_diff", label=r"$p_T$ percentage difference")
            h_pt_percent_diff = hist.Hist(dataset_axis, correction_level_axis, jet_type_axis,
                                          jet_pt_axis, jet_eta_axis, jet_phi_axis,
                                          pt_percent_diff_axis, storage=self.storage,
                                          name="pt_percent_diff", label=r"$p_T$ percentage difference")
            out["pt_percent_difference"] = h_pt_percent_diff
        
        # 2d correlation histogram
        if "comparison" in self.hist_to_fill:
            on_jet_pt_axis = self.get_pt_axis(self.pt_binning, num_bins=100,
                                              name="on_jet_pt", label=r"$p_T^{%s}$"%self.on_jet_label)
            off_jet_pt_axis = self.get_pt_axis(self.pt_binning, num_bins=100,
                                               name="off_jet_pt", label=r"$p_T^{%s}$"%self.off_jet_label)
            out["comparison"] = hist.Hist(dataset_axis, correction_level_axis, jet_type_axis, jet_eta_axis, jet_phi_axis,
                                          off_jet_pt_axis, on_jet_pt_axis, storage=self.storage,
                                          name="comparison", label="Online vs Offline")
        # tag and probe histogram    
        if "tag_and_probe" in self.hist_to_fill and (self.off_jet_tagprobe.status or self.on_jet_tagprobe.status):
            tp_jet_types = list()
            if self.off_jet_tagprobe.status:
                tp_jet_types += [self.off_jet_label]
            if self.on_jet_tagprobe.status:
                tp_jet_types += [self.on_jet_label]
            tp_jet_type_axis = hist.axis.StrCategory(tp_jet_types, name="jet_type", label="Types of Jet", growth=False)
            tp_diff_ratio_axis = hist.axis.Regular(200, -2, 2, name="tp_diff_ratio", 
                                                   label=r"Tag and Probe $p_T$ difference ratio")
            h_tp = hist.Hist(dataset_axis, tp_jet_type_axis, jet_pt_axis, jet_eta_axis, jet_phi_axis, 
                             tp_diff_ratio_axis, storage=self.storage,
                             name="tag_and_probe", label="Tag and Probe")
            out["tag_and_probe"] = h_tp
        
        # 1D histograms (technically, these are reducible from above histograms)
        if "jet_pt" in self.hist_to_fill:
            corrected_jet_types = [self.off_jet_label+"_"+correction_level_suffix_dict[_] for _ in off_correction_level_names]
            corrected_jet_types += [self.on_jet_label+"_"+correction_level_suffix_dict[_] for _ in on_correction_level_names]
            corrected_jet_type_axis = hist.axis.StrCategory(corrected_jet_types, 
                                                            name="jet_type", label="Types of Jet", growth=False)
            h_jet_pt = hist.Hist(dataset_axis, corrected_jet_type_axis, jet_pt_axis, storage=self.storage)
            out["jet_pt"] = h_jet_pt
        if "jet_eta" in self.hist_to_fill:
            h_jet_eta = hist.Hist(dataset_axis, jet_type_axis, jet_eta_axis, storage=self.storage)
            out["jet_eta"] = h_jet_eta
        if "jet_phi" in self.hist_to_fill:
            h_jet_phi = hist.Hist(dataset_axis, jet_type_axis, jet_phi_axis, storage=self.storage)
            out["jet_phi"] = h_jet_phi
                
        elapsed_time = time.time() - last_time
        time_pf["creating histograms"] += elapsed_time
        last_time = time.time()
        
        # filling histograms
        if self.verbose > 1:
            print("filling histogram: linear")

        if len(matched_off_jets) == 0: # ak.flatten has axis=1 as default and this can raise error with 0 length
            # not so sure how to really handle this
            # here, we just skip filling to speed up
            if self.verbose > 1:
                print("no events to fill histograms")
            out["time_pf"] = time_pf
            return out
        
        # get event weight
        if self.is_data:
            weight = None
        else:
            weight = ak.flatten(ak.broadcast_arrays(events.genWeight, matched_off_jets.pt)[0])
        
        # loop correction levels to fill histograms
        for (off_correction_level_name, on_correction_level_name), correction_level_label in correction_level_dict.items():
            start_time = time.time()
            if "pt_response" in out:
                pt_response = matched_on_jets["pt_"+on_correction_level_name] / \
                              matched_off_jets["pt_"+off_correction_level_name]
                # filling offline as x axis
                out["pt_response"].fill(dataset=dataset, correction_level=correction_level_label, 
                                        jet_type=self.off_jet_label,
                                        jet_pt=ak.flatten(matched_off_jets["pt_"+off_correction_level_name]), \
                                        jet_eta=ak.flatten(matched_off_jets.eta), \
                                        jet_phi=ak.flatten(matched_off_jets.phi), \
                                        pt_response=ak.flatten(pt_response),
                                        weight=weight)
                    
                # filling online as x axis
                out["pt_response"].fill(dataset=dataset, correction_level=correction_level_label, \
                                        jet_type=self.on_jet_label, \
                                        jet_pt=ak.flatten(matched_on_jets["pt_"+on_correction_level_name]), \
                                        jet_eta=ak.flatten(matched_on_jets.eta), \
                                        jet_phi=ak.flatten(matched_on_jets.phi), \
                                        pt_response=(1 / ak.flatten(pt_response)),
                                        weight=weight)
                
                # optionally, fill gen as x axis
                if (not self.is_data) and self.fill_gen:
                    try:
                        matched_off_genjets = matched_off_jets.matched_gen
                        out["pt_response"].fill(dataset=dataset, correction_level=correction_level_label, \
                                           jet_type=self.off_jet_label + "_Gen", \
                                           jet_pt=ak.to_numpy(ak.flatten(matched_off_genjets.pt), allow_missing=True), \
                                           jet_eta=ak.to_numpy(ak.flatten(matched_off_genjets.eta), allow_missing=True), \
                                           jet_phi=ak.to_numpy(ak.flatten(matched_off_genjets.phi), allow_missing=True), \
                                           pt_response=ak.flatten(pt_response),
                                           weight=weight)
                    except:   
                        if self.verbose > 0:
                            warnings.warn("Fail to retrieve matched gen for offline")
                    try:
                        matched_on_genjets = matched_on_jets.matched_gen
                        out["pt_response"].fill(dataset=dataset, correction_level=correction_level_label, \
                                                jet_type=self.on_jet_label + "_Gen", \
                                                jet_pt=ak.to_numpy(ak.flatten(matched_on_genjets.pt), allow_missing=True), \
                                                jet_eta=ak.to_numpy(ak.flatten(matched_on_genjets.eta), allow_missing=True), \
                                                jet_phi=ak.to_numpy(ak.flatten(matched_on_genjets.phi), allow_missing=True), \
                                                pt_response=(1 / ak.flatten(pt_response)),
                                                weight=weight)

                    except:
                        if self.verbose > 0:
                            warnings.warn("Fail to retrieve matched gen for online")
            end_time = time.time()
            #print("respone: ", end_time - start_time)
            
            if "pt_percent_difference" in out:
                sum_pt = matched_off_jets["pt_"+off_correction_level_name] + matched_on_jets["pt_"+on_correction_level_name]
                average_pt = 0.5 * sum_pt
                diff_pt = matched_on_jets["pt_"+on_correction_level_name] - matched_off_jets["pt_"+off_correction_level_name]
                percent_diff = diff_pt / average_pt
                
                # filling offline as x axis
                out["pt_percent_difference"].fill(dataset=dataset, correction_level=correction_level_label, 
                                               jet_type=self.off_jet_label, \
                                               jet_pt=ak.flatten(average_pt), \
                                               jet_eta=ak.flatten(matched_off_jets.eta), \
                                               jet_phi=ak.flatten(matched_off_jets.phi), \
                                               pt_percent_diff=ak.flatten(percent_diff),
                                               weight=weight)
                # filling online as x axis
                out["pt_percent_difference"].fill(dataset=dataset, correction_level=correction_level_label, 
                                               jet_type=self.on_jet_label, \
                                               jet_pt=ak.flatten(average_pt), \
                                               jet_eta=ak.flatten(matched_on_jets.eta), \
                                               jet_phi=ak.flatten(matched_on_jets.phi), \
                                               pt_percent_diff= -1*ak.flatten(percent_diff),
                                               weight=weight)
                
                # optionally, fill gen as x axis
                if (not self.is_data) and self.fill_gen:
                    try:
                        matched_off_genjets = matched_off_jets.matched_gen
                        out["pt_percent_difference"].fill(dataset=dataset, correction_level=correction_level_label, \
                                           jet_type=self.off_jet_label + "_Gen", \
                                           jet_pt=ak.to_numpy(ak.flatten(matched_off_genjets.pt), allow_missing=True), \
                                           jet_eta=ak.to_numpy(ak.flatten(matched_off_genjets.eta), allow_missing=True), \
                                           jet_phi=ak.to_numpy(ak.flatten(matched_off_genjets.phi), allow_missing=True), \
                                           pt_percent_diff=ak.flatten(percent_diff_py),
                                                         weight=weight)
                    except:   
                        if self.verbose > 0:
                            warnings.warn("Fail to retrieve matched gen for offline")
                    try:
                        matched_on_genjets = matched_on_jets.matched_gen
                        out["pt_percent_difference"].fill(dataset=dataset, correction_level=correction_level_label, \
                                           jet_type=self.on_jet_label + "_Gen", \
                                           jet_pt=ak.to_numpy(ak.flatten(matched_on_genjets.pt), allow_missing=True), \
                                           jet_eta=ak.to_numpy(ak.flatten(matched_on_genjets.eta), allow_missing=True), \
                                           jet_phi=ak.to_numpy(ak.flatten(matched_on_genjets.phi), allow_missing=True), \
                                           pt_percent_diff= -1*ak.flatten(percent_diff),
                                                         weight=weight)

                    except:
                        if self.verbose > 0:
                            warnings.warn("Fail to retrieve matched gen for online")
            
            # comparison histogram
            start_time = time.time()
            if "comparison" in out:  
                out["comparison"].fill(dataset=dataset, correction_level=correction_level_label, 
                                       jet_type=self.off_jet_label,\
                                       jet_eta=ak.flatten(matched_off_jets.eta), \
                                       jet_phi=ak.flatten(matched_off_jets.phi), \
                                       off_jet_pt=ak.flatten(matched_off_jets["pt_"+off_correction_level_name]), \
                                       on_jet_pt=ak.flatten(matched_on_jets["pt_"+on_correction_level_name]),
                                       weight=weight)
                
                if (not self.is_data) and self.fill_gen:
                    try:
                        matched_off_genjets = matched_off_jets.matched_gen
                        out["comparison"].fill(dataset=dataset, correction_level=correction_level_label, \
                                               jet_type=self.off_jet_label + "_Gen", \
                                               jet_eta=ak.to_numpy(ak.flatten(matched_off_genjets.eta), allow_missing=True),
                                               jet_phi=ak.to_numpy(ak.flatten(matched_off_genjets.phi), allow_missing=True),     
                                               off_jet_pt=ak.to_numpy(ak.flatten(matched_off_genjets.pt), allow_missing=True),
                                               on_jet_pt=ak.flatten(matched_off_jets["pt_"+off_correction_level_name]),
                                               weight=weight)
                    except:   
                        if self.verbose > 0:
                            warnings.warn("Fail to retrieve matched gen for offline")
                            
                    try:
                        matched_on_genjets = matched_on_jets.matched_gen
                        out["comparison"].fill(dataset=dataset, correction_level=correction_level_label, \
                                               jet_type=self.on_jet_label + "_Gen", \
                                               jet_eta=ak.to_numpy(ak.flatten(matched_on_genjets.eta), allow_missing=True),
                                               jet_phi=ak.to_numpy(ak.flatten(matched_on_genjets.phi), allow_missing=True),     
                                               off_jet_pt=ak.to_numpy(ak.flatten(matched_on_genjets.pt), allow_missing=True),
                                               on_jet_pt=ak.flatten(matched_on_jets["pt_"+off_correction_level_name]),
                                               weight=weight)
                    except:   
                        if self.verbose > 0:
                            warnings.warn("Fail to retrieve matched gen for online")
            end_time = time.time()
            #print("comp: ", end_time - start_time)
            
        # tag and probe histogram
        # NB: these are unmatched (before deltaR matching)
        if "tag_and_probe" in out:
            if self.off_jet_tagprobe.status:
                if self.is_data:
                    off_tp_weight = None
                else:
                    off_tp_weight = ak.flatten(ak.broadcast_arrays(events.genWeight, off_jets.pt)[0])
                out["tag_and_probe"].fill(dataset=dataset, jet_type=self.off_jet_label, \
                                          jet_pt=ak.flatten(off_jets_tag.pt), \
                                          jet_eta=ak.flatten(off_jets_tag.eta), \
                                          jet_phi=ak.flatten(off_jets_tag.phi), \
                                          tp_diff_ratio=ak.flatten((off_jets.pt - off_jets_tag.pt) / off_jets_tag.pt),
                                          weight=off_tp_weight)
            if self.on_jet_tagprobe.status:
                if self.is_data:
                    on_tp_weight = None
                else:
                    on_tp_weight = ak.flatten(ak.broadcast_arrays(events.genWeight, on_jets.pt)[0])
                out["tag_and_probe"].fill(dataset=dataset, jet_type=self.on_jet_label, \
                                          jet_pt=ak.flatten(on_jets_tag.pt), \
                                          jet_eta=ak.flatten(on_jets_tag.eta), \
                                          jet_phi=ak.flatten(on_jets_tag.phi), \
                                          tp_diff_ratio=ak.flatten((on_jets.pt - on_jets_tag.pt) / on_jets_tag.pt),
                                          weight=on_tp_weight)
        
        # filling 1D histograms
        if "jet_pt" in out:
            for off_correction_level_name in off_correction_level_names:
                out["jet_pt"].fill(dataset=dataset, 
                                   jet_type=self.off_jet_label+"_"+correction_level_suffix_dict[off_correction_level_name], 
                                   jet_pt=ak.flatten(matched_off_jets["pt_"+off_correction_level_name]),
                                   weight=weight)
            for on_correction_level_name in on_correction_level_names:
                out["jet_pt"].fill(dataset=dataset, 
                                   jet_type=self.on_jet_label+"_"+correction_level_suffix_dict[on_correction_level_name], 
                                   jet_pt=ak.flatten(matched_on_jets["pt_"+on_correction_level_name]),
                                   weight=weight)
            # if (not self.is_data) and self.fill_gen: let's deal with this later

        if "jet_eta" in out:
            out["jet_eta"].fill(dataset=dataset, jet_type=self.off_jet_label, jet_eta=ak.flatten(matched_off_jets.eta),
                                weight=weight)
            out["jet_eta"].fill(dataset=dataset, jet_type=self.on_jet_label, jet_eta=ak.flatten(matched_on_jets.eta),
                                weight=weight)
            # if (not self.is_data) and self.fill_gen: let's deal with this later
        
        if "jet_phi" in out:
            out["jet_phi"].fill(dataset=dataset, jet_type=self.off_jet_label, jet_phi=ak.flatten(matched_off_jets.phi),
                                weight=weight)
            out["jet_phi"].fill(dataset=dataset, jet_type=self.on_jet_label, jet_phi=ak.flatten(matched_on_jets.phi),
                                weight=weight)
            # if (not self.is_data) and self.fill_gen: let's deal with this later
        
        elapsed_time = time.time() - last_time
        time_pf["filling histograms"] += elapsed_time
        last_time = time.time()
        out["time_pf"] = time_pf
        
        return out
        
    def postprocess(self, accumulator):
        # compute integrated luminosity
        # lumidata might fail, so it is recommended to compute integrated lumi later
#         if not self.is_data:
#             return accumulator
#         if self.compute_processed_lumi:
#             lumidata = LumiData(self.lumi_csv_path)
#             for dataset in accumulator["processed_lumi"]:
#                 accumulator["processed_lumi"][dataset]["lumi_list"].unique() # apply unique
#                 accumulator["processed_lumi"][dataset]["lumi"] = lumidata.get_lumi(accumulator["processed_lumi"][dataset]["lumi_list"])
#         else:
#             for dataset in accumulator["processed_lumi"]:
#                 accumulator["processed_lumi"][dataset]["lumi_list"].unique() # apply unique
#                 accumulator["processed_lumi"][dataset]["lumi"] = None
            
#         if self.lumi_csv_path:
#             lumidata = LumiData(self.lumi_csv_path)
#             for dataset in accumulator["processed_lumi"]:
#                 if len(accumulator["processed_lumi"][dataset]["lumi_list"]) == 0:
#                     if self.verbose > 0:
#                         warnings.warn("no lumi blocks are processed for dataset: {}!".format(dataset))
#                 # apply unique
#                 lumi_list = np.array(accumulator["processed_lumi"][dataset]["lumi_list"])
#                 lumi_list = LumiList(lumi_list[:, 0], lumi_list[:, 1]) if len(lumi_list) > 0 else LumiList()
#                 accumulator["processed_lumi"][dataset]["lumi_list"] = lumi_list
#                 # compute integrated luminosity
#                 accumulator["processed_lumi"][dataset]["lumi"] = lumidata.get_lumi(lumi_list)
#         else:
#             for dataset in accumulator["processed_lumi"]:
#                 if len(accumulator["processed_lumi"][dataset]["lumi_list"]) == 0:
#                     if self.verbose > 0:
#                         warnings.warn("no lumi blocks are processed for dataset: {}!".format(dataset))
#                 # apply unique
#                 lumi_list = np.array(accumulator["processed_lumi"][dataset]["lumi_list"])
#                 lumi_list = LumiList(lumi_list[:, 0], lumi_list[:, 1]) if len(lumi_list) > 0 else LumiList()
#                 accumulator["processed_lumi"][dataset]["lumi_list"] = lumi_list

#                 accumulator["processed_lumi"][dataset]["lumi"] = None
                
        return accumulator