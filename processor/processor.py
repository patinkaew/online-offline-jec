import numpy as np
import awkward as ak
import hist

from coffea.processor import ProcessorABC
from coffea.lumi_tools import LumiMask
from coffea.analysis_tools import Weights
from coffea.nanoevents.methods import vector

from processor.selector import *
from processor.accumulator import LumiAccumulator

from collections import OrderedDict#, Callable, defaultdict
import warnings
import json

class OrderedDictWithDefaultInt(OrderedDict):
    #Source: https://stackoverflow.com/a/42404907
    def __missing__(self, key):
        value = 0
        self[key] = value
        return value
    def __repr__(self):
        return json.dumps(dict(self), indent=4, sort_keys=False, default=str)

# class SimpleProcessor(processor.ProcessorABC):
#     def __init__(self):
#         pass
#     def process(self, events):
#         print("nevents: ", len(events))
#         return {"nevents":len(events)}
#     def postprocess(self, accumulator):
#         return accumulator

class OnlineOfflineProcessor(ProcessorABC):
    def __init__(self, 
                 off_jet_name, off_jet_label=None, # offline jet
                 on_jet_name="TrigObjJMEAK4", on_jet_label=None, # online jet
                 lumi_json_path=None, # path to gloden json file (good run certification)
                 lumi_csv_path=None, # path to output lumi csv from brilcalc
                 save_processed_lumi=True, # save processed lumi section (only data)
                 
                 # event-level selections
                 flag_filters=None,  # event_level, apply flag filters, e.g. METfilters
                 off_PV="PV", on_PV=None, # event_level, match PV from two reconstruction types
                 pthatmax_less_than_binvar=True, # event-level, pthat_max < pthat to remove hotspots in MC
                 min_off_jet=1, min_on_jet=1, # event-level, select events with at least n jets
                 MET_cut_MET_type="MET", MET_cut_max_MET=None, # event-level, select max MET and/or max MET/sumET
                 MET_cut_max_MET_sumET=None, MET_cut_min_MET=45, 
                 trigger_min_pt=0, # event-level, trigger cut
                 trigger_type=None, trigger_flag_prefix="PFJet", trigger_all_pts=None,
                 
                 # jet-level selections
                 off_jet_min_pt=0, on_jet_min_pt=0, # jet-level, min pt
                 off_jet_id=None, on_jet_id=None, # jet-level, jet id cut
                 off_jet_type="PUPPI", on_jet_type="CHS", # needed for jet id
                 off_jet_veto_map_path=None, on_jet_veto_map_path=None, # jet-level, jet veto map
                 off_jet_weight_filelist=None, on_jet_weight_filelist=None, # weight file for JEC
                 off_rho_name=None, on_rho_name=None, # rho to use in JEC
                 tag_probe_tag_min_pt=0,
                 tag_probe_max_alpha=1.0,
                 tag_probe_opposite_on_jet=False,
                 tag_probe_on_off_ordering=1,
                 tag_probe_max_deltaR=0.2,
                 tag_probe_third_jet_max_pt=30,
                 tag_probe_match_tag=False,
                 off_MET_name="PuppiMET", # MET for MPF calculation
                 on_MET_name=None,
                 
                 max_deltaR=0.2, # for deltaR matching
                 max_leading_jet=2, # select up to n leading jets to fill histograms
                 same_eta_bin=None, # both jets must be in the same eta bin
                 
                 # histograms
                 is_data=True,
                 mix_correction_level=False,
                 pt_binning="log",
                 eta_binning="coarse",
                 fill_gen=False, # (only MC)
                 
                 # gen-jet selection if fill_gen is True
                 gen_jet_name=None,
                 gen_jet_label="Gen",
                 gen_jet_veto_map_path=None,
                 
                 use_weight=True, # (only MC)
                 hist_to_fill="all",
                 
                 verbose=0):
        
        use_tag_probe = tag_probe_tag_min_pt is not None and tag_probe_tag_min_pt >=0
        assert not (use_tag_probe and fill_gen), "Tag and Probe with Gen Jet matching not supported yet!"
        
        fill_gen = fill_gen if (not is_data) else False #"fill_gen can be only used with MC"
        
        # which online and offline to use
        # name is used to retrieve from NanoAOD
        # label is used for histograms + plots
        self.off_jet_name = off_jet_name
        off_jet_label = off_jet_label if off_jet_label != None else off_jet_name
        self.off_jet_label = off_jet_label
        self.on_jet_name = on_jet_name
        on_jet_label = on_jet_label if on_jet_label != None else on_jet_name
        self.on_jet_label = on_jet_label
        if (not is_data) and fill_gen: # if fill gen
            assert gen_jet_name != None, "Must provide gen jet name"
            self.gen_jet_name = gen_jet_name
            self.gen_jet_label = gen_jet_label if gen_jet_label else gen_jet_name
        
        # luminosity
        self.lumimask = LumiMaskSelector(lumi_json_path)
        self.save_processed_lumi = save_processed_lumi

        # processing pipeline
        # event-level selections
        # good event cuts
        self.min_npvgood = MinNPVGood(min_NPVGood=0)
        self.max_pv_z = MaxPV_z(max_PV_z=24)
        self.max_pv_rxy = MaxPV_rxy(max_PV_rxy=2)
        
        # flag_filters
        self.flag_filters = FlagFilters(flag_filters)
        
        # main PV matching
        self.close_pv_z = ClosePV_z(off_PV, on_PV, sigma_multiple=5) # max_dz=0.2
        
        # pthatmax < pthat
        self.pthatmax_less_than_binvar = PileupPthatmaxLessThanGeneratorBinvar((not is_data) and pthatmax_less_than_binvar)
        
        # minimum number of jets
        # if tag and probe will be applied, need at least 2
        min_off_jet = min_off_jet #if not (off_jet_tag_probe or use_tag_probe) else max(min_off_jet, 2) 
        min_on_jet = min_on_jet #if not (on_jet_tag_probe or use_tag_probe) else max(min_on_jet, 2) 
        self.min_off_jet = MinPhysicsObject(off_jet_name, min_off_jet)
        self.min_on_jet = MinPhysicsObject(on_jet_name, min_on_jet)
        
        # MET cut
        #self.max_MET = MaxMET(MET_cut_max_MET, MET_cut_MET_type)
        self.max_MET_sumET = MaxMET_sumET(MET_cut_max_MET_sumET, MET_cut_min_MET, MET_cut_MET_type)
        
        # trigger cut
        self.min_trigger = MinTrigger(trigger_type, trigger_min_pt, trigger_flag_prefix, trigger_all_pts)
        
        # event-level wrapped jet-level selectors
        # intuitively, these selectors act on jet
        # however, cut earlier is better for performance
        # and make it easier to tag and probe
        self.off_jet_Id = JetID(off_jet_name, off_jet_id, off_jet_type)
        self.on_jet_Id = JetID(on_jet_name, on_jet_id, on_jet_type)
        
        self.off_jet_veto_map = JetVetoMap(off_jet_name, off_jet_veto_map_path, map_type="jetvetomap")
        self.on_jet_veto_map = JetVetoMap(on_jet_name, on_jet_veto_map_path, map_type="jetvetomap")
        # fill_gen
        if (not is_data) and fill_gen:
            self.gen_jet_veto_map = JetVetoMap(gen_jet_name, gen_jet_veto_map_path, map_type="jetvetomap") 
        
        # jet-level selections
        # Jet Energy Correction
        self.off_jet_JEC = JetEnergyCorrector(off_jet_weight_filelist, jet_name=off_jet_name, 
                                              rho_name=off_rho_name, verbose=verbose)
        self.on_jet_JEC = JetEnergyCorrector(on_jet_weight_filelist, jet_name=on_jet_name,
                                             rho_name=on_rho_name, verbose=verbose)

        # minimum jets pt
        self.off_jet_min_pt = PhysicsObjectMinPt(off_jet_name, off_jet_min_pt)
        self.on_jet_min_pt = PhysicsObjectMinPt(on_jet_name, on_jet_min_pt)
        
        # tag and probe
        self.on_off_tagprobe = OnlineOfflineDijetTagAndProbe(off_jet_name, on_jet_name,
                                                            tag_min_pt=tag_probe_tag_min_pt,
                                                            max_alpha=tag_probe_max_alpha,
                                                            max_deltaR=tag_probe_max_deltaR,
                                                            match_tag=tag_probe_match_tag)
        
        # delta R matching
        if is_data or ((not is_data) and (not fill_gen)):
            self.deltaR_matching = DeltaRMatching(max_deltaR, off_jet_name, on_jet_name)
        else:
            self.deltaR_matching = PairwiseDeltaRMatching([off_jet_name, on_jet_name, gen_jet_name], max_deltaR=max_deltaR)
        
        # select only n leading jets to fill histograms
        self.max_leading_off_jet = MaxLeadingPhysicsObject(off_jet_name, max_leading_jet)
        self.max_leading_on_jet = MaxLeadingPhysicsObject(on_jet_name, max_leading_jet)
        if (not is_data) and fill_gen:
            self.max_leading_gen_jet = MaxLeadingPhysicsObject(gen_jet_name, max_leading_jet)
             
        # define fixed pt and eta bins
        pt_bin_dict = {"fine": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 23, 27, 30, 35, 40, 
                                45, 57, 72, 90, 120, 150, 200, 300, 400, 550, 750, 1000, 1500, 2000, 2500, 3000, 
                                3500, 4000, 4500, 5000, 10000],
                       "coarse": [8, 10, 12, 15, 18, 21, 24, 28, 32, 37, 43, 49, 56, 64, 74, 84, 
                                  97, 114, 133, 153, 174, 196, 220, 245, 272, 300, 362, 430,
                                  507, 592, 686, 790, 905, 1032, 1172, 1327, 1497, 1684, 1890,
                                  #1999, 2000, 2238, 2500, 2787, 3103, 3450,
                                  2116, 2366, 2640, 2941, 3273, 3637, 4037, 4477, 4961, 5492, 6076, 7000]}
        eta_bin_dict = {"fine": [-5.191, -4.889,  -4.716,  -4.538,  -4.363,  -4.191,  -4.013,  -3.839,  -3.664,  
                                 -3.489, -3.314,  -3.139,  -2.964,  -2.853,  -2.65,  -2.5,  -2.322,  -2.172,  
                                 -2.043,  -1.93,  -1.83, -1.74,  -1.653,  -1.566,  -1.479,  -1.392,  -1.305,  
                                 -1.218,  -1.131,  -1.044,  -0.957,  -0.879, -0.783,  -0.696,  -0.609,  -0.522,
                                 -0.435,  -0.348,  -0.261,  -0.174,  -0.087,  0,  0.087,  0.174, 0.261,  0.348,
                                 0.435,  0.522,  0.609,  0.696,  0.783,  0.879,  0.957,  1.044,  1.131,  1.218, 
                                 1.305,  1.392,  1.479,  1.566,  1.653,  1.74,  1.83,  1.93,  2.043,  2.172, 
                                 2.322,  2.5,  2.65, 2.853,  2.964,  3.139,  3.314,  3.489, 3.664, 3.839, 4.013, 
                                 4.191,  4.363,  4.538,  4.716,  4.889, 5.191],
                        "coarse": [-5.0, -3.0, -2.5, -1.3, 0.0, 1.3, 2.5, 3.0, 5.0]}
        
        assert same_eta_bin is None or isinstance(same_eta_bin, bool) or same_eta_bin in eta_bin_dict, \
            "Unrecognized same_eta_bin: {}".format(same_eta_bin)
        # default to same as eta_binning, set same eta bin to False to disable
        if same_eta_bin is None or same_eta_bin == True:
            same_eta_bin = eta_binning
        if is_data or ((not is_data) and (not fill_gen)):
            self.same_eta_bin = MultiPhysicsObjectSameEtaBin(eta_bin_dict[same_eta_bin] if same_eta_bin else None, \
                                                             [off_jet_name, on_jet_name])
        else:
            self.same_eta_bin = MultiPhysicsObjectSameEtaBin(eta_bin_dict[same_eta_bin] if same_eta_bin else None, \
                                                             [off_jet_name, on_jet_name, gen_jet_name])

        
        # histograms
        self.is_data = is_data
        self.storage = hist.storage.Weight() #storage
        self.use_weight = use_weight
        self.mix_correction_level = mix_correction_level
        pt_axis_dict = {
                        "log": 
                            lambda num_bins=50, name="pt", label=r"$p_T$": 
                                hist.axis.Regular(num_bins, 1, 10000, transform=hist.axis.transform.log, 
                                                  name=name, label=label),
                        "fine":
                            lambda num_bins=None, name="pt", label=r"$p_T$":
                                hist.axis.Variable(pt_bin_dict["fine"], name=name, label=label),
                        "coarse":
                            lambda num_bins=None, name="pt", label=r"$p_T$":
                                hist.axis.Variable(pt_bin_dict["coarse"], name=name, label=label),
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
                            lambda num_bins=None, name="eta", label=r"$\eta$":
                                hist.axis.Variable(eta_bin_dict["fine"], name=name, label=label),
                         "coarse":
                            lambda num_bins=None, name="eta", label=r"$\eta$":
                                hist.axis.Variable(eta_bin_dict["coarse"], name=name, label=label),
                         "linear":
                            lambda num_bins=50, name="eta", label=r"$\eta$":
                                hist.axis.Regular(50, -5, 5, name=name, label=label),
                        }
        assert eta_binning in eta_axis_dict, "Unrecognized eta_binning: {}".format(eta_binning)
        self.eta_binning = eta_binning
        self.get_eta_axis = lambda eta_binning, num_bins=50, name="jet_eta", label=r"$\eta^{jet}$": \
                               eta_axis_dict[eta_binning](num_bins, name, label) # syntactic sugar
        self.fill_gen = fill_gen
        
        if isinstance(hist_to_fill, str):
            hist_to_fill = hist_to_fill.split()
        hist_to_fill = set(hist_to_fill)
        if "2d" in hist_to_fill or "all" in hist_to_fill:
            hist_to_fill.update(["response", "asymmetry", "comparison"])
        if "1d" in hist_to_fill or "all" in hist_to_fill:
            hist_to_fill.update(["jet_pt", "jet_eta", "jet_phi"])
        if "tp" in hist_to_fill or "all" in hist_to_fill:
            hist_to_fill.update(["tp_response", "tp_asymmetry", "tp_metprojection"])
        if "comp" in hist_to_fill:
            if fill_gen:
                hist_to_fill.update(["comparison", "gen_comparison"])
            elif use_tag_probe:
                hist_to_fill.update(["comparison", "tp_comparison"])
            else:
                hist_to_fill.update(["comparison"])
        if "control" in hist_to_fill:
            hist_to_fill.update(["met", "met_sumEt", "met_orig", "met_sumEt_orig", 
                                 "jet_pt_orig", "jet_eta_orig", "jet_phi_orig"])
        all_hists = {"response", "asymmetry", "comparison", 
                     "jet_pt", "jet_eta", "jet_phi",
                     "tp_response", "tp_asymmetry", "tp_metprojection",
                     "tp_comparison", "gen_comparison",
                     "met", "met_sumEt", "met_orig", "met_sumEt_orig", 
                     "jet_pt_orig", "jet_eta_orig", "jet_phi_orig"}
        hist_to_fill = hist_to_fill.intersection(all_hists)
        print("Number of histograms: ", len(hist_to_fill))
        print("List of histograms: ", " ".join(hist_to_fill))
        if len({"tp_response", "tp_asymmetry", "tp_metprojection", "tp_comparison"}.intersection(hist_to_fill)) > 0:
            assert tag_probe_tag_min_pt is not None and tag_probe_tag_min_pt >= 0, "Must enable tag and probe to fill tp_* histograms"
        if "tp_comparison" in hist_to_fill:
            assert tag_probe_match_tag, "Must set match tag to True to fill tp_comparison histogram"
        
        self.hist_to_fill = hist_to_fill
        
        if any([_.startswith("met_") for _ in self.hist_to_fill]) or "tp_metprojection" in self.hist_to_fill:
            assert off_MET_name is not None, "Must provide offline MET name to use in MPF"
            assert on_MET_name is not None, "Must provide online MET name to use in MPF"
            self.off_MET_name = off_MET_name
            self.on_MET_name = on_MET_name
            
        if "tp_metprojection" in self.hist_to_fill: # need original online jets to compute modified MET
            self.on_off_tagprobe._save_original = True
        
        # printing
        self.verbose = verbose
            
    def process(self, events):
        ##############################################
        ############### Initialization ###############
        ##############################################
        # bookkeeping for dataset's name
        dataset = events.metadata.get("dataset", "untitled")
#         isMC = events.metadata["isMC"]
#         if isinstance(isMC, str): 
#             isMC = eval(isMC) 

        # check consistency between is_data and input data
        has_gen = ("GenJet" in events.fields)
        if self.is_data and has_gen: # say data, but has gen
            raise ValueError("Processor set to process data, but contain gen information.")
        elif (not self.is_data) and (not has_gen): # say MC, does not have gen
            raise ValueError("Processor set to process MC, but does not contain gen information.")
        self.fill_gen = (not self.is_data) and self.fill_gen
        
        # define cutflow
        cutflow = OrderedDictWithDefaultInt() #defaultdict(int)
        cutflow["all events"] += len(events)
        
        # define out accumulator
        out = dict()
        
        # apply lumimask
        events = self.lumimask(events, cutflow)
        # save processed lumi list
        if self.is_data and self.save_processed_lumi:
            lumi_list = LumiAccumulator(events.run, events.luminosityBlock, auto_unique=True)
        
        ##############################################
        ###### Control Histogram Initialization ######
        ##############################################
        # define axes for output histograms
        # bookkeeping axes
        dataset_axis = hist.axis.StrCategory([], name="dataset", label="Primary Dataset", growth=True)
        # jet_type_axis is used to specify which jet is a control, e.g. used for x-axis and selecting eta range
        jet_types = [self.off_jet_label, self.on_jet_label]
        jet_type_axis = hist.axis.StrCategory(jet_types, name="jet_type", label="Types of Jet", growth=False)
        if self.fill_gen:
            if "response" in self.hist_to_fill or "asymmetry" in self.hist_to_fill:
                jet_type_match_gen_axis = hist.axis.StrCategory(jet_types+[self.off_jet_label+"_gen", self.on_jet_label+"_gen"], 
                                                                name="jet_type", label="Types of Jet", growth=False)
            if "jet_pt" in self.hist_to_fill or "jet_eta" in self.hist_to_fill or "jet_phi" in self.hist_to_fill:
                jet_type_plus_gen_axis = hist.axis.StrCategory(jet_types+[self.gen_jet_label], 
                                                               name="jet_type", label="Types of Jet", growth=False)
        
        # original variable axes
        jet_pt_axis = self.get_pt_axis(self.pt_binning, num_bins=50)
        jet_eta_axis = self.get_eta_axis(self.eta_binning, num_bins=50)
        jet_phi_axis = hist.axis.Regular(50, -np.pi, np.pi, circular=True, name="jet_phi", label=r"$\phi^{jet}$")
        
        # MET-related axes
        if any([_.startswith("met_") for _ in self.hist_to_fill]):
            met_type_axis = hist.axis.StrCategory([self.off_MET_name, self.on_MET_name], name="met_type", label="Types of MET")
        if "met" in self.hist_to_fill or "met_orig" in self.hist_to_fill:
            met_axis = hist.axis.Regular(50, 0, 800, name="met", label=r"$MET$", flow=True)
        if "met_sumEt" in self.hist_to_fill or "met_sumEt_orig" in self.hist_to_fill:
            met_sumEt_axis = hist.axis.Regular(50, 0, 1, name="met_sumEt", label=r"$MET/\sum E_T$")
            
        # fill control histograms
        if "met_orig" in self.hist_to_fill or "met_sumEt_orig" in self.hist_to_fill:
            weight = 1
            if not self.is_data:
                weight = events.genWeight
        if "met_orig" in self.hist_to_fill:
            out["met_orig"] = hist.Hist(dataset_axis, met_type_axis, met_axis, storage=self.storage)
            out["met_orig"].fill(dataset=dataset, met_type=self.off_MET_name, 
                                 met=events[self.off_MET_name].pt, 
                                 weight=weight)
            out["met_orig"].fill(dataset=dataset, met_type=self.on_MET_name, 
                                 met=events[self.on_MET_name].pt,
                                 weight=weight)
        if "met_sumEt_orig" in self.hist_to_fill:
            out["met_sumEt_orig"] = hist.Hist(dataset_axis, met_type_axis, met_sumEt_axis, storage=self.storage)
            if "sumEt" in events[self.off_MET_name].fields:
                out["met_sumEt_orig"].fill(dataset=dataset, met_type=self.off_MET_name,
                                           met_sumEt=events[self.off_MET_name].pt/events[self.off_MET_name].sumEt,
                                           weight=weight)
            if "sumEt" in events[self.on_MET_name].fields:
                out["met_sumEt_orig"].fill(dataset=dataset, met_type=self.on_MET_name,
                                           met_sumEt=events[self.on_MET_name].pt/events[self.on_MET_name].sumEt,
                                           weight=weight)
        if "jet_pt_orig" in self.hist_to_fill or "jet_eta_orig" in self.hist_to_fill or "jet_phi_orig" in self.hist_to_fill:
            max_leading = self.max_leading_off_jet._max_leading
            off_jets_orig = events[self.off_jet_name][:, :max_leading]
            on_jets_orig = events[self.on_jet_name][:, :max_leading]
            off_jet_weight = 1
            on_jet_weight = 1
            if not self.is_data:
                off_jet_weight = ak.flatten(ak.broadcast_arrays(events.genWeight, off_jets_orig.pt)[0])
                on_jet_weight = ak.flatten(ak.broadcast_arrays(events.genWeight, on_jets_orig.pt)[0])
                if self.fill_gen:
                    gen_jets_orig = events[self.gen_jet_name][:, :max_leading]
                    gen_jet_weight = ak.flatten(ak.broadcast_arrays(events.genWeight, gen_jets_orig.pt)[0])
            
        if "jet_pt_orig" in self.hist_to_fill:
            if not self.fill_gen:
                out["jet_pt_orig"] = hist.Hist(dataset_axis, jet_type_axis, jet_pt_axis, storage=self.storage)
            else:
                out["jet_pt_orig"] = hist.Hist(dataset_axis, jet_type_plus_gen_axis, jet_pt_axis, storage=self.storage)
            out["jet_pt_orig"].fill(dataset=dataset, jet_type=self.off_jet_label, 
                                    jet_pt=ak.flatten(off_jets_orig.pt), weight=off_jet_weight)
            out["jet_pt_orig"].fill(dataset=dataset, jet_type=self.on_jet_label, 
                                    jet_pt=ak.flatten(on_jets_orig.pt), weight=on_jet_weight)
            if self.fill_gen:
                out["jet_pt_orig"].fill(dataset=dataset, jet_type=self.gen_jet_label, 
                                        jet_pt=ak.flatten(gen_jets_orig.pt), weight=gen_jet_weight)
        if "jet_eta_orig" in self.hist_to_fill:
            if not self.fill_gen:
                out["jet_eta_orig"] = hist.Hist(dataset_axis, jet_type_axis, jet_eta_axis, storage=self.storage)
            else:
                out["jet_eta_orig"] = hist.Hist(dataset_axis, jet_type_plus_gen_axis, jet_eta_axis, storage=self.storage)
            out["jet_eta_orig"].fill(dataset=dataset, jet_type=self.off_jet_label, 
                                    jet_eta=ak.flatten(off_jets_orig.eta), weight=off_jet_weight)
            out["jet_eta_orig"].fill(dataset=dataset, jet_type=self.on_jet_label, 
                                    jet_eta=ak.flatten(on_jets_orig.eta), weight=on_jet_weight)
            if self.fill_gen:
                out["jet_eta_orig"].fill(dataset=dataset, jet_type=self.gen_jet_label, 
                                         jet_eta=ak.flatten(gen_jets_orig.eta), weight=gen_jet_weight)
        if "jet_phi_orig" in self.hist_to_fill:
            if not self.fill_gen:
                out["jet_phi_orig"] = hist.Hist(dataset_axis, jet_type_axis, jet_phi_axis)
            else:
                out["jet_phi_orig"] = hist.Hist(dataset_axis, jet_type_plus_gen_axis, jet_phi_axis)
            out["jet_phi_orig"].fill(dataset=dataset, jet_type=self.off_jet_label, 
                                     jet_phi=ak.flatten(off_jets_orig.phi), weight=off_jet_weight)
            out["jet_phi_orig"].fill(dataset=dataset, jet_type=self.on_jet_label, 
                                     jet_phi=ak.flatten(on_jets_orig.phi), weight=on_jet_weight)
            if self.fill_gen:
                out["jet_phi_orig"].fill(dataset=dataset, jet_type=self.gen_jet_label, 
                                         jet_phi=ak.flatten(gen_jets_orig.phi), weight=gen_jet_weight)
        
        ##############################################
        ############## Event Selection ###############
        ##############################################
        
        # events-level selections
        # good event cuts
        events = self.min_npvgood(events, cutflow)
        events = self.max_pv_z(events, cutflow)
        events = self.max_pv_rxy(events, cutflow)
        
        # Flag filters
        events = self.flag_filters(events, cutflow)
        
        # pthatmax < pthat
        events = self.pthatmax_less_than_binvar(events, cutflow)
        
        # minimum numbers of jets
        events = self.min_off_jet(events, cutflow)
        events = self.min_on_jet(events, cutflow)
        
        # PV matching
        events = self.close_pv_z(events, cutflow)
        
        # MET cuts
        #events = self.max_MET(events, cutflow)
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
        if self.fill_gen:
            events = self.gen_jet_veto_map(events, cutflow)
        
        # jet energy correction
        events = self.off_jet_JEC(events, cutflow)
        events = self.on_jet_JEC(events, cutflow)
        
        # jets minimum pt
        events = self.off_jet_min_pt(events, cutflow)
        events = self.on_jet_min_pt(events, cutflow)
        
#         if self.off_jet_JEC.status:
#             events, off_correction_level_in_use = self.off_jet_JEC(events, cutflow)
#         else:
#             events = self.off_jet_JEC(events, cutflow)
#             off_correction_level_in_use = {"orig"}
        
#         if self.on_jet_JEC.status:
#             events, on_correction_level_in_use = self.on_jet_JEC(events, cutflow)
#         else:
#             events = self.on_jet_JEC(events, cutflow)
#             on_correction_level_in_use = {"orig"}
        
        # tag and probe
        events = self.on_off_tagprobe(events, cutflow)
        
        # delta R matching
        events = self.deltaR_matching(events, cutflow)
                
        # select n leading jets to plot
        events = self.max_leading_off_jet(events)
        events = self.max_leading_on_jet(events)
        if self.fill_gen:
            events = self.max_leading_gen_jet(events)
        
        # same eta binning
        events = self.same_eta_bin(events, cutflow)
        
        # check before filling histogram
        assert len(events[self.off_jet_name]) == len(events[self.on_jet_name]), "online and offline must have the same length for histogram filling, but get online: {} and offline: {}".format(len(events[self.off_jet_name]), len(events[self.on_jet_name]))
        events = events[ak.num(events[self.off_jet_name]) > 0] # remove empty events
        
        ##############################################
        ########## Histogram Initialization ##########
        ##############################################
        
        # out accumulator
        out["cutflow"] = {dataset: cutflow} # {dataset: cutflow}
        # save luminosity
        if self.is_data and self.save_processed_lumi:
            out["processed_lumi"] = {dataset: {"lumi_list": lumi_list}}
        
        # correction levels to fill
#         correction_level_suffix_dict = {"raw":"Raw", "orig":"Original", "jec":"Corrected"}
#         # list of (off_correction_level, on_correction_level)
#         # this makes it possible to opt-out correction levels by removing from correction_level_suffix_dict
#         off_correction_level_names = [_ for _ in correction_level_suffix_dict.keys() if _ in off_correction_level_in_use]
#         on_correction_level_names = [_ for _ in correction_level_suffix_dict.keys() if _ in on_correction_level_in_use]
#         if not self.mix_correction_level:
#             correction_level_names = [(_, _) for _ in correction_level_suffix_dict.keys() \
#                                       if _ in off_correction_level_in_use and _ in on_correction_level_in_use]
#         else:
#             correction_level_names = itertools.product(off_correction_level_names, on_correction_level_names)
            
#         correction_level_dict = { # dict comprehension
#                                  (off_name, on_name):
#                                      "off={}:on={}"\
#                                         .format(correction_level_suffix_dict[off_name], correction_level_suffix_dict[on_name])
#                                  for off_name, on_name in correction_level_names
#                                 }
        
#         correction_level_axis = hist.axis.StrCategory(list(correction_level_dict.values()), \
#                                                       name="correction_level", label=r"Correction levels", growth=False)
        
        # derived variable axes and build histograms
        if "response" in self.hist_to_fill:
            response_axis = hist.axis.Regular(100, 0, 5, name="response", label=r"$p_T$ response")
            if self.fill_gen:
                out["response"] = hist.Hist(dataset_axis, jet_type_match_gen_axis, 
                                            jet_pt_axis, jet_eta_axis, #jet_phi_axis,
                                            response_axis, storage=self.storage,
                                            name="response", label=r"$p_T$ response")
            else:
                out["response"] = hist.Hist(dataset_axis, jet_type_axis, 
                                            jet_pt_axis, jet_eta_axis, #jet_phi_axis,
                                            response_axis, storage=self.storage,
                                            name="response", label=r"$p_T$ response")
                
        if "asymmetry" in self.hist_to_fill:
            asymmetry_axis = hist.axis.Regular(100, -1, 1, name="asymmetry", label=r"Asymmetry $A$")
            if self.fill_gen:
                out["asymmetry"] = hist.Hist(dataset_axis, jet_type_match_gen_axis,
                                             jet_pt_axis, jet_eta_axis, #jet_phi_axis,
                                             asymmetry_axis, storage=self.storage,
                                             name="asymmetry", label=r"Asymmetry $A$")
            else:
                out["asymmetry"] = hist.Hist(dataset_axis, jet_type_axis,
                                             jet_pt_axis, jet_eta_axis, #jet_phi_axis,
                                             asymmetry_axis, storage=self.storage,
                                             name="asymmetry", label=r"Asymmetry $A$")
            
        # 2d correlation histogram
        if "comparison" in self.hist_to_fill:
            first_jet_pt_axis = self.get_pt_axis(self.pt_binning, num_bins=50,
                                                 name="first_jet_pt", label=r"$p_T^{jet_1}$")
            second_jet_pt_axis = self.get_pt_axis(self.pt_binning, num_bins=50,
                                              name="second_jet_pt", label=r"$p_T^{jet_2}$")
            cmp_jet_types = [self.off_jet_label]
            if self.on_off_tagprobe.status:
                cmp_jet_types += [self.off_jet_label + "_tag", self.on_jet_label + "_tag"]
            if (not self.is_data) and self.fill_gen:
                cmp_jet_types += [self.off_jet_label + "_gen", self.on_jet_label + "_gen"]        
            cmp_jet_type_axis = hist.axis.StrCategory(cmp_jet_types, name="jet_type", label="Types of Jet", growth=False)
            out["comparison"] = hist.Hist(dataset_axis, cmp_jet_type_axis, jet_eta_axis, #jet_phi_axis,
                                          first_jet_pt_axis, second_jet_pt_axis, storage=self.storage,
                                          name="comparison", label="$p_T$ comparison")
            
        if "gen_comparison" in self.hist_to_fill:
            gen_jet_pt_axis = self.get_pt_axis(self.pt_binning, num_bins=100,
                                               name="gen_jet_pt", label=r"$p_T^{%s}$"%self.gen_jet_label)
            off_jet_pt_axis = self.get_pt_axis(self.pt_binning, num_bins=100,
                                               name="off_jet_pt", label=r"$p_T^{%s}$"%self.off_jet_label)
            on_jet_pt_axis = self.get_pt_axis(self.pt_binning, num_bins=100,
                                              name="on_jet_pt", label=r"$p_T^{%s}$"%self.on_jet_label)
            out["gen_comparison"] = hist.Hist(dataset_axis, jet_eta_axis, #jet_phi_axis,
                                              gen_jet_pt_axis, off_jet_pt_axis, on_jet_pt_axis, storage=self.storage,
                                              name="gen_comparison", label="Online vs Offline [Gen]")
        # tag and probe histograms
        if "tp_response" in self.hist_to_fill and self.on_off_tagprobe.status:
            tp_response_axis = hist.axis.Regular(100, 0, 5, name="tp_response", 
                                                   label=r"Tag and Probe $p_T$ response")
            out["tp_response"] = hist.Hist(dataset_axis, jet_type_axis, jet_pt_axis, jet_eta_axis, #jet_phi_axis, 
                                           tp_response_axis, storage=self.storage,
                                           name="tp_response", label=r"Tag and Probe $p_T$ response")
            
        if "tp_asymmetry" in self.hist_to_fill and self.on_off_tagprobe.status:
            tp_asymmetry_axis = hist.axis.Regular(100, -1, 1, name="tp_asymmetry", label=r"Tag and Probe Asymmetry $A$")
            out["tp_asymmetry"] = hist.Hist(dataset_axis, jet_type_axis, jet_pt_axis, jet_eta_axis, #jet_phi_axis,
                                            tp_asymmetry_axis, storage=self.storage,
                                            name="tp_asymmetry", label=r"Tag and Probe Asymmetry $A$")
            
        if "tp_metprojection" in self.hist_to_fill and self.on_off_tagprobe.status:
            tp_metprojection_axis = hist.axis.Regular(100, -2, 2, name="tp_metprojection", 
                                                      label=r"Tag and Probe MET Projection $B$", flow=True)
            out["tp_metprojection"] = hist.Hist(dataset_axis, jet_type_axis, jet_pt_axis, jet_eta_axis, #jet_phi_axis,
                                                tp_metprojection_axis, storage=self.storage,
                                                name="tp_metprojection", label=r"Tag and Probe MET Projection $B$")
            
            
        if "tp_comparison" in self.hist_to_fill and self.on_off_tagprobe.status and self.on_off_tagprobe._match_tag:
            tag_jet_pt_axis = self.get_pt_axis(self.pt_binning, num_bins=100,
                                               name="tag_jet_pt", label=r"$p_T^{%s, tag}$"%self.off_jet_label)
            off_jet_probe_pt_axis = self.get_pt_axis(self.pt_binning, num_bins=100,
                                               name="off_jet_probe_pt", label=r"$p_T^{%s, probe}$"%self.off_jet_label)
            on_jet_probe_pt_axis = self.get_pt_axis(self.pt_binning, num_bins=100,
                                              name="on_jet_probe_pt", label=r"$p_T^{%s, probe}$"%self.on_jet_label)
            out["tp_comparison"] = hist.Hist(dataset_axis, jet_eta_axis, tag_jet_pt_axis, 
                                             off_jet_probe_pt_axis, on_jet_probe_pt_axis, storage=self.storage,
                                             name="tp_comparison", label="Online vs Offline Probe [Tag]")
            
        
        # 1D histograms (technically, these are reducible from most of above histograms)
        if "jet_pt" in self.hist_to_fill:
#             corrected_jet_types = [self.off_jet_label+"_"+correction_level_suffix_dict[_] for _ in off_correction_level_names]
#             corrected_jet_types += [self.on_jet_label+"_"+correction_level_suffix_dict[_] for _ in on_correction_level_names]
#             corrected_jet_type_axis = hist.axis.StrCategory(corrected_jet_types, 
#                                                             name="jet_type", label="Types of Jet", growth=False)
#             h_jet_pt = hist.Hist(dataset_axis, corrected_jet_type_axis, jet_pt_axis, storage=self.storage)
            if self.fill_gen:
                out["jet_pt"] = hist.Hist(dataset_axis, jet_type_plus_gen_axis, jet_pt_axis, storage=self.storage)
            else:   
                out["jet_pt"] = hist.Hist(dataset_axis, jet_type_axis, jet_pt_axis, storage=self.storage)
        if "jet_eta" in self.hist_to_fill:
            if self.fill_gen:
                out["jet_eta"] = hist.Hist(dataset_axis, jet_type_plus_gen_axis, jet_eta_axis, storage=self.storage)
            else:   
                out["jet_eta"] = hist.Hist(dataset_axis, jet_type_axis, jet_eta_axis, storage=self.storage)
        if "jet_phi" in self.hist_to_fill:
            if self.fill_gen:
                out["jet_phi"] = hist.Hist(dataset_axis, jet_type_plus_gen_axis, jet_phi_axis, storage=self.storage)
            else:   
                out["jet_phi"] = hist.Hist(dataset_axis, jet_type_axis, jet_phi_axis, storage=self.storage)   
        
        ##############################################
        ############# Histogram Filling ##############
        ##############################################
        # filling histograms
        if self.verbose > 1:
            print("filling histogram: linear")

        if len(events[self.off_jet_name]) == 0: # ak.flatten has axis=1 as default and this can raise error with 0 length
            # not so sure how to really handle this
            # here, we just skip filling to speed up
            if self.verbose > 1:
                print("no events to fill histograms")
            return out
        
        # retrive jets
        off_jets = events[self.off_jet_name]
        on_jets = events[self.on_jet_name]
        if self.fill_gen:
            gen_jets = events[self.gen_jet_name]
        
        # get event weight
        if (not self.is_data) and self.use_weight:
            weight = ak.flatten(ak.broadcast_arrays(events.genWeight, off_jets.pt)[0])
            #if self.fill_gen:
            #    gen_matched_weight = ak.flatten(ak.broadcast_arrays(events.genWeight, gen_matched_off_jets.pt)[0])
        else:
            weight = 1.0
            #gen_matched_weight = None
        
        # loop correction levels to fill histograms
#         for (off_correction_level_name, on_correction_level_name), correction_level_label in correction_level_dict.items():
#             if "response" in out:
#                 pt_response = matched_on_jets["pt_"+on_correction_level_name] / \
#                               matched_off_jets["pt_"+off_correction_level_name]
#                 # filling offline as x axis
#                 out["response"].fill(dataset=dataset, correction_level=correction_level_label, 
#                                         jet_type=self.off_jet_label,
#                                         jet_pt=ak.flatten(matched_off_jets["pt_"+off_correction_level_name]), \
#                                         jet_eta=ak.flatten(matched_off_jets.eta), \
#                                         #jet_phi=ak.flatten(matched_off_jets.phi), \
#                                         pt_response=ak.flatten(pt_response),
#                                         weight=weight)
                    
#                 # filling online as x axis
#                 out["response"].fill(dataset=dataset, correction_level=correction_level_label, \
#                                         jet_type=self.on_jet_label, \
#                                         jet_pt=ak.flatten(matched_on_jets["pt_"+on_correction_level_name]), \
#                                         jet_eta=ak.flatten(matched_on_jets.eta), \
#                                         #jet_phi=ak.flatten(matched_on_jets.phi), \
#                                         pt_response=(1 / ak.flatten(pt_response)),
#                                         weight=weight)
                
                # optionally, fill gen as x axis
#                 if (not self.is_data) and self.fill_gen:
#                     gen_matched_pt_response = gen_matched_on_jets["pt_"+on_correction_level_name] / \
#                                               gen_matched_off_jets["pt_"+off_correction_level_name]
#                     out["pt_response"].fill(dataset=dataset, correction_level=correction_level_label, \
#                                             jet_type="Gen",
#                                             jet_pt=ak.flatten(gen_matched_gen_jets.pt), 
#                                             jet_eta=ak.flatten(gen_matched_gen_jets.eta), 
#                                             #jet_phi=ak.flatten(gen_matched_gen_jets.phi), 
#                                             pt_response=ak.flatten(gen_matched_pt_response),
#                                             weight=gen_matched_weight)
#                     out["pt_response"].fill(dataset=dataset, correction_level=correction_level_label, \
#                                             jet_type=self.off_jet_label + " (Matched Gen)",
#                                             jet_pt=ak.flatten(gen_matched_off_jets["pt_"+off_correction_level_name]), \
#                                             jet_eta=ak.flatten(gen_matched_off_jets.eta), \
#                                             #jet_phi=ak.flatten(gen_matched_off_jets.phi), \
#                                             pt_response=ak.flatten(gen_matched_pt_response),
#                                             weight=gen_matched_weight)
#                     out["pt_response"].fill(dataset=dataset, correction_level=correction_level_label, \
#                                             jet_type=self.on_jet_label + " (Matched Gen)", \
#                                             jet_pt=ak.flatten(gen_matched_on_jets["pt_"+on_correction_level_name]), \
#                                             jet_eta=ak.flatten(gen_matched_on_jets.eta), \
#                                             #jet_phi=ak.flatten(gen_matched_on_jets.phi), \
#                                             pt_response=(1 / ak.flatten(gen_matched_pt_response)),
#                                             weight=gen_matched_weight)
#                 if self.ave_jet:
#                     out["pt_response"].fill(dataset=dataset, correction_level=correction_level_label, \
#                                             jet_type="Ave",
#                                             jet_pt=ak.flatten(0.5*(matched_off_jets["pt_"+on_correction_level_name]\
#                                                               + matched_on_jets["pt_"+on_correction_level_name])), \
#                                             jet_eta=ak.flatten(matched_off_jets.eta), \
#                                             jet_phi=ak.flatten(matched_off_jets.phi), \
#                                             pt_response=ak.flatten(pt_response),
#                                             weight=weight)
#                     if (not self.is_data) and self.fill_gen:
#                         out["pt_response"].fill(dataset=dataset, correction_level=correction_level_label, \
#                                                 jet_type="Ave (Matched Gen)",
#                                                 jet_pt=ak.flatten(0.5*(gen_matched_off_jets["pt_"+on_correction_level_name]\
#                                                                   + gen_matched_on_jets["pt_"+on_correction_level_name])), \
#                                                 jet_eta=ak.flatten(gen_matched_off_jets.eta), \
#                                                 jet_phi=ak.flatten(gen_matched_off_jets.phi), \
#                                                 pt_response=ak.flatten(gen_matched_pt_response),
#                                                 weight=gen_matched_weight)
            
#             if "asymmetry" in out:
#                 sum_pt = matched_off_jets["pt_"+off_correction_level_name] + matched_on_jets["pt_"+on_correction_level_name]
#                 diff_pt = matched_on_jets["pt_"+on_correction_level_name] - matched_off_jets["pt_"+off_correction_level_name]
#                 pt_balance = diff_pt / sum_pt
#                 average_pt = 0.5 * sum_pt
                
#                 # filling offline as x axis
#                 out["asymmetry"].fill(dataset=dataset, correction_level=correction_level_label, 
#                                        jet_type=self.off_jet_label, \
#                                        jet_pt=ak.flatten(average_pt), \
#                                        jet_eta=ak.flatten(matched_off_jets.eta), \
#                                        #jet_phi=ak.flatten(matched_off_jets.phi), \
#                                        pt_balance=ak.flatten(pt_balance),
#                                        weight=weight)
#                 # filling online as x axis
#                 out["asymmetry"].fill(dataset=dataset, correction_level=correction_level_label, 
#                                        jet_type=self.on_jet_label, \
#                                        jet_pt=ak.flatten(average_pt), \
#                                        jet_eta=ak.flatten(matched_on_jets.eta), \
#                                        #jet_phi=ak.flatten(matched_on_jets.phi), \
#                                        pt_percent_diff= -1*ak.flatten(pt_balance),
#                                        weight=weight)
                
                # optionally, fill gen as x axis
#                 if (not self.is_data) and self.fill_gen:
#                     try:
#                         matched_off_genjets = matched_off_jets.matched_gen
#                         out["pt_percent_difference"].fill(dataset=dataset, correction_level=correction_level_label, \
#                                            jet_type=self.off_jet_label + "_Gen", \
#                                            jet_pt=ak.to_numpy(ak.flatten(matched_off_genjets.pt), allow_missing=True), \
#                                            jet_eta=ak.to_numpy(ak.flatten(matched_off_genjets.eta), allow_missing=True), \
#                                            jet_phi=ak.to_numpy(ak.flatten(matched_off_genjets.phi), allow_missing=True), \
#                                            pt_percent_diff=ak.flatten(percent_diff_py),
#                                                          weight=weight)
#                     except:   
#                         if self.verbose > 0:
#                             warnings.warn("Fail to retrieve matched gen for offline")
#                     try:
#                         matched_on_genjets = matched_on_jets.matched_gen
#                         out["pt_percent_difference"].fill(dataset=dataset, correction_level=correction_level_label, \
#                                            jet_type=self.on_jet_label + "_Gen", \
#                                            jet_pt=ak.to_numpy(ak.flatten(matched_on_genjets.pt), allow_missing=True), \
#                                            jet_eta=ak.to_numpy(ak.flatten(matched_on_genjets.eta), allow_missing=True), \
#                                            jet_phi=ak.to_numpy(ak.flatten(matched_on_genjets.phi), allow_missing=True), \
#                                            pt_percent_diff= -1*ak.flatten(percent_diff),
#                                                          weight=weight)

#                     except:
#                         if self.verbose > 0:
#                             warnings.warn("Fail to retrieve matched gen for online")
            
#             # comparison histogram
#             if "comparison" in out:  
#                 out["comparison"].fill(dataset=dataset, correction_level=correction_level_label, 
#                                        jet_type=self.off_jet_label,\
#                                        jet_eta=ak.flatten(matched_off_jets.eta), \
#                                        jet_phi=ak.flatten(matched_off_jets.phi), \
#                                        off_jet_pt=ak.flatten(matched_off_jets["pt_"+off_correction_level_name]), \
#                                        on_jet_pt=ak.flatten(matched_on_jets["pt_"+on_correction_level_name]),
#                                        weight=weight)
                
#                 if (not self.is_data) and self.fill_gen:
#                     out["comparison"].fill(dataset=dataset, correction_level=correction_level_label, 
#                                            jet_type=self.off_jet_label + " (Matched Gen)",\
#                                            jet_eta=ak.flatten(gen_matched_off_jets.eta), \
#                                            jet_phi=ak.flatten(gen_matched_off_jets.phi), \
#                                            off_jet_pt=ak.flatten(gen_matched_off_jets["pt_"+off_correction_level_name]), \
#                                            on_jet_pt=ak.flatten(gen_matched_on_jets["pt_"+on_correction_level_name]),
#                                            weight=gen_matched_weight)
#             if "gen_comparison" in out:
#                 out["gen_comparison"].fill(dataset=dataset, correction_level=correction_level_label, 
#                                            #jet_type="Gen",\
#                                            jet_eta=ak.flatten(gen_matched_gen_jets.eta), \
#                                            #jet_phi=ak.flatten(gen_matched_gen_jets.phi), \
#                                            gen_jet_pt=ak.flatten(gen_matched_gen_jets.pt), \
#                                            off_jet_pt=ak.flatten(gen_matched_off_jets["pt_"+off_correction_level_name]), \
#                                            on_jet_pt=ak.flatten(gen_matched_on_jets["pt_"+on_correction_level_name]),
#                                            weight=gen_matched_weight)

        # response
        if "response" in out:
            pt_response = on_jets.pt / off_jets.pt
            out["response"].fill(dataset=dataset, 
                                jet_type=self.off_jet_label,
                                jet_pt=ak.flatten(off_jets.pt), \
                                jet_eta=ak.flatten(off_jets.eta), \
                                #jet_phi=ak.flatten(off_jets.phi), \
                                response=ak.flatten(pt_response),
                                weight=weight)
            out["response"].fill(dataset=dataset, 
                                jet_type=self.on_jet_label,
                                jet_pt=ak.flatten(on_jets.pt), \
                                jet_eta=ak.flatten(on_jets.eta), \
                                #jet_phi=ak.flatten(on_jets.phi), \
                                response=1/ak.flatten(pt_response),
                                weight=weight)
            if self.fill_gen:
                out["response"].fill(dataset=dataset, 
                                    jet_type=self.off_jet_label+"_gen",
                                    jet_pt=ak.flatten(gen_jets.pt), \
                                    jet_eta=ak.flatten(gen_jets.eta), \
                                    #jet_phi=ak.flatten(off_jets.phi), \
                                    response=ak.flatten(off_jets.pt / gen_jets.pt),
                                    weight=weight)
                out["response"].fill(dataset=dataset, 
                                    jet_type=self.on_jet_label+"_gen",
                                    jet_pt=ak.flatten(gen_jets.pt), \
                                    jet_eta=ak.flatten(gen_jets.eta), \
                                    #jet_phi=ak.flatten(on_jets.phi), \
                                    response=1/ak.flatten(on_jets.pt / gen_jets.pt),
                                    weight=weight)
        # asymmetry
        if "asymmetry" in out:
            sumpt = on_jets.pt + off_jets.pt
            asymmetry = (on_jets.pt - off_jets.pt) / sumpt
            avept = 0.5*sumpt
            out["asymmetry"].fill(dataset=dataset, 
                                jet_type=self.off_jet_label,
                                jet_pt=ak.flatten(avept), \
                                jet_eta=ak.flatten(off_jets.eta), \
                                #jet_phi=ak.flatten(off_jets.phi), \
                                asymmetry=ak.flatten(asymmetry),
                                weight=weight)
            out["asymmetry"].fill(dataset=dataset, 
                                jet_type=self.on_jet_label,
                                jet_pt=ak.flatten(avept), \
                                jet_eta=ak.flatten(on_jets.eta), \
                                #jet_phi=ak.flatten(on_jets.phi), \
                                asymmetry=ak.flatten(-asymmetry),
                                weight=weight)
            if self.fill_gen:
                out["asymmetry"].fill(dataset=dataset, 
                                    jet_type=self.off_jet_label+"_gen",
                                    jet_pt=ak.flatten(0.5*(gen_jets.pt + off_jets.pt)), \
                                    jet_eta=ak.flatten(gen_jets.eta), \
                                    #jet_phi=ak.flatten(off_jets.phi), \
                                    asymmetry=ak.flatten((off_jets.pt - gen_jets.pt) / (off_jets.pt + gen_jets.pt)),
                                    weight=weight)
                out["asymmetry"].fill(dataset=dataset, 
                                    jet_type=self.on_jet_label+"_gen",
                                    jet_pt=ak.flatten(0.5*(gen_jets.pt + on_jets.pt)), \
                                    jet_eta=ak.flatten(gen_jets.eta), \
                                    #jet_phi=ak.flatten(on_jets.phi), \
                                    asymmetry=ak.flatten((off_jets.pt - gen_jets.pt) / (off_jets.pt + gen_jets.pt)),
                                    weight=weight)
        
        # comparison histogram
        if "comparison" in out:  
            out["comparison"].fill(dataset=dataset,
                                   jet_type=self.off_jet_label,\
                                   jet_eta=ak.flatten(off_jets.eta), \
                                   #jet_phi=ak.flatten(off_jets.phi), \
                                   first_jet_pt=ak.flatten(off_jets.pt), \
                                   second_jet_pt=ak.flatten(on_jets.pt),
                                   weight=weight)
            if self.on_off_tagprobe.status:
                off_jets_tag = events[self.off_jet_name + "_tag"]
                on_jets_tag = events[self.on_jet_name + "_tag"]
                out["comparison"].fill(dataset=dataset,
                       jet_type=self.off_jet_label+"_tag",\
                       jet_eta=ak.flatten(off_jets_tag.eta), \
                       #jet_phi=ak.flatten(off_jets_tag.phi), \
                       first_jet_pt=ak.flatten(off_jets_tag.pt), \
                       second_jet_pt=ak.flatten(off_jets.pt),
                       weight=weight)
                out["comparison"].fill(dataset=dataset,
                       jet_type=self.on_jet_label+"_tag",\
                       jet_eta=ak.flatten(on_jets_tag.eta), \
                       #jet_phi=ak.flatten(on_jets_tag.phi), \
                       first_jet_pt=ak.flatten(on_jets_tag.pt), \
                       second_jet_pt=ak.flatten(on_jets.pt),
                       weight=weight)
            if self.fill_gen:
                out["comparison"].fill(dataset=dataset,
                       jet_type=self.off_jet_label+"_gen",\
                       jet_eta=ak.flatten(gen_jets.eta), \
                       #jet_phi=ak.flatten(gen_jets.phi), \
                       first_jet_pt=ak.flatten(off_jets.pt), \
                       second_jet_pt=ak.flatten(off_jets.pt),
                       weight=weight)
                out["comparison"].fill(dataset=dataset,
                       jet_type=self.on_jet_label+"_gen",\
                       jet_eta=ak.flatten(gen_jets.eta), \
                       #jet_phi=ak.flatten(gen_jets.phi), \
                       first_jet_pt=ak.flatten(gen_jets.pt), \
                       second_jet_pt=ak.flatten(on_jets.pt),
                       weight=weight)
        
        # gen_comparison
        if "gen_comparison" in out:
            out["gen_comparison"].fill(dataset=dataset,
                                       jet_eta=ak.flatten(gen_jets.eta),
                                       gen_jet_pt=ak.flatten(gen_jets.pt), 
                                       off_jet_pt=ak.flatten(off_jets.pt), 
                                       on_jet_pt=ak.flatten(on_jets.pt), 
                                       weight=weight)
                                   
                
        # tag and probe histogram
        if self.on_off_tagprobe.status and any([_.startswith("tp_") for _ in out.keys()]):
            off_jets_probe = events[self.off_jet_name]
            on_jets_probe = events[self.on_jet_name]
            off_jets_tag = events[self.off_jet_name + "_tag"]
            on_jets_tag = events[self.on_jet_name + "_tag"]
            
            if self.is_data:
                off_tp_weight = None
                on_tp_weight = None
            else:
                off_tp_weight = ak.flatten(ak.broadcast_arrays(events.genWeight, off_jets_probe.pt)[0])
                on_tp_weight = ak.flatten(ak.broadcast_arrays(events.genWeight, on_jets_probe.pt)[0])
            
        if "tp_response" in out:
            out["tp_response"].fill(dataset=dataset, jet_type=self.off_jet_label, \
                                       jet_pt=ak.flatten(off_jets_tag.pt), \
                                       jet_eta=ak.flatten(off_jets_tag.eta), \
                                       #jet_phi=ak.flatten(off_jets_tag.phi), \
                                       tp_response=ak.flatten(off_jets_probe.pt / off_jets_tag.pt),
                                       weight=off_tp_weight)
                           
            out["tp_response"].fill(dataset=dataset, jet_type=self.on_jet_label, \
                                      jet_pt=ak.flatten(on_jets_tag.pt), \
                                      jet_eta=ak.flatten(on_jets_tag.eta), \
                                      #jet_phi=ak.flatten(on_jets_tag.phi), \
                                      tp_response=ak.flatten(on_jets_probe.pt / on_jets_tag.pt),
                                      weight=on_tp_weight)
        
        if "tp_asymmetry" in out or "tp_metprojection" in out:
            off_jets_tagprobe_sum_pt = off_jets_probe.pt + off_jets_tag.pt
            off_jets_tagprobe_ave_pt = 0.5 * off_jets_tagprobe_sum_pt
            on_jets_tagprobe_sum_pt = on_jets_probe.pt + on_jets_tag.pt
            on_jets_tagprobe_ave_pt = 0.5 * on_jets_tagprobe_sum_pt
            
        if "tp_asymmetry" in out:           
            off_asymmetry = (off_jets_probe.pt - off_jets_tag.pt) / off_jets_tagprobe_sum_pt
            out["tp_asymmetry"].fill(dataset=dataset, jet_type=self.off_jet_label,
                                     jet_pt=ak.flatten(off_jets_tagprobe_ave_pt),
                                     jet_eta=ak.flatten(off_jets_tag.eta),
                                     #jet_phi=ak.flatten(off_jets_tag.phi),
                                     tp_asymmetry=ak.flatten(off_asymmetry),
                                     weight=off_tp_weight)
            
            on_asymmetry = (on_jets_probe.pt - on_jets_tag.pt) / on_jets_tagprobe_sum_pt
            out["tp_asymmetry"].fill(dataset=dataset, jet_type=self.on_jet_label,
                                      jet_pt=ak.flatten(on_jets_tagprobe_ave_pt),
                                      jet_eta=ak.flatten(on_jets_tag.eta),
                                      #jet_phi=ak.flatten(on_jets_tag.phi),
                                      tp_asymmetry=ak.flatten(on_asymmetry),
                                      weight=on_tp_weight)

        if "tp_metprojection" in out:
            off_metprojection = ((events[self.off_MET_name].dot(off_jets_tag))/off_jets_tag.pt) / off_jets_tagprobe_sum_pt
            out["tp_metprojection"].fill(dataset=dataset, jet_type=self.off_jet_label,
                                         jet_pt=ak.flatten(off_jets_tagprobe_ave_pt),
                                         jet_eta=ak.flatten(off_jets_tag.eta),
                                         #jet_phi=ak.flatten(off_jets_tag.phi),
                                         tp_metprojection=ak.flatten(off_metprojection),
                                         weight=off_tp_weight)
            
            # on_MET = - (on_pt1 + on_pt2 + on_pt3); on_pt3 is vector sum of everything else
            # modified_on_MET = - (on_pt1 + on_pt2 + on_pt3) - off_pt1 + on_pt1
            on_jets_1 = on_jets_tag.nearest(events[self.on_jet_name + "_without_tagprobe"][:, :2])
            modified_on_MET = events[self.on_MET_name] - on_jets_tag + on_jets_1
            
            on_metprojection = ((modified_on_MET.dot(on_jets_tag))/on_jets_tag.pt) / on_jets_tagprobe_sum_pt
            
            out["tp_metprojection"].fill(dataset=dataset, jet_type=self.on_jet_label,
                                         jet_pt=ak.flatten(on_jets_tagprobe_ave_pt),
                                         jet_eta=ak.flatten(on_jets_tag.eta),
                                         #jet_phi=ak.flatten(on_jets_tag.phi),
                                         tp_metprojection=ak.flatten(on_metprojection),
                                         weight=on_tp_weight)
            
        if "tp_comparison" in out:
            out["tp_comparison"].fill(dataset=dataset,
                                      jet_eta=ak.flatten(off_jets_tag.eta), 
                                      #jet_phi=ak.flatten(off_jets_tag.phi), \
                                      tag_jet_pt=ak.flatten(off_jets_tag.pt), \
                                      off_jet_probe_pt=ak.flatten(off_jets.pt), \
                                      on_jet_probe_pt=ak.flatten(on_jets.pt),
                                      weight=off_tp_weight)
        
        # filling 1D histograms
        if "jet_pt" in out:
#             for off_correction_level_name in off_correction_level_names:
#                 out["jet_pt"].fill(dataset=dataset, 
#                                    jet_type=self.off_jet_label+"_"+correction_level_suffix_dict[off_correction_level_name], 
#                                    jet_pt=ak.flatten(matched_off_jets["pt_"+off_correction_level_name]),
#                                    weight=weight)
#             for on_correction_level_name in on_correction_level_names:
#                 out["jet_pt"].fill(dataset=dataset, 
#                                    jet_type=self.on_jet_label+"_"+correction_level_suffix_dict[on_correction_level_name], 
#                                    jet_pt=ak.flatten(matched_on_jets["pt_"+on_correction_level_name]),
#                                    weight=weight)
            out["jet_pt"].fill(dataset=dataset, jet_type=self.off_jet_label, jet_pt=ak.flatten(off_jets.pt),
                               weight=weight)
            out["jet_pt"].fill(dataset=dataset, jet_type=self.on_jet_label, jet_pt=ak.flatten(on_jets.pt),
                               weight=weight)
            if self.fill_gen:
                out["jet_pt"].fill(dataset=dataset, jet_type=self.gen_jet_label, jet_pt=ak.flatten(gen_jets.pt),
                                   weight=weight)

        if "jet_eta" in out:
            out["jet_eta"].fill(dataset=dataset, jet_type=self.off_jet_label, jet_eta=ak.flatten(off_jets.eta),
                                weight=weight)
            out["jet_eta"].fill(dataset=dataset, jet_type=self.on_jet_label, jet_eta=ak.flatten(on_jets.eta),
                                weight=weight)
            if self.fill_gen:
                out["jet_eta"].fill(dataset=dataset, jet_type=self.gen_jet_label, jet_pt=ak.flatten(gen_jets.eta),
                                    weight=weight)
        
        if "jet_phi" in out:
            out["jet_phi"].fill(dataset=dataset, jet_type=self.off_jet_label, jet_phi=ak.flatten(off_jets.phi),
                                weight=weight)
            out["jet_phi"].fill(dataset=dataset, jet_type=self.on_jet_label, jet_phi=ak.flatten(on_jets.phi),
                                weight=weight)
            if self.fill_gen:
                out["jet_phi"].fill(dataset=dataset, jet_type=self.gen_jet_label, jet_phi=ak.flatten(gen_jets.phi),
                                    weight=weight)
        
        # additional control
        if "met" in self.hist_to_fill or "met_sumEt" in self.hist_to_fill:
            weight = 1
            if not self.is_data:
                weight = events.genWeight
        if "met" in self.hist_to_fill:
            out["met"] = hist.Hist(dataset_axis, met_type_axis, met_axis, storage=self.storage)
            out["met"].fill(dataset=dataset, met_type=self.off_MET_name, met=events[self.off_MET_name].pt, weight=weight)
            out["met"].fill(dataset=dataset, met_type=self.on_MET_name, met=events[self.on_MET_name].pt, weight=weight)
        if "met_sumEt" in self.hist_to_fill:
            out["met_sumEt"] = hist.Hist(dataset_axis, met_type_axis, met_sumEt_axis, storage=self.storage)
            if "sumEt" in events[self.off_MET_name].fields:
                out["met_sumEt"].fill(dataset=dataset, met_type=self.off_MET_name,
                                      met_sumEt=events[self.off_MET_name].pt/events[self.off_MET_name].sumEt, weight=weight)
            if "sumEt" in events[self.on_MET_name].fields:
                out["met_sumEt"].fill(dataset=dataset, met_type=self.on_MET_name,
                                      met_sumEt=events[self.on_MET_name].pt/events[self.on_MET_name].sumEt, weight=weight)
        
        return out
        
    def postprocess(self, accumulator):
        return accumulator