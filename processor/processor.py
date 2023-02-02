import numpy as np
import awkward as ak
import hist

from coffea import processor
from coffea.lumi_tools import LumiMask, LumiData, LumiList

from .selector import *
from .selectorbase import SelectorList
from .accumulator import LumiAccumulator

from collections import defaultdict
import warnings

class OHProcessor(processor.ProcessorABC):
    def __init__(self, 
                 off_jet_name, off_jet_label=None, # offline jet
                 on_jet_name="TrigObjJMEAK4", on_jet_label=None, # online jet
                 lumi_json_path=None, # path to gloden json file (good run certification)
                 lumi_csv_path=None, # path to output lumi csv from brilcalc
                 
                 # event-level selections
                 flag_filters=None,  # event_level, apply flag filters, e.g. METfilters
                 min_off_jet=0, min_on_jet=0, # event-level, select events with at least n jets
                 MET_type="MET", max_MET=None, max_MET_sumET=None, # event-level, select max MET and/or max MET/sumET
                 trigger_min_pt=0, # event-level, trigger cut
                 trigger_type=None, trigger_flag_prefix="PFJet", trigger_all_pts=None,
                 
                 # jet-level selections
                 off_jet_Id=None, on_jet_Id=None, # jet-level, jet id cut
                 off_jet_veto_map_json_path=None, on_jet_veto_map_json_path=None, # jet-level, jet veto map cut
                 off_jet_veto_map_correction_name=None, on_jet_veto_map_correction_name=None,
                 off_jet_veto_map_year=None, on_jet_veto_map_year=None, 
                 off_jet_veto_map_type="jetvetomap", on_jet_veto_map_type="jetvetomap",
                 off_jet_weight_filelist=None, on_jet_weight_filelist=None, # weight file for JEC
                 off_jet_tag_probe=True, on_jet_tag_probe=True, # whether to apply tag and probe
                 off_jet_tag_min_pt=0, on_jet_tag_min_pt=0, # tag min pt to apply during tag and probe
                 
                 max_leading_jet=None, # select up to n leading jets to fill histograms
                 storage=None, # storage type for Hist histograms
                 verbose=0):
        
        # which online and offline to use
        # name is used to retrieve 
        # label is used for histograms + plots
        self.off_jet_name = off_jet_name
        self.off_jet_label = off_jet_label if off_jet_label is not None else off_jet_name
        self.on_jet_name = on_jet_name
        self.on_jet_label = on_jet_label if on_jet_label is not None else on_jet_name
        
        # luminosity
        self.lumimask = LumiMask(lumi_json_path) if lumi_json_path else None
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
        self.min_off_jet = MinPhysicsObject(off_jet_name, min_off_jet)
        self.min_on_jet = MinPhysicsObject(on_jet_name, min_on_jet)
        
        # MET cut
        self.max_MET = MaxMET(max_MET, MET_type)
        self.max_MET_sumET = MaxMET_sumET(max_MET_sumET, MET_type)
        
        # trigger cut
        self.min_trigger = MinTrigger(trigger_type, trigger_min_pt, trigger_flag_prefix, trigger_all_pts)
        
        # event-level wrapped jet-level selectors
        # intuitively, these selectors act on jet
        # however, cut earlier is better for performance
        # and make it easier to tag and probe
        off_jet_identification = JetIdentification(off_jet_Id, verbose)
        self.off_jet_Id = EventWrappedPhysicsObjectSelector(off_jet_name, off_jet_identification, discard_empty=True)
        on_jet_identification = JetIdentification(on_jet_Id, verbose)
        self.on_jet_Id = EventWrappedPhysicsObjectSelector(on_jet_name, on_jet_identification, discard_empty=True)
        
        off_jet_veto_map = JetVetoMap(off_jet_veto_map_json_path, off_jet_veto_map_correction_name, 
                                      off_jet_veto_map_year, off_jet_veto_map_type)
        self.off_jet_veto_map = EventWrappedPhysicsObjectSelector(off_jet_name, off_jet_veto_map, discard_empty=True)
        
        on_jet_veto_map = JetVetoMap(on_jet_veto_map_json_path, on_jet_veto_map_correction_name, 
                                     on_jet_veto_map_year, on_jet_veto_map_type)
        self.on_jet_veto_map = EventWrappedPhysicsObjectSelector(on_jet_name, on_jet_veto_map, discard_empty=True)
        
        # additional cuts if tag and probe will be applied
        self.tp_min_on_jet = MinPhysicsObject(on_jet_name, 2 if on_jet_tag_probe else 0)
        self.tp_min_off_jet = MinPhysicsObject(off_jet_name, 2 if off_jet_tag_probe else 0)
        
        # jet-level selections
        # apply to offline jet
        self.off_jet_JEC = JECBlock(off_jet_weight_filelist, verbose)
        self.off_jet_tagprobe = TriggerDijetTagAndProbe(off_jet_tag_min_pt if off_jet_tag_probe else None, 
                                                        max_alpha=1.0, swap=True)
        
        # apply to online jet
        self.on_jet_JEC = JECBlock(on_jet_weight_filelist, verbose)
        self.on_jet_tagprobe = TriggerDijetTagAndProbe(on_jet_tag_min_pt if on_jet_tag_probe else None, 
                                                       max_alpha=1.0, swap=True)
        
        # delta R matching
        self.deltaR_matching = DeltaRMatching(max_deltaR=0.2)
        
        # select only n leading jets to fill histograms
        self.max_leading_jet = MaxLeadingObject(max_leading_jet)
        
        # histograms
        self.storage = storage
        
        # printing
        self.verbose = verbose
            
    def process(self, events):
        # bookkeeping for dataset's name
        dataset = events.metadata.get("dataset", "untitled")
        
        # TrigObjJMEAK{4, 8} are not sorted by pt...
        for physics_object_name in ["TrigObjJMEAK4", "TrigObjJMEAK8"]:
            if physics_object_name in [self.off_jet_name, self.on_jet_name]:
                sort_index = ak.argsort(events[physics_object_name].pt, ascending=False)
                events[physics_object_name] = (events[physics_object_name])[sort_index]
        
        # define cutflow
        cutflow = defaultdict(int)
        cutflow["all events"] += len(events)
        
        # apply lumimask
        if self.lumimask:
            events = events[self.lumimask(events.run, events.luminosityBlock)]
            cutflow["lumimask"] += len(events)
        #lumi_list = list(set(zip(events.run, events.luminosityBlock))) # save processed lumi list
        lumi_list = LumiAccumulator(events.run, events.luminosityBlock) # save processed lumi list
        #lumi_list = processor.accumulator.list_accumulator(zip(events.run, events.luminosityBlock))
        
        # events-level selections
        # good event cuts
        events = self.min_npvgood(events)
        cutflow["NPV >= {}".format(self.min_npvgood.min_NPVGood)] += len(events)
        events = self.max_pv_z(events)
        cutflow["PV |Z| < {} cm".format(self.max_pv_z.max_PV_z)] += len(events)
        events = self.max_pv_rxy(events)
        cutflow["PV |r_xy| < {} cm".format(self.max_pv_rxy.max_PV_rxy)] += len(events)
        
        # minimum numbers of jets
        events = self.min_off_jet(events)
        cutflow["offline jet >= {}".format(self.min_off_jet.min_physics_object)] += len(events)
        events = self.min_on_jet(events)
        cutflow["online jet >= {}".format(self.min_on_jet.min_physics_object)] += len(events)
        
        # MET cuts
        events = self.max_MET(events)
        cutflow["MET < {}".format(self.max_MET.max_MET)] += len(events)
        events = self.max_MET_sumET(events)
        cutflow["MET/sumET < {}".format(self.max_MET_sumET.max_MET_sumET)] += len(events)
        
        # trigger cut
        events = self.min_trigger(events)
        cutflow["trigger cut"] += len(events)
        
        # event-level wrapped jet-level selectors
        # jet identification
        events = self.off_jet_Id(events)
        events = self.on_jet_Id(events)
        # jet veto map
        events = self.off_jet_veto_map(events)
        events = self.on_jet_veto_map(events)
            
        # additional cuts if tag and probe will be applied
        events = self.tp_min_on_jet(events) # at least two HLT dijet (online)
        cutflow["tag and probe online jet >= {}".format(self.tp_min_on_jet.min_physics_object)] += len(events)
        events = self.tp_min_off_jet(events) # at least two offline dijet
        cutflow["tag and probe offline jet >= {}".format(self.tp_min_off_jet.min_physics_object)] += len(events)
        
        # jet-level selections
        # retrive offline and online jets
        on_jets = events[self.on_jet_name]
        off_jets = events[self.off_jet_name]
        
        # apply to offline jets
        if self.verbose > 1:
            print("processing offline jets")
        off_jets = self.off_jet_JEC(off_jets, events)
        if self.off_jet_tagprobe.tag_min_pt:
            off_jets_tag, off_jets = self.off_jet_tagprobe(off_jets[:, 0], off_jets[:, 1], off_jets)
        
        # apply to online jets
        if self.verbose > 1:
            print("processing online jets")
        on_jets = self.on_jet_JEC(on_jets, events)
        if self.on_jet_tagprobe.tag_min_pt:
            on_jets_tag, on_jets = self.on_jet_tagprobe(on_jets[:, 0], on_jets[:, 1], on_jets)
        
        # delta R matching
        matched_off_jets, matched_on_jets, matched_count = self.deltaR_matching(off_jets, on_jets)
        cutflow["delta R < {}".format(self.deltaR_matching.max_deltaR)] += matched_count
        
        # select n leading jets to plot
        matched_off_jets = self.max_leading_jet(matched_off_jets)
        matched_on_jets = self.max_leading_jet(matched_on_jets)
        
        # define axes for output histograms
        # this is roughly log-scale
        pt_bins = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 23, 27, 30, 35, 40, 45, 57, 72, 90, 120, 
                            150, 200, 300, 400, 550, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 10000 ])

        eta_bins = np.array([-5.191, -4.889,  -4.716,  -4.538,  -4.363,  -4.191,  -4.013,  -3.839,  -3.664,  -3.489,
                   -3.314,  -3.139,  -2.964,  -2.853,  -2.65,  -2.5,  -2.322,  -2.172,  -2.043,  -1.93,  -1.83,
                   -1.74,  -1.653,  -1.566,  -1.479,  -1.392,  -1.305,  -1.218,  -1.131,  -1.044,  -0.957,  -0.879,
                   -0.783,  -0.696,  -0.609,  -0.522,  -0.435,  -0.348,  -0.261,  -0.174,  -0.087,  0,  0.087,  0.174,
                   0.261,  0.348,  0.435,  0.522,  0.609,  0.696,  0.783,  0.879,  0.957,  1.044,  1.131,  1.218,
                   1.305,  1.392,  1.479,  1.566,  1.653,  1.74,  1.83,  1.93,  2.043,  2.172,  2.322,  2.5,  2.65,
                   2.853,  2.964,  3.139,  3.314,  3.489,  3.664,  3.839,  4.013,  4.191,  4.363,  4.538,  4.716,
                   4.889, 5.191 ])
        
        
        # bookkeeping axes
        dataset_axis = hist.axis.StrCategory([], name="dataset", label="Primary Dataset", growth=True)
        response_type_axis = hist.axis.StrCategory(["Raw", "Original", "Corrected"], 
                                                   name="response_type", label=r"Types of $p_T$ Response", growth=False)
        jet_type_axis = hist.axis.StrCategory([self.off_jet_label, self.on_jet_label], 
                                               name="jet_type", label="Types of Jet", growth=True)
        
        pt_response_axis = hist.axis.Regular(200, 0, 5, name="pt_response", label=r"$p_T$ response")
        pt_percent_diff_axis = hist.axis.Regular(200, -2, 2, name="pt_percent_diff", label=r"$p_T$ percentage difference")
        tp_diff_ratio_axis = hist.axis.Regular(200, -2, 2, name="tp_diff_ratio", 
                                               label=r"Tag and Probe $p_T$ difference ratio")
        
        jet_pt_axis = hist.axis.Variable(pt_bins, name="jet_pt", label=r"$p_T^{jet}$")
        jet_eta_axis = hist.axis.Variable(eta_bins, name="jet_eta", label=r"$\eta^{jet}$")
        jet_phi_axis = hist.axis.Regular(50, -np.pi, np.pi, name="jet_phi", label=r"$\phi^{jet}$")
        
        on_jet_pt_axis = hist.axis.Regular(100, 1, 10000, transform=hist.axis.transform.log,
                                           name="on_jet_pt", label=r"$p_T^{%s}$"%self.on_jet_label)
        off_jet_pt_axis = hist.axis.Regular(100, 1, 10000, transform=hist.axis.transform.log,
                                            name="off_jet_pt", label=r"$p_T^{%s}$"%self.off_jet_label)
        
        # build histograms
        # response histograms
        h_pt_response = hist.Hist(dataset_axis, response_type_axis, jet_type_axis, 
                                  jet_pt_axis, jet_eta_axis, jet_phi_axis,
                                  pt_response_axis, storage=self.storage,
                                  name="pt_response", label=r"$p_T$ response")
        
        # percentage difference histogram
        h_pt_percent_diff = hist.Hist(dataset_axis, response_type_axis, jet_type_axis,
                                      jet_pt_axis, jet_eta_axis, jet_phi_axis,
                                      pt_percent_diff_axis, storage=self.storage,
                                      name="pt_percent_diff", label=r"$p_T$ percentage difference")
        
        # 1D histograms (technically, these are reducible from above histograms)
        h_jet_pt = hist.Hist(dataset_axis, jet_type_axis, jet_pt_axis, storage=self.storage)
        h_jet_eta = hist.Hist(dataset_axis, jet_type_axis, jet_eta_axis, storage=self.storage)
        h_jet_phi = hist.Hist(dataset_axis, jet_type_axis, jet_phi_axis, storage=self.storage)
        
        # online vs offline histogram
        h_comp = hist.Hist(dataset_axis, jet_type_axis, jet_eta_axis, jet_phi_axis,
                           off_jet_pt_axis, on_jet_pt_axis, storage=self.storage,
                           name="comparison", label="Online vs Offline")
        
        # tag and probe histogram
        if self.off_jet_tagprobe.tag_min_pt or self.on_jet_tagprobe.tag_min_pt:
            h_tp = hist.Hist(dataset_axis, jet_pt_axis, jet_type_axis, jet_eta_axis, jet_phi_axis, 
                             tp_diff_ratio_axis, storage=self.storage,
                             name="tag_and_probe", label="Tag and Probe")
        
        # filling histograms
        if self.verbose > 1:
            print("filling histogram: linear")
        
        assert len(matched_on_jets) == len(matched_off_jets), "online and offline must have the same length for histogram filling, but get online: {} and offline: {}".format(len(matched_on_jets), len(matched_off_jets))
        if len(matched_on_jets) == 0: # ak.flatten has axis=1 as default and this can raise error with 0 length
            # no so sure how to really handle this
            out = {
                "pt_response": h_pt_response,
                "pt_percent_diffence": h_pt_percent_diff, 
                "jet_pt": h_jet_pt, 
                "jet_eta": h_jet_eta,
                "jet_phi": h_jet_phi,
                "comparison": h_comp,
                "cutflow": {dataset: cutflow},
                "processed_lumi": {dataset: {"lumi_list": lumi_list}} #{dataset: {"lumi_list": lumi_list}}
              }
        
            if self.off_jet_tagprobe.tag_min_pt or self.on_jet_tagprobe.tag_min_pt:
                out["tag_and_probe"] = h_tp
            return out
        
        # TODO fix response type
        for response_type, pt_type in zip(["Raw", "Original", "Corrected"], ["pt_raw", "pt_orig", "pt_jec"]):
            pt_response = matched_on_jets.pt / matched_off_jets[pt_type]
            # filling response histogram
            # filling offline as x axis
            h_pt_response.fill(dataset=dataset, response_type=response_type, \
                               jet_type=self.off_jet_label, \
                               jet_pt=ak.flatten(matched_off_jets[pt_type]), \
                               jet_eta=ak.flatten(matched_off_jets.eta), \
                               jet_phi=ak.flatten(matched_off_jets.phi), \
                               pt_response=ak.flatten(pt_response))
            
            # filling online as x axis
            h_pt_response.fill(dataset=dataset, response_type=response_type, \
                               jet_type=self.on_jet_label, \
                               jet_pt=ak.flatten(matched_on_jets.pt), \
                               jet_eta=ak.flatten(matched_on_jets.eta), \
                               jet_phi=ak.flatten(matched_on_jets.phi), \
                               pt_response=ak.flatten(matched_off_jets[pt_type] / matched_on_jets.pt))
            
            # filling percentage difference histogram
            average_pt = 0.5 * (matched_off_jets[pt_type] + matched_on_jets.pt)
            diff_pt = matched_on_jets.pt - matched_off_jets[pt_type]
            
            h_pt_percent_diff.fill(dataset=dataset, response_type=response_type, 
                                   jet_type=self.off_jet_label, \
                                   jet_pt=ak.flatten(average_pt), \
                                   jet_eta=ak.flatten(matched_off_jets.eta), \
                                   jet_phi=ak.flatten(matched_off_jets.phi), \
                                   pt_percent_diff=ak.flatten(diff_pt / average_pt))
            
            h_pt_percent_diff.fill(dataset=dataset, response_type=response_type, 
                                   jet_type=self.on_jet_label, \
                                   jet_pt=ak.flatten(average_pt), \
                                   jet_eta=ak.flatten(matched_on_jets.eta), \
                                   jet_phi=ak.flatten(matched_on_jets.phi), \
                                   pt_percent_diff=ak.flatten(-diff_pt / average_pt))
            
            # filling gen jets as x axis (only for MC)
            if "Gen" not in self.off_jet_name and "Gen" not in self.on_jet_name:
                try:
                    matched_off_genjets = matched_off_jets.matched_gen
                    # filling offline gen as x axis
                    h_pt_response.fill(dataset=dataset, response_type=response_type, \
                                       jet_type=self.off_jet_label + "_Gen", \
                                       jet_pt=ak.to_numpy(ak.flatten(matched_off_genjets.pt), allow_missing=True), \
                                       jet_eta=ak.to_numpy(ak.flatten(matched_off_genjets.eta), allow_missing=True), \
                                       jet_phi=ak.to_numpy(ak.flatten(matched_off_genjets.phi), allow_missing=True), \
                                       pt_response=ak.flatten(pt_response))
                except:   
                    if self.verbose > 0:
                        warnings.warn("Fail to retrieve matched gen for offline")

                try:
                    matched_on_genjets = matched_on_jets.matched_gen
                    # filling online gen as x axis
                    h_pt_response.fill(dataset=dataset, response_type=response_type, \
                                       jet_type=self.on_jet_label + "_Gen", \
                                       jet_pt=ak.flatten(matched_on_genjets.pt), \
                                       jet_eta=ak.flatten(matched_on_genjets.eta), \
                                       jet_phi=ak.flatten(matched_on_genjets.phi), \
                                       pt_response=ak.flatten(matched_off_jets[pt_type] / matched_on_jets.pt))
                    
                except:
                    if self.verbose > 0:
                        warnings.warn("Fail to retrieve matched gen for online")
            
            # 1D pt histogram: offline pt
            h_jet_pt.fill(dataset=dataset, jet_type=self.off_jet_label + "_{}".format(response_type), \
                          jet_pt=ak.flatten(matched_off_jets[pt_type]))
            
            # comparison histogram
            h_comp.fill(dataset=dataset, jet_type=self.off_jet_label + "_{}".format(response_type), \
                        jet_eta=ak.flatten(matched_off_jets.eta), \
                        jet_phi=ak.flatten(matched_off_jets.phi), \
                        off_jet_pt=ak.flatten(matched_off_jets[pt_type]), \
                        on_jet_pt=ak.flatten(matched_on_jets.pt))
            
            # tag and probe histogram
            # NB: these are unmatched (before deltaR matching)!
            if self.off_jet_tagprobe.tag_min_pt:  
                h_tp.fill(dataset=dataset, jet_type=self.off_jet_label, \
                          jet_pt=ak.flatten(off_jets_tag.pt), \
                          jet_eta=ak.flatten(off_jets_tag.eta), \
                          jet_phi=ak.flatten(off_jets_tag.phi), \
                          tp_diff_ratio=ak.flatten((off_jets.pt - off_jets_tag.pt) / off_jets_tag.pt))
            if self.on_jet_tagprobe.tag_min_pt:
                h_tp.fill(dataset=dataset, jet_type=self.on_jet_label, \
                          jet_pt=ak.flatten(on_jets_tag.pt), \
                          jet_eta=ak.flatten(on_jets_tag.eta), \
                          jet_phi=ak.flatten(on_jets_tag.phi), \
                          tp_diff_ratio=ak.flatten((on_jets.pt - on_jets_tag.pt) / on_jets_tag.pt))
        
        # filling the remaining histograms independent of response type
        h_jet_pt.fill(dataset=dataset, jet_type=self.on_jet_label, jet_pt=ak.flatten(matched_on_jets.pt))

        h_jet_eta.fill(dataset=dataset, jet_type=self.off_jet_label, jet_eta=ak.flatten(matched_off_jets.eta))
        h_jet_eta.fill(dataset=dataset, jet_type=self.on_jet_label, jet_eta=ak.flatten(matched_on_jets.eta))

        h_jet_phi.fill(dataset=dataset, jet_type=self.off_jet_label, jet_phi=ak.flatten(matched_off_jets.phi))
        h_jet_phi.fill(dataset=dataset, jet_type=self.on_jet_label, jet_phi=ak.flatten(matched_on_jets.phi))
        
        out = {
                "pt_response": h_pt_response,
                "pt_percent_diffence": h_pt_percent_diff, 
                "jet_pt": h_jet_pt, 
                "jet_eta": h_jet_eta,
                "jet_phi": h_jet_phi,
                "comparison": h_comp,
                "cutflow": {dataset: cutflow},
                "processed_lumi": {dataset: {"lumi_list": lumi_list}} #{dataset: {"lumi_list": lumi_list}}
              }
        
        if self.off_jet_tagprobe.tag_min_pt or self.on_jet_tagprobe.tag_min_pt:
            out["tag_and_probe"] = h_tp
            
        return out
        
    def postprocess(self, accumulator):
        # compute integrated luminosity
        if self.lumi_csv_path:
            lumidata = LumiData(self.lumi_csv_path)
            for dataset in accumulator["processed_lumi"]:
                if len(accumulator["processed_lumi"][dataset]["lumi_list"]) == 0:
                    if self.verbose > 0:
                        warnings.warn("no lumi blocks are processed for dataset: {}!".format(dataset))
                # apply unique
                lumi_list = np.array(accumulator["processed_lumi"][dataset]["lumi_list"])
                lumi_list = LumiList(lumi_list[:, 0], lumi_list[:, 1]) if len(lumi_list) > 0 else LumiList()
                accumulator["processed_lumi"][dataset]["lumi_list"] = lumi_list
                # compute integrated luminosity
                accumulator["processed_lumi"][dataset]["lumi"] = lumidata.get_lumi(lumi_list)
        else:
            for dataset in accumulator["processed_lumi"]:
                if len(accumulator["processed_lumi"][dataset]["lumi_list"]) == 0:
                    if self.verbose > 0:
                        warnings.warn("no lumi blocks are processed for dataset: {}!".format(dataset))
                # apply unique
                lumi_list = np.array(accumulator["processed_lumi"][dataset]["lumi_list"])
                lumi_list = LumiList(lumi_list[:, 0], lumi_list[:, 1]) if len(lumi_list) > 0 else LumiList()
                accumulator["processed_lumi"][dataset]["lumi_list"] = lumi_list

                accumulator["processed_lumi"][dataset]["lumi"] = None
                
        return accumulator