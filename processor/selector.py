from .selectorbase import SelectorABC
import numpy as np
import awkward as ak
from abc import abstractmethod

from coffea.lumi_tools import LumiMask, LumiData, LumiList
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
import correctionlib

from functools import partial
import itertools
import string
import warnings

### event-level ###
### apply: events -> events
class LumiMaskSelector(SelectorABC):
    def __init__(self, lumi_json_path):
        super().__init__(lumi_json_path)
        self._lumimask = LumiMask(lumi_json_path) if lumi_json_path else None
    def __str__(self):
        return "lumimask"
    def apply(self, events):
        return self._lumimask(events.run, events.luminosityBlock)
    
class MinNPVGood(SelectorABC):
    def __init__(self, min_NPVGood=0):
        super().__init__(min_NPVGood)
        self._min_NPVGood = min_NPVGood
    def __str__(self):
        return "NPV > {}".format(self._min_NPVGood)
    def apply(self, events):
        return events[events.PV.npvsGood > self._min_NPVGood]
    
class MaxPV_z(SelectorABC):
    def __init__(self, max_PV_z=24):
        super().__init__(max_PV_z)
        self._max_PV_z = max_PV_z
    def __str__(self):
        return "PV |z| < {} cm".format(self._max_PV_z)
    def apply(self, events):
        return events[np.abs(events.PV.z) < self._max_PV_z]
    
class MaxPV_rxy(SelectorABC):
    def __init__(self, max_PV_rxy=2):
        super().__init__(max_PV_rxy)
        self._max_PV_rxy = max_PV_rxy
    def __str__(self):
        return "PV |r_xy| < {} cm".format(self._max_PV_rxy)
    def apply(self, events):
        return events[np.sqrt(events.PV.x*events.PV.x + events.PV.y*events.PV.y) < self._max_PV_rxy]
    
class ClosePV_z(SelectorABC):
    def __init__(self, first_PV, second_PV, max_dz=None, sigma_multiple=None): # 0.2 for gen, 5 * sigma
        enable = (first_PV is not None) and (second_PV is not None) and (self._max_dz is not None or self._sigma_multiple is not None)
        super().__init__(enable)
        
        assert (max_dz == None and sigma_multiple == None) or (bool(max_dz) != bool(sigma_multiple)) , "Must pick either absolute max dz or multiple of z sigma"
        self._first_PV = first_PV
        self._second_PV = second_PV
        self._max_dz = max_dz
        self._sigma_multiple = sigma_multiple
    def __str__(self):
        if self._max_dz:
            return "|PV1_z - PV2_z| < {}".format(self._max_dz)
        else:
            return "|PV1_z - PV2_z| < {} * sigma".format(self._sigma_multiple)
    def apply(self, events):
        if self._max_dz == None and self._sigma_multiple == None:
            return events
        ndim1 = events[self._first_PV].ndim
        ndim2 = events[self._second_PV].ndim
        z1 = events[self._first_PV].z if ndim1 == 1 else events[self._first_PV].z[:, 0]
        z2 = events[self._second_PV].z if ndim2 == 1 else events[self._second_PV].z[:, 0]
        
        if self._max_dz:
            return events[np.abs(z1 - z2) < self._max_dz]
        else:
            assert "zError" in events[self._first_PV].fields or "zError" in events[self._second_PV].fields, "At least one type of PV must contain error information"
            if "zError" in events[self._first_PV].fields:
                pv1_zError = events[self._first_PV].zError if ndim1 == 1 else events[self._first_PV].zError[:, 0]
            else:
                pv1_zError = np.full(len(events), -np.inf)
            if "zError" in events[self._second_PV].fields:
                pv2_zError = events[self._second_PV].zError if ndim2 == 1 else events[self._second_PV].zError[:, 0]
            else:
                pv2_zError = np.full(len(events), -np.inf)
            zError = np.maximum(pv1_zError, pv2_zError)
            return events[np.abs(z1-z2) < self._sigma_multiple * zError]

class MinPhysicsObject(SelectorABC):
    def __init__(self, physics_object_name, min_physics_object=0, name=""):
        super().__init__(min_physics_object > 0)
        self._physics_object_name = physics_object_name
        self._min_physics_object = min_physics_object
        self._name = name if (len(name) > 0 and name is not None) else physics_object_name
    def __str__(self):
        return "Number of {} >= {}".format(self._name, self._min_physics_object)
    def apply(self, events):
        return events[(ak.num(events[self._physics_object_name]) >= self._min_physics_object)] 

class MinJet(MinPhysicsObject):
    def __init__(self, jet_name, min_jet=0, name=""):
        super().__init__(jet_name, min_jet, name)

class MinTrigger(SelectorABC):
    def __init__(self, trigger_type="single", trigger_min_pt=0, trigger_flag_prefix="PFJet", trigger_all_pts=None):
        enable = (trigger_type is not None) and (trigger_min_pt is not None) and (trigger_min_pt >= 0)
        super().__init__(enable)
        if trigger_type in ["only", "lower_not", "upper_not"]:
            assert trigger_all_pts != None, "trigger type: {} need all trigger pts".format(trigger_type)
            
        self._trigger_min_pt = trigger_min_pt
        self._trigger_flag_prefix = trigger_flag_prefix
        self._trigger_all_pts = trigger_all_pts
        self._trigger_type = trigger_type
        
        if trigger_type == None or trigger_type == "None" or trigger_type == "none":
            self._trigger_type = None
            self._comparison_operation = None
        elif trigger_type == "single":
            self._comparison_operation = None # can also set to function which always returns false
        elif trigger_type == "only":
            self._comparison_operation = (lambda pt, trigger_min_pt: pt != trigger_min_pt)
        elif trigger_type == "lower_not":
            self._comparison_operation = (lambda pt, trigger_min_pt: pt < trigger_min_pt)
        elif trigger_type == "upper_not":
            self._comparison_operation = (lambda pt, trigger_min_pt: pt > trigger_min_pt)
        else:
            raise ValueError("Invalid type of trigger cut")
    def __str__(self):
        if self.status:
            return "Trigger {}{} ({})".format(self._trigger_flag_prefix, self._trigger_min_pt, self._trigger_type)
        else:
            return "Trigger"
    def apply(self, events):
        mask = events.HLT[self._trigger_flag_prefix + str(self._trigger_min_pt)] # single mask as base
        if self._trigger_all_pts != None and self._comparison_operation != None:
            # alternatively, setting trigger_all_pts = [trigger_min_pt] will give the same result
            for pt in self._trigger_all_pts:
                if self._comparison_operation(pt, self._trigger_min_pt):
                    flag = self._trigger_flag_prefix + str(pt)
                    mask = np.logical_and(mask, np.logical_not(events.HLT[flag]))
        return events[mask]

class FlagFilter(SelectorABC):
    def __init__(self, flag_filter):
        super().__init__(flag_filter)
        self._flag_filter = flag_filter
    def __str__(self):
        return self._flag_filter
    def apply(self, events):
        return events[events.Flag[self._flag_filter]]
    
class FlagFilters(SelectorABC):
    def __init__(self, flag_filters):
        super().__init__(flag_filters)
        if isinstance(flag_filters, str):
            flag_filters = flag_filters.split()
        self._flag_filters = flag_filters
    def __str__(self):
        if self._flag_filters is None:
            return "Flag filters"
        return "Flag filters: {}".format(" ".join(self._flag_filters))
    def apply(self, events):
        return events[ak.all([events.Flag[flag] for flag in self._flag_filters], axis=0)]

class METFilter(FlagFilter):
    def __init__(self):
        super().__init__("METFilters")
    apply_met_filter = FlagFilter.apply

class MaxMET(SelectorABC):
    def __init__(self, max_MET, MET_type="MET"):
        super().__init__(max_MET)
        self._MET_type = MET_type
        self._max_MET = max_MET
    def __str__(self):
        return "{} < {} GeV".format(self._MET_type, self._max_MET)
    def apply(self, events):
        return events[events[self._MET_type].pt < self._max_MET]
    
class MaxMET_sumET(SelectorABC):
    def __init__(self, max_MET_sumET, min_MET=0, MET_type="MET"):
        super().__init__(max_MET_sumET)
        self._MET_type = MET_type
        self._max_MET_sumET = max_MET_sumET
        self._min_MET = min_MET
    def __str__(self):
        if self.status:
            if self._min_MET <= 0:
                return "{}/sumET < {}".format(self._MET_type, self._max_MET_sumET)
            return "{} <= {} GeV or {}/sumET < {}".format(self._MET_type, self._min_MET, self._MET_type, self._max_MET_sumET)
        else:
            return "MET/sumET"
    def apply(self, events):
        mask = (events[self._MET_type].pt < self._max_MET_sumET * events[self._MET_type].sumEt)
        if self._min_MET > 0:
            mask = mask | (events[self._MET_type].pt <= self._min_MET)
        return events[mask]
    
class PileupPthatmaxLessThanGeneratorBinvar(SelectorABC):
    def __init__(self, enable):
        super().__init__(enable)
    def __str__(self):
        return "Pile-up pthat_max < Generator binvar"
    def apply(self, events):
        mask = (events.Pileup.pthatmax < events.Generator.binvar)
        return events[mask]
        
# wrap jet-level to event-level selector
class EventWrappedPhysicsObjectSelector(SelectorABC): 
    def __init__(self, physics_object_name, physics_object_selector, discard_empty=False):
        super().__init__()
        self._physics_object_name = physics_object_name
        self._physics_object_selector = physics_object_selector
        self._discard_empty = discard_empty
        self.enable() if self._physics_object_selector.status else self.disable() # copy initial status
    def __str__(self):
        return str(self._physics_object_selector)
    def apply(self, events):
        # physics_object_selector.apply will ignore physics_object_selector's status
        # now this will use EventPhysicsObject's status during __call__
        physics_object = self._physics_object_selector.apply(events[self._physics_object_name])
        events[self._physics_object_name] = physics_object
        if self._discard_empty:
            events = events[ak.num(events[self._physics_object_name]) > 0]
        return events
    
### jet-level ###
### apply: physics_objects -> physics_objects, e.g. jets -> jets
class PhysicsObjectMinField(SelectorABC):
    def __init__(self, physics_object_name, field, min_value):
        super().__init__(field)
        self._physics_object_name = physics_object_name
        self._field = field
        self._min_value = min_value
        
    def __str__(self):
        return "{} {} > {}".format(self._physics_object_name, self._field, self._min_value)
    def apply(self, events):
        physics_object = events[self._physics_object_name]
        mask = (physics_object[self._field] > self._min_value)
        events[self._physics_object_name] = events[self._physics_object_name][mask]
        if self._physics_object_name + "_tag" in events.fields:
            events[self._physics_object_name + "_tag"] = events[self._physics_object_name + "_tag"][mask]
        events = events[ak.num(events[self._physics_object_name]) > 0]
        return events
    def count(self, events):
        return np.sum(ak.num(events[self._physics_object_name]) > 0)

class PhysicsObjectMinPt(PhysicsObjectMinField):
    def __init__(self, physics_object_name, min_pt=0):
        super().__init__(physics_object_name, "pt" if min_pt > 0 else None, min_value=min_pt)
    def __str__(self):
        return "{} pT > {} GeV".format(self._physics_object_name, self._min_value)

class ObjectInRange(SelectorABC):
    def __init__(self, field, min_value=-np.inf, max_value=np.inf, mirror=False, name=""):
        super().__init__(field)
        if field:
            assert len(name) > 0 and name != None, "must provide unique name"
        self._field = field
        self._min_value = min_value
        self._max_value = max_value
        self._mirror = mirror
        self._name = name
    def __str__(self):
        return "{} {} in range ({}, {})".format(self._name, self._field, self._min_value, self._max_value)
    def apply(self, physics_object):
        mask = (physics_object[self._field] > self._min_value) & (physics_object[self._field] < self._max_value)
        if self._mirror:
            mask = mask | ((physics_object[self._field] < -self._min_value) & (physics_object[self._field] > -self._max_value))
        return physics_object[mask]

class ObjectInPtRange(ObjectInRange):
    def __init__(self, min_pt=-np.inf, max_pt=np.inf, name=""):
        super().__init__("pt", min_pt, max_pt, False, name)
    
class ObjectInEtaRange(ObjectInRange):
    def __init__(self, min_eta=-np.inf, max_eta=np.inf, mirror=False, name=""):
        super().__init__("eta", min_eta, max_eta, mirror, name)
        
class MaxLeadingObject(SelectorABC): #TODO: check when this should be applied actually. For T&P, this shouldn't matter
    def __init__(self, max_leading, name=""):
        super().__init__(max_leading)
        if max_leading:
            assert len(name) and name != None, "must provide unique name"
        self._max_leading = max_leading
        self._name = name
    def __str__(self):
        return "{}: upto {} leading".format(self._name, self._max_leading)
    def apply(self, physics_objects):
        if len(physics_objects) == 0:
            return physics_objects
        return physics_objects[:, :self._max_leading]
    
class MaxLeadingPhysicsObject(SelectorABC): #TODO: check when this should be applied actually. For T&P, this shouldn't matter
    def __init__(self, physics_object_name, max_leading):
        super().__init__(max_leading)
        if max_leading:
            assert len(physics_object_name) and physics_object_name != None, "Must specify physics object name"
        self._max_leading = max_leading
        self._physics_object_name = physics_object_name
    def __str__(self):
        return "{}: upto {} leading".format(self._physics_object_name, self._max_leading)
    def apply(self, events):
        physics_objects = events[self._physics_object_name]
        physics_objects = physics_objects[:, :self._max_leading]
        events[self._physics_object_name] = physics_objects
        return events
    def count(self, events):
        return np.sum(ak.num(events[self._physics_object_name]) > 0)

class JetID(SelectorABC):
    def __init__(self, jet_name, jet_id, jet_type="PUPPI"):
        assert jet_id is None\
            or (isinstance(jet_id, int) and jet_id >=2 and jet_id <=7)\
            or (isinstance(jet_id, str) and 
                len({"tight", "tightleptonveto"}.intersection([_.lower() for _ in jet_id.split()]))),\
            "Unrecognized jet id: {}".format(jet_id)
        assert jet_type in ["PUPPI", "CHS"], "Unrecognized jet type: {}".format(jet_type)
        
        self._jet_name = jet_name
        super().__init__(jet_id is not None)
        if jet_id is not None:
            assert self._jet_name is not None, "Must specify jet name"
            self._jet_id = jet_id
            if isinstance(jet_id, str):
                jet_id_bit_dict = {"loose":1, "tight":2, "tightleptonveto":4}
                jet_id_lst = jet_id.split()
                if len(jet_id_lst) == 1:
                    jet_id = jet_id_lst[0]
                    assert jet_id.lower() in jet_id_bit_dict.keys(), "cannot identify jet_id {}".format(jet_id)
                    self._jet_id = jet_id_bit_dict[jet_id.lower()]
                else:
                    self._jet_id = 0
                    for jet_id in jet_id_lst:
                        assert jet_id.lower() in jet_id_bit_dict.keys(), "cannot identify jet_id {}".format(jet_id)
                        self._jet_id += jet_id_bit_dict[jet_id.lower()] # bitwise-or in base 2 is add in base 10

            if isinstance(self._jet_id, int):
                if self._jet_id == 2:
                    jet_id_name = "Tight"
                elif self._jet_id == 3:
                    jet_id_name = "Loose Tight"
                elif self._jet_id == 4:
                    jet_id_name = "TightLeptonVeto"
                elif self._jet_id == 5:
                    jet_id_name = "Loose TightLeptonVeto"
                elif self._jet_id == 6:
                    jet_id_name = "Tight TightLeptonVeto"
                elif self._jet_id == 7:
                    jet_id_name = "Tight TightLeptonVeto"

            self._jet_id_name = jet_id_name
            self._use_lepton_veto = self._jet_id >= 4
            self._jet_type = jet_type if jet_type is not None else "PUPPI"
         
    def __str__(self):
        if self.status:
            return "{}: JetID {} ({})".format(self._jet_name, self._jet_id, self._jet_id_name)
        else:
            return "{}: JetID".format(self._jet_name)
        
    def apply(self, events):
        jets = events[self._jet_name]
        if "jetId" in jets.fields:
            jets = jets[jets.jetId >= self._jet_id]
        else:
            year = events.metadata.get("year", 2022) # default to 2022F for now
            era = events.metadata.get("era", "F")

            if (year == "2022" or year == 2022) and era in "BCDEFG":
                # from https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13p6TeV
                # change to notations used in JetID twiki
                abs_eta = np.abs(jets.eta)

                # it is unclear whether to use nConstElecs or nElectrons fields...
                # from a quick test, it seems like jetId uses nConstElecs
                nElectrons = jets["nConstElecs"] if "nConstElecs" in jets.fields else jets["nElectrons"]
                # also for nConstMuons and nMuons...
                # from a quick test, it seems like jetId uses nConstMuons
                nMuons = jets["nConstMuons"] if "nConstMuons" in jets.fields else jets["nMuons"]
                nPhotons = jets["nConstPhotons"] if "nConstPhotons" in jets.fields else jets["nPhotons"]
                nCh = jets["nConstChHads"] if "nConstChHads" in jets.fields else jets["nCh"]
                nNh = jets["nConstNeuHads"] if "nConstNeuHads" in jets.fields else jets["nNh"]

                NumConst = jets["nConstituents"]
                CHM = nCh + nElectrons + nMuons
                CEMF = jets["chEmEF"]
                CHF = jets["chHEF"]
                NumNeutralParticle = nNh + nPhotons
                NEMF = jets["neEmEF"]
                NHF = jets["neHEF"]
                MUF = jets["muEF"] if "muEF" in jets.fields else jets["muEmEF"]

                if era in "BCDE":
                    mask0 = (abs_eta<=2.6) & (CHM>0) & (CHF>0.01) & (NumConst>1) & (NEMF<0.9) & (NHF<0.9)
                    if self._use_lepton_veto:
                        mask0 = mask0 & (CEMF<0.8) & (MUF<0.8)
                    
                    mask1 = (abs_eta>2.6) & (abs_eta<=2.7) & (NEMF<0.99) & (NHF<0.9) 
                    if self._use_lepton_veto:
                        mask1 = mask1 & (CEMF<0.8) & (MUF<0.8)
                    if self._jet_type == "CHS":
                        mask1 = mask1 & (CHM>0)
                    
                    if self._jet_type == "PUPPI":
                        mask2 = (abs_eta>2.7) & (abs_eta<=3.0) & (NHF<0.9999)
                    else:
                        mask2 = (abs_eta>2.7) & (abs_eta<=3.0) & (NEMF<0.99) & (NumNeutralParticle>1)
                    
                    if self._jet_type == "PUPPI":
                        mask3 = (abs_eta>3.0) & (NEMF<0.90) & (NumNeutralParticle>=2)
                    else:
                        mask3 = (abs_eta>3.0) & (NEMF<0.90) & (NumNeutralParticle>10)
                    mask = mask0 | mask1 | mask2 | mask3
                    jets = jets[mask]
                    
                elif era in "FG":
                    mask0 = (abs_eta<=2.6) & (CHM>0) & (CHF>0.01) & (NumConst>1) & (NEMF<0.9) & (NHF<0.99)
                    if self._use_lepton_veto:
                        mask0 = mask0 & (CEMF<0.8) & (MUF<0.8)
                    
                    mask1 = (abs_eta>2.6) & (abs_eta<=2.7) & (NEMF<0.99) & (NHF<0.9) 
                    if self._use_lepton_veto:
                        mask1 = mask1 & (CEMF<0.8) & (MUF<0.8)
                    if self._jet_type == "CHS":
                        mask1 = mask1 & (CHM>0)
                    
                    if self._jet_type == "PUPPI":
                        mask2 = (abs_eta>2.7) & (abs_eta<=3.0) & (NHF<0.9999)
                    else:
                        mask2 = (abs_eta>2.7) & (abs_eta<=3.0) & (NEMF<0.99) & (NumNeutralParticle>1)
                    
                    if self._jet_type == "PUPPI":
                        mask3 = (abs_eta>3.0) & (NEMF<0.90) & (NumNeutralParticle>=2)
                    else:
                        mask3 = (abs_eta>3.0) & (NEMF<0.90) & (NumNeutralParticle>10)
                    
                    mask = mask0 | mask1 | mask2 | mask3
                    jets = jets[mask]
                else:
                    raise ValueError("Unrecognized era: 2022{}".format(era))
                
            else:
                raise ValueError("Unrecognized year and era: {}{}".format(year, era))
        
        events[self._jet_name] = jets
        events = events[ak.num(events[self._jet_name]) > 0]
        return events
    
    def count(self, events):
        return np.sum(ak.num(events[self._jet_name]) > 0)
    
class JetVetoMap(SelectorABC):
    def __init__(self, jet_name, jet_veto_map_path, map_type="jetvetomap"):
        super().__init__(jet_veto_map_path)
        self._jet_name = jet_name
        if jet_veto_map_path:
            assert len(jet_name) is not None, "Must specify jet name"
            ext = extractor()
            ext.add_weight_sets(["jetvetomap {} {}".format(map_type, jet_veto_map_path)])
            ext.finalize()
            self._jet_veto_map = ext.make_evaluator()
    def __str__(self):
        return "{}: jet veto map".format(self._jet_name)
    def apply(self, events):
        # hard coded eta range
        jets = events[self._jet_name]
        jets = jets[np.abs(jets.eta) <= 5.191]
        # wrap phi in (-pi, pi]
        jets_phi = jets.phi
        jets_phi = ak.where(jets_phi <= -np.pi, 2*np.pi + jets_phi, jets_phi)
        jets_phi = ak.where(jets_phi > np.pi, -2*np.pi + jets_phi, jets_phi)
        veto_map = self._jet_veto_map["jetvetomap"](jets.eta, jets_phi) == 0
        jets = jets[veto_map]
        events[self._jet_name] = jets
        events = events[ak.num(events[self._jet_name]) > 0]
        return events
    def count(self, events):
        return np.sum(ak.num(events[self._jet_name]) > 0)

### tag and probe ###    
class OnlineOfflineDijetTagAndProbe(SelectorABC):
    def __init__(self, off_jet_name, on_jet_name, tag_min_pt, 
                 max_alpha=1.0, third_jet_max_pt=30,
                 opposite_on_jet=False, on_off_ordering=1,
                 max_deltaR=0.2, 
                 match_tag=False, save_original=False):
        super().__init__(tag_min_pt is not None and tag_min_pt >=0)
        self._off_jet_name = off_jet_name
        self._on_jet_name = on_jet_name
        self._tag_min_pt = tag_min_pt
        self._third_jet_max_pt = third_jet_max_pt
        self._max_alpha = max_alpha
        self._max_deltaR = max_deltaR
        self._opposite_on_jet = opposite_on_jet
        self._on_off_ordering = on_off_ordering
        self._match_tag = match_tag
        self._save_original = save_original

    def __str__(self):
        return "Dijet T&P: {} and {}".format(self._off_jet_name, self._on_jet_name)
    def apply(self, events):        
        def trigger_dijet_tag_and_probe(tag, probe, others):
            # tag condition
            tag_mask = tag.pt > self._tag_min_pt
            
            # tag probe condition
            opposite_cut = (np.abs(tag.phi - probe.phi) > 2.7)
            #close_pt_cut = (np.abs(tag.pt - probe.pt) < 0.7 * (tag.pt + probe.pt))
            probe_mask = opposite_cut #& close_pt_cut
            
            tag_probe_mask = tag_mask & probe_mask
            
            # others condition
            # alpha = 2*jet3/(jet1 + jet2) <= 1, so if max_alpha > 1, then alpha_cut does nothing
            if self._max_alpha and (self._max_alpha <= 1):
                three_jets = others[:, :3]
                alpha_cut = (2 * three_jets[:, -1].pt < self._max_alpha * (tag.pt + probe.pt))
                alpha_cut = alpha_cut | (ak.num(others) == 2) | (three_jets[:, -1].pt < self._third_jet_max_pt)
                
                other_mask = alpha_cut
                tag_probe_mask = tag_probe_mask & other_mask
            
            tag_probe_counts = ak.values_astype(tag_probe_mask, int)
            apply_mask = lambda x: ak.unflatten(x[tag_probe_mask], counts=tag_probe_counts)
            return map(apply_mask, (tag, probe))
        
        def match_offline_online_tag(off_tag, off_probe, on_tag, on_probe):
            min_tag_count = ak.min([ak.num(off_tag), ak.num(on_tag)], axis=0)
            min_tag_mask = np.array(ak.values_astype(min_tag_count, bool))
            apply_mask = lambda x: ak.unflatten(ak.flatten(x[min_tag_mask]), counts=min_tag_count)
            return map(apply_mask, (off_tag, off_probe, on_tag, on_probe))

        
        # select events with at least two offline and online jets
        mask = (ak.num(events[self._off_jet_name])>=2) & (ak.num(events[self._on_jet_name])>=2)
        events = events[mask]
        # save original
        if self._save_original:
            events[self._off_jet_name + "_without_tagprobe"] = events[self._off_jet_name]
            events[self._on_jet_name + "_without_tagprobe"] = events[self._on_jet_name]
        
        # select back-to-back on offline
        # this will be done during tag-probe test on offline, regardless
        events = events[np.abs(events[self._off_jet_name][:, 0].phi - events[self._off_jet_name][:, 1].phi) > 2.7]
        if self._opposite_on_jet:
            events = events[np.abs(events[self._on_jet_name][:, 0].phi - events[self._on_jet_name][:, 1].phi) > 2.7]

        # order online with offline jet
        if self._on_off_ordering == 0:
            off_jets = events[self._off_jet_name][:, :3]
            on_jets = events[self._on_jet_name][:, :3]
            
        elif self._on_off_ordering == 1:
            # simultaneously ordering leading and subleading jets
            two_leading_delta_r = events[self._off_jet_name][:, :2].metric_table(events[self._on_jet_name][:, :2])
            two_leading_sort_idx = ak.argmin(two_leading_delta_r, axis=2)
            mask = two_leading_sort_idx[:, 0] != two_leading_sort_idx[:, 1] # there is ambiguity if two indices are the same
            # apply mask
            two_leading_sort_idx = two_leading_sort_idx[mask]
            # propagate mask to events
            events = events[mask]
            # retrive offline jets
            off_jets = events[self._off_jet_name][:, :3]
            two_leading_on_jets = events[self._on_jet_name][:, :2][two_leading_sort_idx]
            on_jets = ak.concatenate([two_leading_on_jets, events[self._on_jet_name][:, 2:3]], axis=1)
            
        elif self._on_off_ordering == 2:
            # first, we consider the possibility that first and second online jets can be switched
            # we pick subleading online jet from either first or second jets which has lower delta r
            off_jets = events[self._off_jet_name][:, :3]
            leading_on_jets_idx = ak.argmin(off_jets[:, 0].delta_r(events[self._on_jet_name][:, :2]), axis=1)
            subleading_on_jets_idx = 1 - leading_on_jets_idx
            two_leading_sort_idx = ak.concatenate([ak.unflatten(leading_on_jets_idx, 1), ak.unflatten(subleading_on_jets_idx, 1)],
                                                  axis=1)
            two_leading_on_jets = events[self._on_jet_name][:, :2][two_leading_sort_idx]
            on_jets = ak.concatenate([two_leading_on_jets, events[self._on_jet_name][:, 2:3]], axis=1) # at third jet, if any
            
            # now consider possibility that second and third online jets can be switched
            # similarly, we pick subleading online jet from either second or third jets which has lower delta r 
            second_on_jets_idx = ak.argmin(off_jets[:, 1].delta_r(events[self._on_jet_name][:, 1:3]), axis=1)
            third_on_jets_idx = 1 - second_on_jets_idx
            second_on_jets_idx_unflatten = ak.unflatten(second_on_jets_idx, 1)
            second_third_idx = ak.concatenate([second_on_jets_idx_unflatten, ak.unflatten(third_on_jets_idx, 1)], axis=1)
            second_third_sort_idx = ak.where(ak.num(on_jets) == 2, second_on_jets_idx_unflatten, second_third_idx)
            on_jets = ak.concatenate([on_jets[:, 0:1], on_jets[:, 1:3][second_third_sort_idx]], axis=1)
            # we can also do the same for third and fourth jets...
            
        elif self._on_off_ordering == 3:
            off_jets = events[self._off_jet_name][:, :3]
            on_jets = off_jets.nearest(events[self._on_jet_name])
            
        else:
            raise ValueError("Invalid online pre-ordering with offline: {}".format(self._on_off_ordering))
        
        # now require that with this new ordering, leading and subleading are differed by at most max_deltaR, e.g. 0.2
        mask = (off_jets[:, :2].delta_r(on_jets[:, :2]) < self._max_deltaR)
        mask = mask[:, 0] & mask[:, 1]
        off_jets = off_jets[mask]
        on_jets = on_jets[mask]
        events = events[mask]
        
        # apply tag and probe
        off_jets_tag0, off_jets_probe0 = trigger_dijet_tag_and_probe(off_jets[:, 0], off_jets[:, 1], off_jets)
        on_jets_tag0, on_jets_probe0 = trigger_dijet_tag_and_probe(off_jets[:, 0], on_jets[:, 1], on_jets)
        off_jets_tag1, off_jets_probe1 = trigger_dijet_tag_and_probe(off_jets[:, 1], off_jets[:, 0], off_jets)
        on_jets_tag1, on_jets_probe1 = trigger_dijet_tag_and_probe(off_jets[:, 1], on_jets[:, 0], on_jets)
        
        # match tag, if requested
        if self._match_tag:
            off_jets_tag0, off_jets_probe0, on_jets_tag0, on_jets_probe0 \
                = match_offline_online_tag(off_jets_tag0, off_jets_probe0, on_jets_tag0, on_jets_probe0)
            off_jets_tag1, off_jets_probe1, on_jets_tag1, on_jets_probe1 \
                = match_offline_online_tag(off_jets_tag1, off_jets_probe1, on_jets_tag1, on_jets_probe1)
        
        # combine results from swap
        off_jets_tag = ak.concatenate([off_jets_tag0, off_jets_tag1], axis=1, mergebool=False)
        off_jets_probe = ak.concatenate([off_jets_probe0, off_jets_probe1], axis=1, mergebool=False)
        on_jets_tag = ak.concatenate([on_jets_tag0, on_jets_tag1], axis=1, mergebool=False)
        on_jets_probe = ak.concatenate([on_jets_probe0, on_jets_probe1], axis=1, mergebool=False)
        
        # save results
        events[self._off_jet_name + "_tag"] = off_jets_tag 
        events[self._on_jet_name + "_tag"] = on_jets_tag 
        #events[self._off_jet_name + "_probe"] = off_jets_probe
        #events[self._on_jet_name + "_probe"] = on_jets_probe
        events[self._off_jet_name] = off_jets_probe
        events[self._on_jet_name] = on_jets_probe
        
        return events
    
    def count(self, events):
        return np.sum((ak.num(events[self._off_jet_name]) > 0) | (ak.num(events[self._on_jet_name]) > 0))
    
    
### Jet Energy Correction ###
# Dummy Jets Factory # no correction
class NothingJetsFactory(object):
    def build(self, jets, lazy_cache):
        # simply copying fields
        jets["pt_orig"] = jets["pt"]
        jets["mass_orig"] = jets["mass"]
        jets["pt"] = jets["pt_raw"]
        jets["mass"] = jets["mass_raw"]
        jets["pt_jec"] = jets["pt"]
        jets["mass_jec"] = jets["mass"]
        jets["jet_energy_correction"] = jets["pt_jec"] / jets["pt_raw"]
        return jets
    
class JetEnergyCorrector(SelectorABC):
    def __init__(self, weight_filelist, jet_name, rho_name, verbose=0):
        super().__init__(weight_filelist)
        self._jet_name = jet_name
        self._rho_name = rho_name
        self._verbose = verbose

        # build jet factory
        if weight_filelist is not None:
            if len(weight_filelist) > 0:
                assert len(jet_name) and jet_name != None, "must provide unique name"
                assert rho_name != None, "must specify rho to use"
                ext = extractor()
                ext.add_weight_sets(self.build_weightsdesc(weight_filelist))
                ext.finalize()
                evaluator = ext.make_evaluator()

                jec_stack_names = evaluator.keys()
                jec_inputs = {name: evaluator[name] for name in jec_stack_names}
                jec_stack = JECStack(jec_inputs)

                if verbose > 1:
                    print("="*50)
                    print("JEC stack for {}:".format(name))
                    print("Jet Energy Scale:")
                    print(jec_stack._jec)
                    print("Jet Energy Scale Uncertainty:")
                    print(jec_stack._junc)
                    print("="*50)

                self.name_map = jec_stack.blank_name_map
                self.name_map['JetPt'] = 'pt'
                self.name_map['JetMass'] = 'mass'
                self.name_map['JetEta'] = 'eta'
                self.name_map['JetA'] = 'area'
                self.name_map['ptGenJet'] = 'pt_gen'
                self.name_map['ptRaw'] = 'pt_raw'
                self.name_map['massRaw'] = 'mass_raw'
                self.name_map['Rho'] = 'rho'

                self.jet_factory = CorrectedJetsFactory(self.name_map, jec_stack)
            else:
                self.name_map = None
                self.jet_factory = NothingJetsFactory()
    
    def __str__(self):
        return "{}: JEC".format(self._jet_name)
    
    def build_weightsdesc(self, filelist, local_names=None, names=None):
        def has_whitespace(s):
            return any([c in s for c in string.whitespace])
        assert any([~has_whitespace(filename) for filename in filelist]), "filename cannot contain whitespace"
        if names is None:
            names = ["*"] * len(filelist) # wildcard to import whole file
        if local_names is None:
            local_names = ["*"] * len(filelist) # wildcard
        assert any([~has_whitespace(name) for name in local_names]), "local name cannot contain whitespace"
        assert any([~has_whitespace(name) for name in names]), "name cannot contain whitespace"
        assert len(names) == len(filelist)
        assert len(local_names) == len(filelist)
        return [" ".join(_) for _ in zip(local_names, names, filelist)]
        
    def apply(self, events):
        jets = events[self._jet_name]
        # pt and mass will be corrected
        if "mass" not in jets.fields: # placeholder for jet mass
            jets["mass"] = -np.inf * ak.ones_like(jets["pt"])
            if self._verbose:
                warnings.warn("No Jet mass, set all Jet mass to -inf")
        #correction_level_in_use = set()
        # undo correction in NanoAOD
        if "rawFactor" in jets.fields:
            jets["pt_raw"] = (1 - jets["rawFactor"]) * jets["pt"]
            jets["mass_raw"] = (1 - jets["rawFactor"]) * jets["mass"]
            # indicate that raw and orig are different (for filling hist)
            # correction_level_in_use = {"raw", "orig"}
        else:
            if self._verbose > 0:
                warnings.warn("No rawFactor, treat as raw!")
            jets["pt_raw"] = jets["pt"]
            jets["mass_raw"] = jets["mass"]
            # indicate that raw and orig are same, and now refer as raw (for filling hist)
            # correction_level_in_use = {"raw"}
            
        # pt, eta, area, and rho are needed for JEC
        if self._rho_name:
            jets['rho'] = ak.broadcast_arrays(events[tuple(self._rho_name.split("_"))], jets.pt)[0]
        if "area" not in jets.fields: # placeholder for jet area
            jets["area"] = 0.5 * ak.ones_like(jets["pt"])
            if self._verbose > 0:
                warnings.warn("No Jet area, set all Jet area to 0.5")

        # additionally, gen pt is needed for JER (only for MC)
        try:
            jets['pt_gen'] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
        except:
            if self._verbose > 0:
                warnings.warn("No GenJet information needed for JER/JERSF")
                
        # apply JEC
        if not isinstance(self.jet_factory, NothingJetsFactory):
            correction_level_in_use.add("jec")
        jets = self.jet_factory.build(jets, lazy_cache=events.caches[0])
        events[self._jet_name] = jets
        
        return events#, correction_level_in_use

    def count(self, events):
        # by definition, JEC doesn't change number of events
        # here, we will count events with at least one jets instead, more useful for cutflow
        return np.sum(ak.num(events[self._jet_name]) > 0)

### delta R matching ###
class DeltaRMatching(SelectorABC):
    def __init__(self, max_deltaR, first_physics_object_name, second_physics_object_name, save_original=False):
        super().__init__(max_deltaR)
        self._max_deltaR = max_deltaR
        self._first_physics_object_name = first_physics_object_name
        self._second_physics_object_name = second_physics_object_name
        self._save_original = save_original
    
    def __str__(self):
        return "{} {}: delta R < {}".format(self._first_physics_object_name, self._second_physics_object_name, self._max_deltaR) 

    def apply(self, events):
        first_physics_object = events[self._first_physics_object_name]
        second_physics_object = events[self._second_physics_object_name]
        assert len(first_physics_object) == len(second_physics_object), "length of two physics objects must equal, but got {} and {}".format(len(first_physics_object), len(second_physics_object))
        
        if len(first_physics_object) == 0:
            return events
        if self._save_original:
            events[self._first_physics_object_name + "_unmatched"] = first_physics_object
            events[self._second_physics_object_name + "_unmatched"] = second_physics_object
            if self._first_physics_object_name + "_tag" in events.fields:
                events[self._tag_first_physics_object_name + "_unmatched"] = events[self._tag_first_physics_object_name]
            if self._second_physics_object_name + "_tag" in events.fields:  
                events[self._tag_second_physics_object_name + "_unmatched"] = events[self._tag_second_physics_object_name]
        
        matched = ak.cartesian([first_physics_object, second_physics_object])
        delta_R_one = matched.slot0.delta_r(matched.slot1) # compute delta r
        matched_mask = (delta_R_one < self._max_deltaR) # create mask
        if self._first_physics_object_name + "_tag" in events.fields \
            or self._second_physics_object_name + "_tag" in events.fields: # tag and probe applied
            assert self._first_physics_object_name + "_tag" in events.fields \
                and self._second_physics_object_name + "_tag" in events.fields, \
                "If there is tag jets, there must be both tag jets for offline and online jets"
            
            tag_first_physics_object = events[self._first_physics_object_name+"_tag"]
            tag_second_physics_object = events[self._second_physics_object_name+"_tag"]
            
            assert len(first_physics_object) == len(tag_first_physics_object), "length of tag and probe of first physics objects must equal, but got {} and {}".format(len(first_physics_object), len(tag_first_physics_object))
            assert len(second_physics_object) == len(tag_second_physics_object), "length of tag and probe of second physics objects must equal, but got {} and {}".format(len(second_physics_object), len(tag_second_physics_object))

            # zip to pair (probe, tag)
            tp_first_physics_object = ak.zip([first_physics_object, tag_first_physics_object]) 
            tp_second_physics_object = ak.zip([second_physics_object, tag_second_physics_object])
            tp_matched = ak.cartesian([tp_first_physics_object, tp_second_physics_object])
            tp_matched = tp_matched[matched_mask] # apply mask
            # unzip pair (probe, tag)
            events[self._first_physics_object_name], events[self._first_physics_object_name+"_tag"] = ak.unzip(tp_matched.slot0)
            events[self._second_physics_object_name], events[self._second_physics_object_name+"_tag"] = ak.unzip(tp_matched.slot1)
            
        else: # tag and probe applied
            matched = matched[matched_mask] # apply mask
            events[self._first_physics_object_name] = matched.slot0
            events[self._second_physics_object_name] = matched.slot1
        
        events = events[ak.num(events[self._first_physics_object_name]) > 0]
        return events
    
    def count(self, events):
        return np.sum(ak.num(events[self._first_physics_object_name]) > 0)
    
class PairwiseDeltaRMatching(SelectorABC):
    def __init__(self, physics_object_names, max_deltaR):
        super().__init__(max_deltaR)
        self._max_deltaR = max_deltaR
        self._physics_object_names = physics_object_names
    def __str__(self):
        return "{}: delta R < {}".format(" ".join(self._physics_object_names), self._max_deltaR) 
    def apply(self, events):
        num_physics_objects = len(self._physics_object_names)
        physics_objects = [events[name] for name in self._physics_object_names]
        for i in range(num_physics_objects):
            assert len(physics_objects[0]) == len(physics_objects[i]), "length of all physics objects must equal"
        matched = ak.cartesian(physics_objects)
        counts = ak.num(matched)
        mask = ak.unflatten(np.full(np.sum(counts), True), counts) # all True mask
        
        for first_slot, second_slot in itertools.combinations(map(str, range(num_physics_objects)), 2):
            mask = (mask & (matched[first_slot].delta_r(matched[second_slot]) < self._max_deltaR))
        matched = matched[mask] # apply mask
        for slot in range(num_physics_objects):
            events[self._physics_object_names[slot]] = matched[str(slot)]
        events = events[ak.num(events[self._physics_object_names[0]]) > 0]
        return events
    
    def count(self, events):
        return np.sum(ak.num(events[self._physics_object_names[0]]) > 0)
    
class SameBin(SelectorABC):
    def __init__(self, bins, first_physics_object_name, second_physics_object_name, physics_object_field):
        super().__init__(bins)
        self._bins = bins
        self._first_physics_object_name = first_physics_object_name
        self._second_physics_object_name = second_physics_object_name
        self._physics_object_field = physics_object_field

    def __str__(self):
        return "{} {}: Same {} bin".format(self._first_physics_object_name, self._second_physics_object_name, self._physics_object_field)
    
    def apply(self, events):
        first_physics_object = events[self._first_physics_object_name]
        second_physics_object = events[self._second_physics_object_name]
        first_physics_object_field = first_physics_object[self._physics_object_field]
        second_physics_object_field = second_physics_object[self._physics_object_field]
        assert all(ak.num(first_physics_object_field) == ak.num(second_physics_object_field)), "length of both physics objects must be the same in every event"
        counts = ak.num(first_physics_object_field)
        get_bin_idx = lambda arr: ak.unflatten(np.searchsorted(self._bins, ak.to_numpy(ak.flatten(arr))), counts)
        first_physics_object_field_bin_idx = get_bin_idx(first_physics_object_field)
        second_physics_object_field_bin_idx = get_bin_idx(second_physics_object_field)
        mask = (first_physics_object_field_bin_idx == second_physics_object_field_bin_idx)
        events[self._first_physics_object_name] = first_physics_object[mask]
        events[self._second_physics_object_name] = second_physics_object[mask]
        # propagate mask to tag jets
        if self._first_physics_object_name + "_tag" in events.fields:
            events[self._first_physics_object_name + "_tag"] = events[self._first_physics_object_name + "_tag"][mask]
        if self._second_physics_object_name + "_tag" in events.fields:
            events[self._second_physics_object_name + "_tag"] = events[self._second_physics_object_name + "_tag"][mask]
        return events
    
    def count(self, events):
        return ak.sum((ak.num(events[self._first_physics_object_name]) > 0))

class SameEtaBin(SameBin):
    def __init__(self, eta_bins, first_physics_object_name, second_physics_object_name):
        super().__init__(eta_bins, first_physics_object_name, second_physics_object_name, physics_object_field="eta")
        
class MultiPhysicsObjectSameBin(SelectorABC):
    def __init__(self, bins, physics_object_names, physics_object_field):
        super().__init__(bins is not None and len(physics_object_names) > 1)
        self._bins = bins
        self._physics_object_names = physics_object_names
        self._physics_object_field = physics_object_field

    def __str__(self):
        return "{}: Same {} bin".format(" ".join(self._physics_object_names), self._physics_object_field)
    
    def apply(self, events):
        num_physics_objects = len(self._physics_object_names)
        physics_objects_field = [events[name][self._physics_object_field] for name in self._physics_object_names]
        counts = ak.num(physics_objects_field[0])
        for i in range(1, num_physics_objects):
            assert ak.all((counts) == ak.num(physics_objects_field[i])), "length of both physics objects must be the same in every event"
        get_bin_idx = lambda arr: ak.unflatten(np.searchsorted(self._bins, ak.to_numpy(ak.flatten(arr))), counts) 
        physics_objects_field_bin_idx = [get_bin_idx(_) for _ in physics_objects_field]
        
        mask = (physics_objects_field_bin_idx[0] == physics_objects_field_bin_idx[1])
        for i in range(2, num_physics_objects):
            mask = mask & (physics_objects_field_bin_idx[0] == physics_objects_field_bin_idx[i])
            
        for physics_object_name in self._physics_object_names:
            events[physics_object_name] = events[physics_object_name][mask]
            # propagate mask to tag jets if any
            if physics_object_name + "_tag" in events.fields:
                events[physics_object_name + "_tag"] = events[physics_object_name + "_tag"][mask]
        events = events[ak.num(events[self._physics_object_names[0]]) > 0]
        return events
    
    def count(self, events):
        return ak.sum((ak.num(events[self._physics_object_names[0]]) > 0))
    
class MultiPhysicsObjectSameEtaBin(MultiPhysicsObjectSameBin):
    def __init__(self, eta_bins, physics_object_names):
        super().__init__(eta_bins, physics_object_names, physics_object_field="eta")
