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
        super().__init__()
        self._lumimask = LumiMask(lumi_json_path) if lumi_json_path else None
    def __str__(self):
        return "lumimask"
    def apply(self, events):
        if not self._lumimask:
            return events
        return self._lumimask(events.run, events.luminosityBlock)
    
class MinNPVGood(SelectorABC):
    def __init__(self, min_NPVGood=0):
        super().__init__()
        self._min_NPVGood = min_NPVGood
    def __str__(self):
        return "NPV > {}".format(self._min_NPVGood)
    def apply_min_NPVGood(self, events):
        return events[events.PV.npvsGood > self._min_NPVGood]
    apply = apply_min_NPVGood
    
class MaxPV_z(SelectorABC):
    def __init__(self, max_PV_z=24):
        super().__init__()
        self._max_PV_z = max_PV_z
    def __str__(self):
        return "PV |z| < {} cm".format(self._max_PV_z)
    def apply_max_PV_z(self, events):
        return events[np.abs(events.PV.z) < self._max_PV_z]
    apply = apply_max_PV_z
    
class MaxPV_rxy(SelectorABC):
    def __init__(self, max_PV_rxy=2):
        super().__init__()
        self._max_PV_rxy = max_PV_rxy
    def __str__(self):
        return "PV |r_xy| < {} cm".format(self._max_PV_rxy)
    def apply_max_PV_rxy(self, events):
        return events[np.sqrt(events.PV.x*events.PV.x + events.PV.y*events.PV.y) < self._max_PV_rxy]
    apply = apply_max_PV_rxy
    
class ClosePV_z(SelectorABC):
    def __init__(self, first_PV, second_PV, max_dz=None, sigma_multiple=None): # 0.2 for gen, 5 * sigma
        super().__init__()
        assert (max_dz == None and sigma_multiple == None) or (bool(max_dz) != bool(sigma_multiple)) , "Must pick either absolute max dz or multiple of z sigma"
        self._first_PV = first_PV
        self._second_PV = second_PV
        self._max_dz = max_dz
        self._sigma_multiple = sigma_multiple
        if first_PV == None or second_PV == None or (self._max_dz == None and self._sigma_multiple == None):
            self.off()
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
    apply_close_PV_z = apply

class MinPhysicsObject(SelectorABC):
    def __init__(self, physics_object_name, min_physics_object=0, name=""):
        super().__init__()
        self._physics_object_name = physics_object_name
        self._min_physics_object = min_physics_object
        self._name = name if name != None or len(name) > 0 else physics_object_name
    def __str__(self):
        return "{} >= {}".format(self._name, self._min_physics_object)
    def apply_min_physics_object(self, events):
        return events[(ak.num(events[self._physics_object_name]) >= self._min_physics_object)] 
    apply = apply_min_physics_object

class MinJet(MinPhysicsObject):
    def __init__(self, jet_name, min_jet=0, name=""):
        super().__init__(jet_name, min_jet, name)
    apply_min_jet = MinPhysicsObject.apply

class MinTrigger(SelectorABC):
    def __init__(self, trigger_type="single", trigger_min_pt=0, trigger_flag_prefix="PFJet", trigger_all_pts=None):
        super().__init__()
        if trigger_type in ["only", "lower_not", "upper_not"]:
            assert trigger_all_pts != None, "trigger type: {} need all trigger pts".format(trigger_type)
            
        self._trigger_min_pt = trigger_min_pt
        self._trigger_flag_prefix = trigger_flag_prefix
        self._trigger_all_pts = trigger_all_pts
        self._trigger_type = trigger_type
        
        # match-case only works with python >=3.10
#         match trigger_type:
#             case "None" | "none" | None:
#                 self.trigger_type = None
#                 self.comparison_operation = None
#             case "single":
#                 self.comparison_operation = None # can also set to function which always returns false
#             case "only":
#                 self.comparison_operation = (lambda pt, trigger_min_pt: pt != trigger_min_pt)
#             case "lower_not":
#                 self.comparison_operation = (lambda pt, trigger_min_pt: pt < trigger_min_pt)
#             case "upper_not":
#                 self.comparison_operation = (lambda pt, trigger_min_pt: pt > trigger_min_pt)
#             case _:
#                 raise ValueError("Invalid type of trigger cut")
        # LCG 102-103 use python 3.9.12
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
        return "trigger {}{} ({})".format(self._trigger_flag_prefix, self._trigger_min_pt, self._trigger_type)
    def apply_min_trigger(self, events):
        if (not self._trigger_type) or (not self._trigger_min_pt) or (self._trigger_min_pt <= 0):
            return events
        
        mask = events.HLT[self._trigger_flag_prefix + str(self._trigger_min_pt)] # single mask as base
        if self._trigger_all_pts != None and self._comparison_operation != None:
            # alternatively, setting trigger_all_pts = [trigger_min_pt] will give the same result
            for pt in self._trigger_all_pts:
                if self._comparison_operation(pt, self._trigger_min_pt):
                    flag = self._trigger_flag_prefix + str(pt)
                    mask = np.logical_and(mask, np.logical_not(events.HLT[flag]))
        return events[mask]
    apply = apply_min_trigger
    
class FlagFilter(SelectorABC):
    def __init__(self, flag_filter):
        super().__init__()
        self._flag_filter = flag_filter
    def __str__(self):
        return self._flag_filter
    def apply_flag_filter(self, events):
        if not self._flag_filter:
            return events
        return events[events.Flag[self._flag_filter]]
    apply = apply_flag_filter

class METFilter(FlagFilter):
    def __init__(self):
        super().__init__("METFilters")
    apply_met_filter = FlagFilter.apply

class MaxMET(SelectorABC):
    def __init__(self, max_MET, MET_type="MET"):
        super().__init__()
        self._MET_type = MET_type
        self._max_MET = max_MET
    def __str__(self):
        return "{} < {} GeV".format(self._MET_type, self._max_MET)
    def apply_max_MET(self, events):
        if not self._max_MET:
            return events
        return events[events[self._MET_type].pt < self._max_MET]
    apply = apply_max_MET
    
class MaxMET_sumET(SelectorABC):
    def __init__(self, max_MET_sumET, min_MET=0, MET_type="MET"):
        super().__init__()
        self._MET_type = MET_type
        self._max_MET_sumET = max_MET_sumET
        self._min_MET = min_MET
    def __str__(self):
        if self._min_MET <= 0:
            return "{}/sumET < {}".format(self._MET_type, self._max_MET_sumET)
        return "{} <= {} GeV or {}/sumET < {}".format(self._MET_type, self._min_MET, self._MET_type, self._max_MET_sumET)
    def apply_max_MET_sumET(self, events):
        if not self._max_MET_sumET:
            return events
        mask = (events[self._MET_type].pt < self._max_MET_sumET * events[self._MET_type].sumEt)
        if self._min_MET > 0:
            mask = mask | (events[self._MET_type].pt <= self._min_MET)
        return events[mask]
    apply = apply_max_MET_sumET

# wrap jet-level to event-level selector
class EventWrappedPhysicsObjectSelector(SelectorABC): 
    def __init__(self, physics_object_name, physics_object_selector, discard_empty=False):
        super().__init__()
        self._physics_object_name = physics_object_name
        self._physics_object_selector = physics_object_selector
        self._discard_empty = discard_empty
        self.on() if self._physics_object_selector.status else self.off() # copy initial status
    def __str__(self):
        return str(self._physics_object_selector)
    def apply_event_wrapped_physics_object_selector(self, events):
        # physics_object_selector.apply will ignore physics_object_selector's status
        # now this will use EventPhysicsObject's status during __call__
        physics_object = self._physics_object_selector.apply(events[self._physics_object_name])
        events[self._physics_object_name] = physics_object
        if self._discard_empty:
            events = events[ak.num(events[self._physics_object_name]) > 0]
        return events
    apply = apply_event_wrapped_physics_object_selector
    
### jet-level ###
### apply: physics_objects -> physics_objects, e.g. jets -> jets
class MinObject(SelectorABC):
    def __init__(self, min_object=0, name=""):
        super().__init__()
        assert len(name) and name != None, "must provide unique name"
        self._min_physics_object = min_physics_object
        self._name = name
    def __str__(self):
        return "{} >= {}".format(self._name, self._min_physics_object)
    def apply_min_object(self, physics_objects):
        count = ak.num(physics_objects)
        event_mask = (coutn >= self._object)
        mask = ak.unflatten(np.repeat(event_mask, count), count)
        return physics_objects[mask] 
    apply = apply_min_object

class ObjectInRange(SelectorABC):
    def __init__(self, field, min_value=-np.inf, max_value=np.inf, mirror=False, name=""):
        super().__init__()
        if field:
            assert len(name) > 0 and name != None, "must provide unique name"
        self._field = field
        self._min_value = min_value
        self._max_value = max_value
        self._mirror = mirror
        self._name = name
    def __str__(self):
        return "select {} {} in range ({}, {})".format(self._name, self._field, self._min_value, self._max_value)
    def apply_object_in_range(self, physics_object):
        if not self._field:
            return physics_object
        mask = (physics_object[self._field] > self._min_value) & (physics_object[self._field] < self._max_value)
        if self._mirror:
            mask = mask | ((physics_object[self._field] < -self._min_value) & (physics_object[self._field] > -self._max_value))
        return physics_object[mask]
    apply = apply_object_in_range

class ObjectInPtRange(ObjectInRange):
    def __init__(self, min_pt=-np.inf, max_pt=np.inf, name=""):
        super().__init__("pt", min_pt, max_pt, False, name)
    apply_object_in_pt_range = ObjectInRange.apply
    
class ObjectInEtaRange(ObjectInRange):
    def __init__(self, min_eta=-np.inf, max_eta=np.inf, mirror=False, name=""):
        super().__init__("eta", min_eta, max_eta, mirror, name)
    apply_object_in_eta_range = ObjectInRange.apply
        
class MaxLeadingObject(SelectorABC): #TODO: check when this should be applied actually. For T&P, this shouldn't matter
    def __init__(self, max_leading, name=""):
        super().__init__()
        if max_leading:
            assert len(name) and name != None, "must provide unique name"
        self._max_leading = max_leading
        self._name = name
    def __str__(self):
        return "{}: upto {} leading".format(self._name, self._max_leading)
    def apply_max_leading_object(self, physics_objects):
        if not self._max_leading:
            return physics_objects
        if len(physics_objects) == 0:
            return physics_objects
        return physics_objects[:, :self._max_leading]
    apply = apply_max_leading_object

class JetIdentification(SelectorABC):
    def __init__(self, jet_Id, name="", verbose=0):
        super().__init__()
        if jet_Id:
            assert len(name) and name != None, "must provide unique name"
            self._jet_Id = jet_Id
            jet_Id_bit_dict = {"loose":1, "tight":2, "tightleptonveto":4}
            if isinstance(jet_Id, str):
                jet_Id = jet_Id.split()
                if len(jet_Id) == 1:
                    jet_Id = jet_Id[0]
                    assert jet_Id.lower() in jet_Id_bit_dict.keys(), "cannot identify jet_Id {}".format(jet_Id)
                    self._jet_Id = jet_Id_bit_dict[jet_Id.lower()]
                else:
                    self._jet_Id = 0
                    for jet_id in jet_Id:
                        assert jet_id.lower() in jet_Id_bit_dict.keys(), "cannot identify jet_Id {}".format(jet_id)
                        self._jet_Id += jet_Id_bit_dict[jet_id.lower()] # bitwise-or in base 2 is add in base 10
            elif isinstance(jet_Id, int):
                assert 0 <= jet_Id <= 7, "jet_Id must be in [0, 7], but get {}".format(jet_Id)
            else:
                raise TypeError("expect jet_Id as int or str, but get {}".format(type(jet_Id)))
        else:
            self._jet_Id = None
        self._name = name
        self._verbose = verbose
    def __str__(self):
        return "{}: jet id {}".format(self._name, self._jet_Id)
    def apply_jet_identification(self, jets):
        if not self._jet_Id:
            return jets
        if not "jetId" in jets.fields:
            if self._verbose > 0:
                warnings.warn("cannot retrieve jetId, no jetId will be applied")
            return jets
        return jets[jets.jetId == self._jet_Id]
    apply = apply_jet_identification
    
class JetVetoMap(SelectorABC):
    def __init__(self, jet_veto_map_json_path, jet_veto_map_correction_name,
                 jet_veto_map_year, jet_veto_map_type="jetvetomap", name=""):
        super().__init__()
        if jet_veto_map_json_path:
            assert len(name) and name != None, "must provide unique name"
            corr = correctionlib.CorrectionSet.from_file(jet_veto_map_json_path)[jet_veto_map_correction_name]
            self._jet_veto_map = partial(corr.evaluate, jet_veto_map_year, jet_veto_map_type)
        else:
            self._jet_veto_map = None
            self.off()
        self._name = name
    def __str__(self):
        return "{}: jet veto map".format(self._name)
    def apply_jet_veto_map(self, jets):
        if not self._jet_veto_map:
            return jets
        # hard coded eta range
        jets = ObjectInEtaRange(-5.191, 5.191, mirror=False, name="|eta| < 5.191")(jets)
        # flatten and unflatten (this is required for correctionlib API)
        jets_flat = ak.flatten(jets)
        counts = ak.num(jets)
        jets_flat_phi = jets_flat.phi # wrap phi in (-pi, pi]
        jets_flat_phi = ak.where(jets_flat_phi <= -np.pi, 2*np.pi + jets_flat_phi, jets_flat_phi)
        jets_flat_phi = ak.where(jets_flat_phi > np.pi, -2*np.pi + jets_flat_phi, jets_flat_phi)
        mask_flat = (self._jet_veto_map(ak.to_numpy(jets_flat.eta), ak.to_numpy(jets_flat_phi)) == 0)
        mask = ak.unflatten(mask_flat, counts=counts)
        return jets[mask]
    apply = apply_jet_veto_map

### Other non standard selectors (not usual x -> smaller x)
### tag and probe ###
class TagAndProbeABC(SelectorABC):
    def __init__(self, swap=True, name=""):
        # in fact, tag_condition is subset of tag_probe_condition which is also subset of other_condition
        # this is just to make an explicit distinction
        super().__init__()
        self._swap = swap
        assert len(name) and name != None, "must provide unique name"
        self._name = name
    def __str__(self):
        return "{}: unnamed tag and probe".format(self._name)
    @abstractmethod
    def tag_condition(self, tag): 
        # return tag mask, i.e. what condition make an object passing tag
        raise NotImplementedError
    
    @abstractmethod
    def tag_probe_condition(self, tag, probe): 
        # return tag-probe mask, i.e. what condition make two objects are paired as tag and probe
        raise NotImplementedError
        
    def other_condition(self, tag, probe, others):
        return True
        
    def apply_tag_and_probe(self, tag, probe, others=None):
        assert len(tag) == len(probe), "length of tag and probe must equal, but get tag: {} and probe:{}".format(len(tag), len(probe))
        if len(tag) == 0:
            return tag, probe
        
        #tag = tag if tag.ndim == 2 else ak.unflatten(tag, counts=1)
        #probe = probe if probe.ndim == 2 else ak.unflatten(probe, counts=1)
        
        def make_tag_probe_mask(tag, probe):
            tag_mask = self.tag_condition(tag)
            probe_mask = self.tag_probe_condition(tag, probe)
            other_mask = self.other_condition(tag, probe, others)
            mask = tag_mask & probe_mask & other_mask
            return mask
        mask = make_tag_probe_mask(tag, probe)
        if self._swap:
            combine = lambda x, y: ak.concatenate([ak.unflatten(x, counts=1), ak.unflatten(y, counts=1)], 
                                                  axis=1, mergebool=False)
            swap_mask = make_tag_probe_mask(probe, tag)
            tag, probe = combine(tag, probe), combine(probe, tag)
            mask = combine(mask, swap_mask)
            return tag[mask], probe[mask]
        else:
            counts = ak.values_astype(mask, int)
            return ak.unflatten(tag[mask], counts=counts), ak.unflatten(probe[mask], counts=counts)
    apply = apply_tag_and_probe
    
    def __call__(self, tag, probe, others=None, cutflow=None):
        if self.status:
            tag, probe = self.apply(tag, probe, others)
        else:
            if self._swap:
                combine = lambda x, y: ak.concatenate([ak.unflatten(x, counts=1), ak.unflatten(y, counts=1)], 
                                                      axis=1, mergebool=False)
                tag, probe = combine(tag, probe), combine(probe, tag)
        if cutflow: 
            # to prevent confusion during delta R matching, number of events will not change
            # but for cutflow, it is more useful to count non-empty events
            cutflow[str(self)] += np.sum(ak.num(probe) > 0)
        return tag, probe

class TriggerDijetTagAndProbe(TagAndProbeABC):
    def __init__(self, tag_min_pt, max_alpha=None, swap=True, name=""):
        super().__init__(swap=swap, name=name)
        self._tag_min_pt = tag_min_pt
        self._max_alpha = max_alpha
        if not tag_min_pt:
            self.off()
    def __str__(self):
        return "{}: dijet tag and probe".format(self._name)
    def tag_condition(self, tag):
        tag_cut = (tag.pt >= self._tag_min_pt)
        return tag_cut
    
    def tag_probe_condition(self, tag, probe):
        opposite_cut = (np.abs(tag.phi - probe.phi) > 2.7)
        close_pt_cut = (np.abs(tag.pt - probe.pt) < 0.7 * (tag.pt + probe.pt))
                            # 0.5 * ak.max((tag.pt, probe.pt), axis=0))
        return opposite_cut & close_pt_cut
        
    def other_condition(self, tag, probe, jets):
        # alpha cut
        if self._max_alpha: # alpha = 2*jet3/(jet1 + jet2) <= 1, so if max_alpha > 1, then alpha_cut does nothing
            three_jets = jets[:, :3]
            alpha_cut = (2 * three_jets[:, -1].pt < self._max_alpha * (tag.pt + probe.pt))
            alpha_cut = alpha_cut | (ak.num(jets) == 2)
        return alpha_cut
    
    apply_trigger_dijet_tag_and_probe = TagAndProbeABC.apply_tag_and_probe
    
class OnlineOfflineDijetTagAndProbe(SelectorABC):
    def __init__(self, off_jet_name, on_jet_name, tag_min_pt, max_alpha=1.0):
        super().__init__()
        self._off_jet_name = off_jet_name
        self._on_jet_name = on_jet_name
        self._tag_min_pt = tag_min_pt
        self._max_alpha = max_alpha
        
        if off_jet_name is None:
            self.off()
    def __str__(self):
        return "T&P: {} and {}".format(self._off_jet_name, self._on_jet_name)
    def apply(self, events):
        mask = (ak.num(events[self._off_jet_name])>=2) & (ak.num(events[self._on_jet_name])>=2)
        events = events[mask]
        events[self._off_jet_name + "_orig"] = events[self._off_jet_name]
        events[self._on_jet_name + "_orig"] = events[self._on_jet_name]
        off_jets = events[self._off_jet_name][:, :4]
        on_jets = off_jets.nearest(events[self._on_jet_name])
        off_jets_tag, off_jets_probe = TriggerDijetTagAndProbe(swap=True, tag_min_pt=self._tag_min_pt, 
                                                               max_alpha=self._max_alpha, name="off")\
                                       .apply(off_jets[:, 0], off_jets[:, 1], off_jets)
                      
        on_jets_tag0, on_jets_probe0 = TriggerDijetTagAndProbe(swap=False, tag_min_pt=self._tag_min_pt, 
                                                               max_alpha=self._max_alpha, name="on0")\
                               .apply(off_jets[:, 0], on_jets[:, 1], on_jets)
        
        on_jets_tag1, on_jets_probe1 = TriggerDijetTagAndProbe(swap=False, tag_min_pt=self._tag_min_pt, 
                                                               max_alpha=self._max_alpha, name="on1")\
                               .apply(off_jets[:, 1], on_jets[:, 0], on_jets)
                                                       
        on_jets_tag = ak.concatenate([on_jets_tag0, on_jets_tag1], axis=1, mergebool=False)
        on_jets_probe = ak.concatenate([on_jets_probe0, on_jets_probe1], axis=1, mergebool=False)
        
        events[self._off_jet_name + "_tag"] = off_jets_tag 
        events[self._on_jet_name + "_tag"] = on_jets_tag 
        events[self._off_jet_name + "_probe"] = off_jets_probe
        events[self._on_jet_name + "_probe"] = on_jets_probe
        events[self._off_jet_name] = off_jets_probe
        events[self._on_jet_name] = on_jets_probe
        
        return events
    
### JEC block ###
### apply: jets, events -> jets
# Dummy Jets Factory # no correction
class NothingJetsFactory(object):
    def build(self, jets, lazy_cache):
        # simply copying fields
        jets["pt_orig"] = jets["pt"]
        jets["mass_orig"] = jets["mass"]
        jets["pt_jec"] = jets["pt"]
        jets["mass_jec"] = jets["mass"]
        jets["jet_energy_correction"] = jets["pt_jec"] / jets["pt_raw"]
        return jets
    
class JECBlock(SelectorABC): # TODO: think about better naming...
    def __init__(self, weight_filelist, rho_name, name="", verbose=0):
        super().__init__()
        # build jet factory
        if weight_filelist != None and len(weight_filelist) != 0:
            assert len(name) and name != None, "must provide unique name"
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
            if weight_filelist == None: # if weight_filelist = [], then it is on
                self.off()
        self._rho_name = rho_name
        self._name = name
        self._verbose = verbose
        
    def __str__(self):
        return "{}: JEC".format(self._name)
    
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
        
    def apply_JEC(self, jets, events):            
        # pt and mass will be corrected
        if "mass" not in jets.fields: # placeholder for jet mass
            jets["mass"] = -np.inf * ak.ones_like(jets["pt"])
            if self._verbose:
                warnings.warn("No Jet mass, set all Jet mass to -inf")
        correction_level_in_use = set()
        # undo correction in NanoAOD
        if "rawFactor" in jets.fields:
            jets["pt_raw"] = (1 - jets["rawFactor"]) * jets["pt"]
            jets["mass_raw"] = (1 - jets["rawFactor"]) * jets["mass"]
            # indicate that raw and orig are different (for filling hist)
            correction_level_in_use = {"raw", "orig"}
        else:
            if self._verbose > 0:
                warnings.warn("No rawFactor, treat as raw!")
            jets["pt_raw"] = jets["pt"]
            jets["mass_raw"] = jets["mass"]
            # indicate that raw and orig are same, and now refer as raw (for filling hist)
            correction_level_in_use = {"raw"}

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
        return jets, correction_level_in_use
    apply = apply_JEC
    
    def __call__(self, jets, events, cutflow=None):
        if self.status:
            jets, correction_level_in_use = self.apply(jets, events)
        else:
            jets["pt_orig"] = jets["pt"]
            correction_level_in_use = {"orig"}
        if cutflow: 
            # by definition, JEC doesn't change number of events
            # here, we will count events with at least one jets instead, more useful for cutflow
            if self.status:
                cutflow[str(self)] += np.sum(ak.num(jets) > 0)
            else:
                cutflow[str(self)+" (off)"] += np.sum(ak.num(jets) > 0)
        return jets, correction_level_in_use

### delta R matching ###
### apply: (physics_objects, physics_objects) -> (physics_objects, physics_objects), e.g. (jets, jets) -> (jets, jets)
class DeltaRMatching(SelectorABC):
    def __init__(self, max_deltaR):
        super().__init__()
        self._max_deltaR = max_deltaR
    def __str__(self):
        return "delta R < {}".format(self._max_deltaR) 
    def apply_deltaR_matching(self, first, second):
        assert len(first) == len(second), "length of two physics objects must equal, but got {} and {}".format(len(first), len(second))
        if len(first) == 0:
            return first, second
        matched = ak.cartesian([first, second])
        delta_R_one = matched.slot0.delta_r(matched.slot1) # compute delta r
        matched_mask = (delta_R_one < self._max_deltaR) # create mask
        matched = matched[matched_mask] # apply mask
        #matched = matched[ak.num(matched) > 0] # select only non-empty entries
        matched_first = matched.slot0
        matched_second = matched.slot1
        return matched_first, matched_second 
    apply = apply_deltaR_matching
    
    def __call__(self, first, second, cutflow=None):
        if self.status:
            first, second = self.apply(first, second)
        if cutflow:
            if self.status:
                cutflow[str(self)] += np.sum(ak.num(first) > 0)
            else:
                cutflow[str(self)+" (off)"] += np.sum(ak.num(first) > 0)
        return first, second
    
class PairwiseDeltaRMatching(SelectorABC):
    def __init__(self, max_deltaR):
        super().__init__()
        self._max_deltaR = max_deltaR
    def __str__(self):
        return "pair-wise delta R < {}".format(self._max_deltaR) 
    def apply_pairwise_deltaR_matching(self, arrays):
        if len(arrays) == 0:
            return arrays
        for array in arrays:
            assert len(array) == len(arrays[0]), "length of all physics objects must equal, but got {}".format(list(map(len, arrays)))
        if len(arrays[0]) == 0:
            return arrays
        matched = ak.cartesian(arrays)
        counts = ak.num(matched)
        mask = ak.unflatten(np.full(np.sum(counts), True), counts) # all True mask
        num_objects = len(arrays)
        for first_slot, second_slot in itertools.combinations(map(str, range(num_objects)), 2):
            mask = (mask & (matched[first_slot].delta_r(matched[second_slot]) < self._max_deltaR))
        matched = matched[mask] # apply mask
        return tuple([matched[slot] for slot in map(str, range(num_objects))])
    apply = apply_pairwise_deltaR_matching
    
    def __call__(self, arrays, cutflow=None):
        if self.status:
            arrays = self.apply(arrays)
        if cutflow:
            if self.status:
                cutflow[str(self)] += np.sum(ak.num(arrays[0]) > 0)
            else:
                cutflow[str(self)+" (off)"] += np.sum(ak.num(arrays[0]) > 0)
        return arrays
    
class SameBin(SelectorABC):
    def __init__(self, bins, first_object_name, second_object_name, object_field):
        super().__init__()
        self._bins = bins
        self._first_object_name = first_object_name
        self._second_object_name = second_object_name
        self._object_field = object_field
        if bins is None:
            self.off()
    def __str__(self):
        return "same bins"
    def apply(self, events):
        first_object = events[self._first_object_name]
        second_object = events[self._second_object_name]
        first_object_field = first_object[self._object_field]
        second_object_field = second_object[self._object_field]
        assert all(ak.num(first_object_field) == ak.num(second_object_field)), "length of both physics objects must be the same in every event"
        counts = ak.num(first_object_field)
        get_bin_idx = lambda arr: ak.unflatten(np.searchsorted(self._bins, ak.to_numpy(ak.flatten(arr))), counts)
        first_object_field_bin_idx = get_bin_idx(first_object_field)
        second_object_field_bin_idx = get_bin_idx(second_object_field)
        mask = (first_object_field_bin_idx == second_object_field_bin_idx)
        events[self._first_object_name] = first_object[mask]
        events[self._second_object_name] = second_object[mask]
        return events

class SameEtaBin(SameBin):
    def __init__(self, eta_bins, first_object_name, second_object_name):
        super().__init__(eta_bins, first_object_name, second_object_name, object_field="eta")