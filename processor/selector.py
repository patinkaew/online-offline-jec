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
import string
import warnings

### event-level ###
### apply: events -> events
class MinNPVGood(SelectorABC):
    def __init__(self, min_NPVGood=0):
        super().__init__()
        self.min_NPVGood = min_NPVGood
    def apply_min_NPVGood(self, events):
        return events[events.PV.npvsGood > self.min_NPVGood]
    apply = apply_min_NPVGood
    
class MaxPV_z(SelectorABC):
    def __init__(self, max_PV_z=24):
        super().__init__()
        self.max_PV_z = max_PV_z
    def apply_max_PV_z(self, events):
        return events[np.abs(events.PV.z) < self.max_PV_z]
    apply = apply_max_PV_z
    
class MaxPV_rxy(SelectorABC):
    def __init__(self, max_PV_rxy=2):
        super().__init__()
        self.max_PV_rxy = max_PV_rxy
    def apply_max_PV_rxy(self, events):
        return events[np.sqrt(events.PV.x*events.PV.x + events.PV.y*events.PV.y) < self.max_PV_rxy]
    apply = apply_max_PV_rxy

class MinPhysicsObject(SelectorABC):
    def __init__(self, physics_object_name, min_physics_object=0):
        super().__init__()
        self.physics_object_name = physics_object_name
        self.min_physics_object = min_physics_object
    def apply_min_physics_object(self, events):
        return events[(ak.num(events[self.physics_object_name]) >= self.min_physics_object)] 
    apply = apply_min_physics_object

class MinTrigger(SelectorABC):
    def __init__(self, trigger_type="single", trigger_min_pt=0, trigger_flag_prefix="PFJet", trigger_all_pts=None):
        super().__init__()
        if trigger_type in ["only", "lower_not", "upper_not"]:
            assert trigger_all_pts != None, "trigger type: {} need all trigger pts".format(trigger_type)
            
        self.trigger_min_pt = trigger_min_pt
        self.trigger_flag_prefix = trigger_flag_prefix
        self.trigger_all_pts = trigger_all_pts
        self.trigger_type = trigger_type
        
        # match-case only works with python >=3.10
        match trigger_type:
            case "None" | "none" | None:
                self.trigger_type = None
                self.comparison_operation = None
            case "single":
                self.comparison_operation = None # can also set to function which always returns false
            case "only":
                self.comparison_operation = (lambda pt, trigger_min_pt: pt != trigger_min_pt)
            case "lower_not":
                self.comparison_operation = (lambda pt, trigger_min_pt: pt < trigger_min_pt)
            case "upper_not":
                self.comparison_operation = (lambda pt, trigger_min_pt: pt > trigger_min_pt)
            case _:
                raise ValueError("Invalid type of trigger cut")

    def apply_min_trigger(self, events):
        if (not self.trigger_type) or (not self.trigger_min_pt) or (self.trigger_min_pt <= 0):
            return events
        
        mask = events.HLT[self.trigger_flag_prefix + str(self.trigger_min_pt)] # single mask as base
        if self.trigger_all_pts != None and self.comparison_operation != None:
            # alternatively, setting trigger_all_pts = [trigger_min_pt] will give the same result
            for pt in self.trigger_all_pts:
                if self.comparison_operation(pt, self.trigger_min_pt):
                    flag = self.trigger_flag_prefix + str(pt)
                    mask = np.logical_and(mask, np.logical_not(events.HLT[flag]))
        return events[mask]
    apply = apply_min_trigger
    
class FlagFilter(SelectorABC):
    def __init__(self, flag_filter):
        super().__init__()
        self.flag_filter = flag_filter
    def apply_flag_filter(self, events):
        if not self.flag_filter:
            return events
        return events[events.Flag[self.flag_filter]]
    apply = apply_flag_filter

class METFilter(FlagFilter):
    def __init__(self):
        super().__init__("METFilters")
    apply_met_filter = FlagFilter.apply_flag_filter

class MaxMET(SelectorABC):
    def __init__(self, max_MET, MET_type="MET"):
        super().__init__()
        self.MET_type = MET_type
        self.max_MET = max_MET
    def apply_max_MET(self, events):
        if not self.max_MET:
            return events
        return events[events[self.MET_type].pt < self.max_MET]
    apply = apply_max_MET

class MaxMET_sumET_old(SelectorABC):
    def __init__(self, max_MET_sumET, MET_type="MET"):
        super().__init__()
        self.MET_type = MET_type
        self.max_MET_sumET = max_MET_sumET
    def apply_max_MET_sumET(self, events):
        if not self.max_MET_sumET:
            return events
        return events[events[self.MET_type].pt < self.max_MET_sumET * events[self.MET_type].sumEt]
    apply = apply_max_MET_sumET
    
class MaxMET_sumET_above(SelectorABC):
    def __init__(self, max_MET_sumET, min_MET=0, MET_type="MET"):
        super().__init__()
        self.MET_type = MET_type
        self.max_MET_sumET = max_MET_sumET
        self.min_MET = min_MET
    def apply_max_MET_sumET(self, events):
        if not self.max_MET_sumET:
            return events
        mask = (events[self.MET_type].pt < self.max_MET_sumET * events[self.MET_type].sumEt)
        if self.min_MET > 0:
            mask = mask | (events[self.MET_type].pt <= self.min_MET)
        return events[mask]
    apply = apply_max_MET_sumET

# wrap jet-level to event-level selector
class EventWrappedPhysicsObjectSelector(SelectorABC): 
    def __init__(self, physics_object_name, physics_object_selector, discard_empty=False):
        super().__init__()
        self.physics_object_name = physics_object_name
        self.physics_object_selector = physics_object_selector
        self.discard_empty = discard_empty
    def apply_event_wrapped_physics_object_selector(self, events):
        # physics_object_selector.apply will ignore physics_object_selector's status
        # now this will use EventPhysicsObject's status during __call__
        physics_object = self.physics_object_selector.apply(events[self.physics_object_name])
        events[self.physics_object_name] = physics_object
        if self.discard_empty:
            events = events[ak.num(events[self.physics_object_name]) > 0]
        return events
    apply = apply_event_wrapped_physics_object_selector
    
### jet-level ###
### apply: physics_objects -> physics_objects, e.g. jets -> jets
class MaxLeadingObject(SelectorABC):
    def __init__(self, max_leading=None):
        super().__init__()
        self.max_leading = max_leading
    def apply_max_leading_object(self, physics_objects):
        if not self.max_leading:
            return physics_objects
        if len(physics_objects) == 0:
            return physics_objects
        return physics_objects[:, :self.max_leading]
    apply = apply_max_leading_object

class JetIdentification(SelectorABC):
    def __init__(self, jet_Id, verbose):
        super().__init__()
        if jet_Id:
            self.jet_Id = jet_Id
            jet_Id_bit_dict = {"loose":1, "tight":2, "tightleptonveto":4}
            if isinstance(jet_Id, str):
                jet_Id = jet_Id.split()
                if len(jet_Id) == 1:
                    jet_Id = jet_Id[0]
                    assert jet_Id.lower() in jet_Id_bit_dict.keys(), "cannot identify jet_Id {}".format(jet_Id)
                    self.jet_Id = jet_Id_bit_dict[jet_Id.lower()]
                else:
                    self.jet_Id = 0
                    for jet_id in jet_Id:
                        assert jet_id.lower() in jet_Id_bit_dict.keys(), "cannot identify jet_Id {}".format(jet_id)
                        self.jet_Id += jet_Id_bit_dict[jet_id.lower()]
            elif isinstance(jet_Id, int):
                assert 0 <= jet_Id <= 7, "jet_Id must be in [0, 7], but get {}".format(jet_Id)
            else:
                raise TypeError("expect jet_Id as int or str, but get {}".format(type(jet_Id)))
        else:
            self.jet_Id = None
            self.off()
        self.verbose = verbose
    def apply_jet_identification(self, jets):
        if not self.jet_Id:
            return jets
        if not "jetId" in jets.fields:
            if self.verbose > 0:
                warnings.warn("cannot retrieve jetId, no jetId will be applied")
            return jets
        return jets[jets.jetId == self.jet_Id]
    apply = apply_jet_identification
    
class JetVetoMap(SelectorABC):
    def __init__(self, jet_veto_map_json_path, jet_veto_map_correction_name,
                 jet_veto_map_year, jet_veto_map_type="jetvetomap"):
        super().__init__()
        if jet_veto_map_json_path:
            corr = correctionlib.CorrectionSet.from_file(jet_veto_map_json_path)[jet_veto_map_correction_name]
            self.jet_veto_map = partial(corr.evaluate, jet_veto_map_year, jet_veto_map_type)
        else:
            self.jet_veto_map = None
            self.off()
        
    def apply_jet_veto_map(self, jets):
        if not self.jet_veto_map:
            return jets
        # flatten and unflatten (this is required for correctionlib API)
        jets_flat = ak.flatten(jets)
        counts = ak.num(jets)
        jets_flat_phi = jets_flat.phi # wrap phi in (-pi, pi]
        jets_flat_phi = ak.where(jets_flat_phi <= -np.pi, 2*np.pi + jets_flat_phi, jets_flat_phi)
        jets_flat_phi = ak.where(jets_flat_phi > np.pi, -2*np.pi + jets_flat_phi, jets_flat_phi)
        mask_flat = (self.jet_veto_map(jets_flat.eta, jets_flat_phi) == 0)
        mask = ak.unflatten(mask_flat, counts=counts)
        return jets[mask]
    apply = apply_jet_veto_map

### Other no standard selectors (not x -> smaller x)
### tag and probe ###
class TagAndProbeABC(SelectorABC):
    def __init__(self, swap=True):
        # in fact, tag_condition is subset of tag_probe_condition which is also subset of other_condition
        # this is just to make an explicit distinction
        super().__init__()
        self.swap = swap
        
    @abstractmethod
    def tag_condition(self, tag):
        raise NotImplementedError
    
    @abstractmethod
    def tag_probe_condition(self, tag, probe):
        raise NotImplementedError
        
    def other_condition(self, tag, probe, others):
        return True
        
    def apply_tag_and_probe(self, tag, probe, others=None):
        assert len(tag) == len(probe), "length of tag and probe must equal, but get tag: {} and probe:{}".format(len(tag), len(probe))
        if len(tag) == 0:
            return tag, probe
        
        def make_tag_probe_mask(tag, probe):
            tag_mask = self.tag_condition(tag)
            probe_mask = self.tag_probe_condition(tag, probe)
            other_mask = self.other_condition(tag, probe, others)
            mask = tag_mask & probe_mask & other_mask
            return mask
        
        mask = make_tag_probe_mask(tag, probe)
        if self.swap:
            combine = lambda x, y: ak.concatenate([ak.unflatten(x, counts=1), ak.unflatten(y, counts=1)], 
                                                  axis=1, mergebool=False)
            swap_mask = make_tag_probe_mask(probe, tag)
            tag, probe = combine(tag, probe), combine(probe, tag)
            mask = combine(mask, swap_mask)
        return tag[mask], probe[mask]
    apply = apply_tag_and_probe
    
    def __call__(self, tag, probe, others=None):
        if self.status:
            return self.apply(tag, probe, others)
        else:
            if self.swap:
                combine = lambda x, y: ak.concatenate([ak.unflatten(x, counts=1), ak.unflatten(y, counts=1)], 
                                                      axis=1, mergebool=False)
                tag, probe = combine(tag, probe), combine(probe, tag)
            return tag, probe

class TriggerDijetTagAndProbe(TagAndProbeABC):
    def __init__(self, tag_min_pt, max_alpha=None, swap=True):
        super().__init__(swap=swap)
        self.tag_min_pt = tag_min_pt
        self.max_alpha = max_alpha
        if not tag_min_pt:
            self.off()
            
    def tag_condition(self, tag):
        tag_cut = (tag.pt >= self.tag_min_pt)
        return tag_cut
    
    def tag_probe_condition(self, tag, probe):
        opposite_cut = (np.abs(tag.phi - probe.phi) > 2.7)
        close_pt_cut = (np.abs(tag.pt - probe.pt) < 0.7 * (tag.pt + probe.pt))
                            # 0.5 * ak.max((tag.pt, probe.pt), axis=0))
        return opposite_cut & close_pt_cut
        
    def other_condition(self, tag, probe, jets):
        # alpha cut
        if self.max_alpha: # alpha = 2*jet3/(jet1 + jet2) <= 1, so if max_alpha > 1, then alpha_cut does nothing
            three_jets = jets[:, :3]
            alpha_cut = (2 * three_jets[:, -1].pt < self.max_alpha * (tag.pt + probe.pt))
            alpha_cut = alpha_cut | (ak.num(jets) == 2)
        return alpha_cut
    
    apply_trigger_dijet_tag_and_probe = TagAndProbeABC.apply_tag_and_probe
    
### JEC block ###
### apply: jets, events -> jets
# Dummy Jets Factory # no correction
class NothingJetsFactory(object):
    def build(self, jets, lazy_cache):
        # simply copying fields
        jets["pt_orig"] = jets["pt"]
        jets["mass_orig"] = jets["mass"]
        jets["pt_jec"] = jets["pt_raw"]
        jets["mass_jec"] = jets["mass_raw"]
        jets["jet_energy_correction"] = jets["pt_jec"] / jets["pt_raw"]
        return jets
    
class JECBlock(SelectorABC): # TODO: think about better naming...
    def __init__(self, weight_filelist, verbose=0):
        super().__init__()
        # build jet factory
        if weight_filelist is not None and len(weight_filelist) != 0:
            ext = extractor()
            ext.add_weight_sets(self.build_weightsdesc(weight_filelist))
            ext.finalize()
            evaluator = ext.make_evaluator()

            jec_stack_names = evaluator.keys()

            jec_inputs = {name: evaluator[name] for name in jec_stack_names}
            jec_stack = JECStack(jec_inputs)

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
            
        self.verbose = verbose
        
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
            if self.verbose:
                warnings.warn("No Jet mass, set all Jet mass to -inf")
                
        # undo correction in NanoAOD
        if "rawFactor" in jets.fields:
            jets["pt_raw"] = (1 - jets["rawFactor"]) * jets["pt"]
            jets["mass_raw"] = (1 - jets["rawFactor"]) * jets["mass"]
        else:
            if self.verbose > 0:
                warnings.warn("No rawFactor, treat as raw!")
            jets["pt_raw"] = jets["pt"]
            jets["mass_raw"] = jets["mass"]

        # pt, eta, area, and rho are needed for JEC
        jets['rho'] = ak.broadcast_arrays(events.Rho.fixedGridRhoFastjetAll, jets.pt)[0]
        if "area" not in jets.fields: # placeholder for jet area
            jets["area"] = -np.inf * ak.ones_like(jets["pt"])
            if self.verbose > 0:
                warnings.warn("No Jet area, set all Jet area to -inf")
        
        # additionally, gen pt is needed for JER (only for MC)
        try:
            jets['pt_gen'] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
        except:
            if self.verbose > 0:
                warnings.warn("No GenJet information needed for JER/JERSF")
                
        # apply JEC
        return self.jet_factory.build(jets, lazy_cache=events.caches[0])
    apply = apply_JEC
    
    def __call__(self, jets, events):
        if self.status:
            return self.apply(jets, events)
        else:
            return jets

### delta R matching ###
### apply: (physics_objects, physics_objects) -> (physics_objects, physics_objects), e.g. (jets, jets) -> (jets, jets)
class DeltaRMatching(SelectorABC):
    def __init__(self, max_deltaR):
        super().__init__()
        self.max_deltaR = max_deltaR
        
    def apply_deltaR_matching(self, first, second):
        assert len(first) == len(second), "length of two physics objects must equal, but get {} and {}".format(len(first), len(second))
        if len(first) == 0:
            return first, second, 0
        matched = ak.cartesian([first, second])
        delta_R_one = matched.slot0.delta_r(matched.slot1) # compute delta r
        matched_mask = (delta_R_one < self.max_deltaR) # create mask
        matched = matched[matched_mask] # apply mask
        matched = matched[ak.num(matched) > 0] # select only non-empty entries
        matched_first = matched.slot0
        matched_second = matched.slot1
        
        # apply match mask to event to compute cutflow
        event_matched_mask = map(lambda x: any(x), matched_mask) # make event-level mask
        matched_count = ak.sum(event_matched_mask)
        
        return matched_first, matched_second, matched_count  
    apply = apply_deltaR_matching
    
    def __call__(self, first, second):
        if self.status:
            return self.apply(first, second)
        else:
            return first, second, len(first)