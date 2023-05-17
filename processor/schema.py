from coffea.nanoevents import NanoAODSchema

class JMENanoAODSchema(NanoAODSchema):
    """JMENano schema builder

    JMENano is an extended NanoAOD format that includes various jet collections down to low pt for JME studies
    More info at https://twiki.cern.ch/twiki/bin/viewauth/CMS/JMECustomNanoAOD
    Customization at https://github.com/nurfikri89/cmssw/blob/master/PhysicsTools/NanoAOD/python/custom_jme_cff.py
    """

    mixins = {
        **NanoAODSchema.mixins,
        "JetCHS": "Jet",
        "JetCalo": "Jet",
        #"JetPuppi": "Jet", # PUPPI is default "Jet" in JMENano, included here for completion
        "FatJetForJEC": "FatJet",
        "FatJetCHS": "FatJet",
        "TrigObjJMEAK4": "PtEtaPhiMCollection",
        "TrigObjJMEAK8": "PtEtaPhiMCollection"
    }
    
    all_cross_references = {
        **NanoAODSchema.all_cross_references,
        "JetCHS_genJetIdx": "GenJet",
        "JetCalo_genJetIdx": "GenJet",
        #"JetPuppi_genJetIdx": "GenJet", # PUPPI is default "Jet" in JMENano, included here for completion
        "FatJetForJEC_genJetIdx": "GenJetAK8ForJEC",
        "FatJetCHS_genJetIdx": "GenJetAK8ForJEC",
    }

class ScoutingJMENanoAODSchema(JMENanoAODSchema):
    
    mixins = {
        **JMENanoAODSchema.mixins,
        "ScoutingJet": "Jet",
        "ScoutingFatJet": "FatJet",
        "ScoutingRho": "Rho"
    }
#     all_cross_references = {
#         **JMENanoAODSchema.all_cross_references,
#         "ScoutingJet_genJetIdx": "GenJet",
#         "ScoutingFatJet_genJetAK8Idx": "GenJetAK8ForJEC",
#     }