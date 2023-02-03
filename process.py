import argparse
import configparser
import glob
import json
import datetime
from functools import partial
import sys
import os

from coffea import processor
from coffea import util as cutil

from dask.distributed import Client 
from dask_lxplus import CernCluster
import socket

from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from processor.processor import OHProcessor
from processor.schema import JMENanoAODSchema
from util import *

def get_default_dataset_name(data_dir):
    end_index = data_dir.rfind("/")
    if end_index +1 == len(data_dir):
        end_index = data_dir[:-1].rfind("/")
    start_index = data_dir[:end_index].rfind("/")
    return data_dir[start_index+1: end_index]
    
def get_filelist(data_dir):
    return glob.glob(os.path.join(data_dir, "*/*/*"))

def build_fileset(data_dir, dataset_names=None):
    if dataset_names is not None:
        assert len(data_dir) == len(dataset_names)
        dataset_names = [dataset_name if dataset_name != "*" else get_default_dataset_name(data_dir[i]) 
                         for i, dataset_name in enumerate(dataset_names)]
    else:
        dataset_names = map(get_default_dataset_name, data_dir)
    
    filelist = map(get_filelist, data_dir)
    return dict(zip(dataset_names, filelist))

def build_processor_config(args, configs):
    def try_from_args(field, default=None, apply_eval=False):
        arg_value = vars(args)[field]
        if arg_value is not None:
            print("Overwrite {} from {} to {}".format(field, processor_section.get(field), arg_value))
            processor_config[field] = arg_value
            configs._sections["Processor"][field] = str(arg_value)
        else:
            if default:
                processor_config[field] = processor_section.get(field, default)
            else:
                processor_config[field] = processor_section.get(field)
                
            if apply_eval:
                processor_config[field] = eval(processor_config[field])
        
    processor_config = dict()
    processor_section = configs["Processor"]
    processor_config["off_jet_name"] = processor_section.get("off_jet_name", "Jet")
    processor_config["off_jet_label"] = processor_section.get("off_jet_label")
    processor_config["on_jet_name"] = processor_section.get("on_jet_name", "TrigObjJMEAK4")
    processor_config["on_jet_label"] = processor_section.get("on_jet_label")
    
    # luminosity 
    processor_config["lumi_json_path"] = processor_section.get("lumi_json_path")
    processor_config["lumi_csv_path"] = processor_section.get("lumi_csv_path")
    
    # event-level selections
    processor_config["flag_filters"] = processor_section.get("flag_filters", "None")
    
    processor_config["min_off_jet"] = processor_section.getint("min_off_jet", 0)
    processor_config["min_on_jet"] = processor_section.getint("min_on_jet", 0)
    
    processor_config["MET_type"] = processor_section.get("MET_type", "MET")
    processor_config["max_MET"] = eval(processor_section.get("max_MET", "None"))
    processor_config["max_MET_sumET"] = eval(processor_section.get("max_MET_sumET", "None"))
    
    try_from_args("trigger_min_pt", default="0", apply_eval=True)     
    try_from_args("trigger_type", default="None", apply_eval=False)
    
    processor_config["trigger_flag_prefix"] = processor_section.get("trigger_flag_prefix", "PFJet")
    processor_config["trigger_all_pts"] = eval(processor_section.get("trigger_all_pts", \
                                                                     "[40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]"))
    
    # jet-level selections
    try:
        processor_config["off_jet_Id"] = processor_section.getint("off_jet_Id")
    except:
        processor_config["off_jet_Id"] = processor_section.get("off_jet_Id", "None")
    try:
        processor_config["on_jet_Id"] = processor_section.getint("on_jet_Id")
    except:
        processor_config["on_jet_Id"] = processor_section.get("on_jet_Id", "None")
        
    processor_config["off_jet_veto_map_json_path"] = processor_section.get("off_jet_veto_map_json_path", "None")
    processor_config["on_jet_veto_map_json_path"] = processor_section.get("on_jet_veto_map_json_path", "None")
    processor_config["off_jet_veto_map_correction_name"] = processor_section.get("off_jet_veto_map_correction_name", "None")
    processor_config["on_jet_veto_map_correction_name"] = processor_section.get("on_jet_veto_map_correction_name", "None")
    processor_config["off_jet_veto_map_year"] = processor_section.get("off_jet_veto_map_year", "None")
    processor_config["on_jet_veto_map_year"] = processor_section.get("on_jet_veto_map_year", "None")
    processor_config["off_jet_veto_map_type"] = processor_section.get("off_jet_veto_map_type", "None")
    processor_config["on_jet_veto_map_type"] = processor_section.get("on_jet_veto_map_type", "None")  
    
    processor_config["off_jet_weight_filelist"] = eval(processor_section.get("off_jet_weight_filelist", "jetvetomap"))
    processor_config["on_jet_weight_filelist"] = eval(processor_section.get("on_jet_weight_filelist", "jetvetomap"))
    
    try_from_args("off_jet_tag_probe", default="True", apply_eval=True)
    try_from_args("on_jet_tag_probe", default="True", apply_eval=True)
    try_from_args("off_jet_tag_min_pt", default="0", apply_eval=True)
    try_from_args("on_jet_tag_min_pt", default="0", apply_eval=True)
    
    processor_config["max_leading_jet"] = eval(processor_section.get("max_leading_jet", "None"))
    
    # histogram
    processor_config["storage"] = eval(processor_section.get("storage"))
    
    # printing
    processor_config["verbose"] = processor_section.getint("verbose")
    
    for key in processor_config:
        processor_config[key] = processor_config[key] if processor_config[key] != "None" else None
    
    return processor_config

def print_num_inputfiles(fileset):
    print("="*50)
    print("Number of input files to be processed")
    [print("{} : {}".format(key, len(value))) for key, value in fileset.items()]
    print("Total : {}".format(sum(map(len, fileset.values()))))
    print("="*50)

if __name__ == "__main__":
    # parsing arguments and configurations
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, nargs="+", required=True)
    parser.add_argument("--dataset_name", type=str, nargs="+", required=False)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--trigger_type", type=str, required=False)
    parser.add_argument("--trigger_min_pt", type=float, required=False)
    parser.add_argument('--off_jet_tag_probe', action=argparse.BooleanOptionalAction, required=False)
    parser.add_argument('--on_jet_tag_probe', action=argparse.BooleanOptionalAction, required=False)
    parser.add_argument("--off_jet_tag_min_pt", type=float, required=False)
    parser.add_argument("--on_jet_tag_min_pt", type=float, required=False)
    
    args = parser.parse_args()
    print_dict_json(vars(args), title="Arguments")
    
    configs = configparser.ConfigParser()
    configs.read(args.config_file)
    
    print("="*50)
    print("Process configuration file to processor (Priority args > configs)")
    processor_config = build_processor_config(args, configs)
    print("="*50)
    
    # change to list for printing
    configs._sections["Processor"]["off_jet_weight_filelist"] = eval(configs._sections["Processor"]["off_jet_weight_filelist"])
    configs._sections["Processor"]["on_jet_weight_filelist"] = eval(configs._sections["Processor"]["on_jet_weight_filelist"])
    print_dict_json(configs._sections, title="Configurations")
    
    # build fileset
    fileset = build_fileset(args.input_dir, args.dataset_name)
    max_file = 10 # for testing
    if max_file is not None:
        for dataset in fileset:
            fileset[dataset] = sorted(fileset[dataset])[:max_file]
    print_num_inputfiles(fileset)
    
#     p = OHProcessor(**processor_config)
#     data_dir = "/data/data/JME_NANO_DATA/2022/JMENanoRun3_v2p1_Run2022D-PromptReco-v2/JetMET/220915_173253/0000/"
#     fname = "tree_155.root"
#     events_data = NanoEventsFactory.from_root(
#                 os.path.join(data_dir, fname), 
#                 schemaclass=JMENanoAODSchema,
#                 metadata={"dataset": "test"}
#                 ).events()
#     mc_dir = "/data/data/JME_NANO_MC/2022/QCD_Pt-15to7000_TuneCP5_Flat_13p6TeV_pythia8/JMENanoRun3_v2p1_MC22_122/220915_171347/0000/"
#     mc_fname = "tree_1.root"
#     events_mc = NanoEventsFactory.from_root(
#             os.path.join(mc_dir, mc_fname), 
#             schemaclass=JMENanoAODSchema,
#             metadata={"dataset": "tree_1"}
#             ).events()
#     print("trying data")
#     p.process(events_data)
#     print("trying mc")
#     p.process(events_mc)
#     exit()
      
    # define runner
#     executor = processor.IterativeExecutor(compression=None)
#     if configs["Runner"].getboolean("parallel", False):
#         executor = processor.FuturesExecutor(compression=None, workers=configs["Runner"].getint("num_workers", 1))
#     runner = processor.Runner(
#                     executor=executor,
#                     schema=JMENanoAODSchema,
#                     # size of each chunk to process (a unit of work), default to 100000
#                     # approximately, grow linearly with memory usage
#                     chunksize=configs["Runner"].getint("chunksize", 100_000),

#                     # number of maximum chunks to process in each dataset, default to whole dataset
#                     # do not set this when running the full analysis
#                     # set this when testing
#                     maxchunks=eval(configs["Runner"].get("maxchunks", "None")),
#                     )
#         # processing
#     print("="*50)
#     print("Begin Processing")
#     print("(Save file: {})".format(args.out_file))
#     mkdir_if_not_exists(os.path.dirname(args.out_file))
#     print("="*50)
#     start_time = datetime.datetime.now()
#     out = runner(fileset, treename="Events", processor_instance=OHProcessor(**processor_config))
#     end_time = datetime.datetime.now()
#     elapsed_time = end_time-start_time
#     print("="*50)
    
#     print("Finish Processing")
#     print("Elapsed time: {:.3f} s".format(elapsed_time.total_seconds()))
#     print_dict_json(out.get("cutflow", dict()), title="Cutflow")
    
#     # post-processing output
#     out["arguments"] = vars(args)
#     out["configurations"] = configs._sections
#     out["start_timestamp"] = start_time.strftime("%d/%m/%Y, %H:%M:%S")
#     out["process_time"] = elapsed_time
#     print("="*50)
    
#     print("Save to Output file: {}".format(args.out_file))
#     cutil.save(out, args.out_file)
#     print("All Complete!")
    
#     espresso     = 20 minutes
#     microcentury = 1 hour
#     longlunch    = 2 hours
#     workday      = 8 hours
#     tomorrow     = 1 day
#     testmatch    = 3 days
#     nextweek     = 1 week
    
    print("current interpreter: {}".format(sys.executable))
    
    # configure for grid authentication
    proxy_path = "/afs/cern.ch/user/p/pinkaew/private/gridproxy.pem"
    os.environ['X509_USER_PROXY'] = proxy_path
    if os.path.isfile(os.environ['X509_USER_PROXY']):
        print("Found proxy at {}".format(os.environ['X509_USER_PROXY']))
    else:
        print("os.environ['X509_USER_PROXY'] ",os.environ['X509_USER_PROXY'])
    os.environ['X509_CERT_DIR'] = '/cvmfs/cms.cern.ch/grid/etc/grid-security/certificates'
    os.environ['X509_VOMS_DIR'] = '/cvmfs/cms.cern.ch/grid/etc/grid-security/vomsdir'
    os.environ['X509_USER_CERT'] = proxy_path
    
    env_extra = [
            'export XRD_RUNFORKHANDLER=1',
            'export X509_USER_PROXY={}'.format(proxy_path),
            'export X509_CERT_DIR={}'.format(os.environ["X509_CERT_DIR"]),
        ]

    port_number = 9997
    cern_cluster_config = {"cores": 1,
                           "memory": "2000MB",
                           "disk": "2000MB",
                           "death_timeout":"60",
                           "lcg": True, #is there a bug in dask_lxplus?, submitted issue
                           "nanny": False,
                           "container_runtime": "none",
                           "log_directory": "/eos/user/p/pinkaew/condor/log",
                           "scheduler_options": {"port": port_number, # port number to communicate with cluster
                                                 "host": socket.gethostname()
                                                },
                           "job_extra": {"MY.JobFlavour": "'espresso'",
                                        },
                           "extra": ["--worker-port 10000:10100"],
                           "env_extra": env_extra

    }

    # with defines the scope of cluster, client
    # this ensures that cluster.close() and client.close() are called at the end
    print("Iniatiating CernCluster")
    with CernCluster(**cern_cluster_config) as cluster:
        cluster.adapt(minimum=2, maximum=100)
        cluster.scale(8)
        print("Iniatiating Client")
        with Client(cluster) as client:
            # define runner
            runner = processor.Runner(
                                executor=processor.DaskExecutor(client=client, retries=6),
                                schema=JMENanoAODSchema,
                                # size of each chunk to process (a unit of work)
                                # approximately, grow linearly with memory usage
                                # chunksize=100000,

                                # number of maximum chunks to process in each dataset, default to whole dataset.
                                # do not set this when running the full analysis.
                                # set this when testing
                                maxchunks=10,
                                # other arguments
                                skipbadfiles=True,
                                )

            # processing
            print("="*50)
            print("Begin Processing")
            print("(Save file: {})".format(args.out_file))
            mkdir_if_not_exists(os.path.dirname(args.out_file))
            print("="*50)
            start_time = datetime.datetime.now()
            out = runner(fileset, treename="Events", processor_instance=OHProcessor(**processor_config))
            end_time = datetime.datetime.now()
            elapsed_time = end_time-start_time
            print("="*50)

            print("Finish Processing")
            print("Elapsed time: {:.3f} s".format(elapsed_time.total_seconds()))
            print_dict_json(out.get("cutflow", dict()), title="Cutflow")

            # post-processing output
            out["arguments"] = vars(args)
            out["configurations"] = configs._sections
            out["start_timestamp"] = start_time.strftime("%d/%m/%Y, %H:%M:%S")
            out["process_time"] = elapsed_time
            print("="*50)

            print("Save to Output file: {}".format(args.out_file))
            cutil.save(out, args.out_file)
            print("All Complete!")
