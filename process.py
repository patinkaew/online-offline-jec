import argparse
import configparser
import glob
import json
import datetime
from functools import partial
import sys
import os
import inspect

from coffea import processor
from coffea import util as cutil

from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from processor.processor import OHProcessor, SimpleProcessor
from processor.schema import JMENanoAODSchema, ScoutingJMENanoAODSchema
from util import *

import warnings
import time

def get_default_dataset_name(data_dir):
    end_index = data_dir.rfind("/")
    if end_index +1 == len(data_dir):
        end_index = data_dir[:-1].rfind("/")
    start_index = data_dir[:end_index].rfind("/")
    return data_dir[start_index+1: end_index]
    
def get_filelist(data_dir):
    filelist = glob.glob(os.path.join(data_dir, "*/*/*.root"))
    if len(filelist) == 0:
        filelist = glob.glob(os.path.join(data_dir, "*/*.root"))
    return filelist

def remove_badfiles(fileset):
    print("="*50)
    print("checking bad files")
    for dataset in fileset:
        badcounts = 0
        filecounts = len(fileset[dataset])
        for filename in fileset[dataset]:
            with uproot.open(filename) as f:
                if "Events" not in f.keys():
                    #print("remove file: {} from dataset {}".format(filename, dataset))
                    fileset[dataset].remove(filename)
                    badcounts += 1
        print("remove {} bad files of {} files from dataset {}".format(badcounts, filecounts, dataset))
    print("="*50)
    
# def build_fileset(data_dir, dataset_names=None):
#     if dataset_names is not None:
#         assert len(data_dir) == len(dataset_names)
#         dataset_names = [dataset_name if dataset_name != "*" else get_default_dataset_name(data_dir[i]) 
#                          for i, dataset_name in enumerate(dataset_names)]
#     else:
#         dataset_names = map(get_default_dataset_name, data_dir)
    
#     filelist = map(get_filelist, data_dir)
#     return dict(zip(dataset_names, filelist))

# work in progress
def build_fileset(input_paths, dataset_names=None):
    if dataset_names is not None: # is directory
        assert len(input_paths) == len(dataset_names), "Number of provided dataset names must equal to input path"
    fileset = dict()
    dataset_name = "ScoutingPFMonitor"
    for i, input_path in enumerate(input_paths):
        if os.path.isdir(input_path):
            #dataset_name = dataset_names[i] if dataset_names[i] != "*" else get_default_dataset_name(input_path)
            filelist = get_filelist(input_path)
            fileset[dataset_name] = filelist
        elif os.path.isfile(input_path): # is file
            if input_path.endswith("txt"):
                with open(filename) as file:
                    filelist = [line.rstrip() for line in file]
                #dataset_name = dataset_name if dataset_names[] != "*" else get_default_dataset_name(filelist[0])
                fileset[dataset_name] = filelist
            elif input_path.endswith("json"):
                if len(input_paths) == 1:
                    return input_path
            else:
                raise ValueError("Only txt and json are supported!")
        else:
            raise ValueError("input_path is invalid (neither director nor file). It might not exist.")
    return fileset

def build_processor_config(processor_class, configs, args):
    processor_config = dict()
    fields, defaults = list(zip(*[(parameter.name, \
                                   parameter.default if not isinstance(parameter.default, inspect._empty) else None) 
                            for parameter in inspect.signature(processor_class).parameters.values()]))
    defaults = [None] * len(fields) # will require config file to explicitly have all arguments
    arg_dict = vars(args)
    for field, default in zip(fields, defaults):
        if not field in arg_dict or arg_dict[field] == None:
            assert field in configs["Processor"], "config file does not contain {} for processor".format(field)
            if default:
                processor_config[field] = configs["Processor"].get(field, default)
            else:
                processor_config[field] = configs["Processor"].get(field)
            # try eval
            try:
                processor_config[field] = eval(processor_config[field])
            except:
                pass

        else: # overwrite from args
            print("Overwrite {} from {} to {}".format(field, configs["Processor"].get(field), arg_value))
            processor_config[field] = arg_value
            configs._sections["Processor"][field] = str(arg_value)
    return processor_config

def print_num_inputfiles(fileset):
    print("="*50)
    print("Number of input files to be processed")
    [print("{} : {}".format(key, len(value))) for key, value in fileset.items()]
    print("Total : {}".format(sum(map(len, fileset.values()))))
    print("="*50)
            
def processing(args, configs, runner, fileset, processor_instance, treename="Events"):
    print("="*50)
    print("Begin Processing")
    print("(Save file: {})".format(args.out_file))
    mkdir_if_not_exists(os.path.dirname(args.out_file))
    print("="*50)
    start_time = datetime.datetime.now()
    out = runner(fileset, treename=treename, processor_instance=processor_instance)
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
    print("Process configuration to processor (Priority args > configs)")
    processor_config = build_processor_config(OHProcessor, configs, args)
    print("="*50)
    
    # change to list for printing
    configs._sections["Processor"]["off_jet_weight_filelist"] = eval(configs._sections["Processor"]["off_jet_weight_filelist"])
    configs._sections["Processor"]["on_jet_weight_filelist"] = eval(configs._sections["Processor"]["on_jet_weight_filelist"])
    print_dict_json(configs._sections, title="Configurations")
    
    # build fileset
    fileset = build_fileset(args.input_dir, args.dataset_name)
    remove_badfiles(fileset)
    fileset_json_filename = args.out_file + "_fileset.json"
    with open(fileset_json_filename , "w") as file:
        json.dump(fileset, file, indent=4)
#     max_file = 1 # for testing
#     if max_file is not None:
#         for dataset in fileset:
#             fileset[dataset] = sorted(fileset[dataset])[:max_file]
    print_num_inputfiles(fileset)
    
#     p = OHProcessor(**processor_config)
#     exit()
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
#     num_trial = 1
#     print("trying data")
#     for trial in range(num_trial):
#         print("Data {}".format(trial))
#         data_dir = "/data/data/JME_NANO_DATA/2022/JMENanoRun3_v2p1_Run2022D-PromptReco-v2/JetMET/220915_173253/0000/"
#         fname = "tree_{}.root".format(num_trial)
#         #data_dir = "/data/data/ScoutingJMENano/ScoutingPFMonitor/Run3Summer22/230207_113750/0000/"
#         #fname = "jmenano_data_145.root"
#         last_time = time.time()
#         events_data = NanoEventsFactory.from_root(
#                 os.path.join(data_dir, fname), 
#                 schemaclass=ScoutingJMENanoAODSchema,
#                 metadata={"dataset": fname}
#                 ).events()
#         #print(sorted(events_data.fields))
#         read_time = time.time() - last_time
#         out = p.process(events_data)
#         out["time_pf"]["reading"] += read_time
#         print_dict_json(out.get("time_pf", dict()), title="Time profile")
#         print_dict_json(out.get("cutflow", dict()), title="Cutflow")

#     print("trying mc")
#     for trial in range(num_trial):
#         print("MC {}".format(trial))
#         mc_dir = "/data/data/JME_NANO_MC/2022/QCD_Pt-15to7000_TuneCP5_Flat_13p6TeV_pythia8/JMENanoRun3_v2p1_MC22_122/220915_171347/0000/"
#         mc_fname = "tree_{}.root".format(num_trial)
#         #mc_dir = "/data/data/ScoutingJMENano/QCD_Pt-15To7000_TuneCP5_13p6TeV_pythia8/Run3Summer22/230130_122950/0000/"
#         #mc_fname = "nanoaod_10.root"
#         last_time = time.time()
#         events_mc = NanoEventsFactory.from_root(
#             os.path.join(mc_dir, mc_fname), 
#             schemaclass=ScoutingJMENanoAODSchema,
#             metadata={"dataset": mc_fname}
#             ).events()
#         read_time = time.time() - last_time
#         out = p.process(events_mc)
#         out["time_pf"]["reading"] += read_time
#         print_dict_json(out.get("time_pf", dict()), title="Time profile")
#         print_dict_json(out.get("cutflow", dict()), title="Cutflow")
#     exit()

    # prepare runner
    print("Prepare Runner")
    executor_type = configs["Runner"].get("executor", "iterative")
    assert executor_type.lower() in ["iterative", "future", "dask"], "unrecongized executor: {}".format(executor_type)
    executor_type = executor_type.lower()
    if executor_type == "iterative" or executor_type == "future": # local
        # read configurations
        num_workers = configs["Runner"].getint("num_workers", 1)
        if num_workers == 1 and executor_type != "iterative":
            warnings.warn("Get num_workers = 1, change to iterative executor")
            executor_type = "iterative"
        compression = eval(configs["Runner"].get("compression", "None")) # compression is either int or None
        assert compression == None or isinstance(compression, int), "invalid compression level"
        
        executor = processor.IterativeExecutor(compression=compression)
        if executor_type == "future":
            executor = processor.FuturesExecutor(compression=compression, workers=num_workers)

        runner = processor.Runner(
                executor=executor,
                schema=ScoutingJMENanoAODSchema,
                # size of each chunk to process (a unit of work), default to 100000
                # approximately, grow linearly with memory usage
                chunksize=configs["Runner"].getint("chunksize", 100_000),

                # number of maximum chunks to process in each dataset, default to whole dataset
                # do not set this when running the full analysis
                # set this when testing
                maxchunks=eval(configs["Runner"].get("maxchunks", "None")),
                )
        
        # processing
        processing(args, configs, runner, fileset, treename="Events", processor_instance=OHProcessor(**processor_config))
        
    else: # distributed
        # currently only for lxplus
        import socket
        from dask.distributed import Client 
        from dask_lxplus import CernCluster
        import shutil
    
#     job flavor
#     espresso     = 20 minutes
#     microcentury = 1 hour
#     longlunch    = 2 hours
#     workday      = 8 hours
#     tomorrow     = 1 day
#     testmatch    = 3 days
#     nextweek     = 1 week
    
        print("Current interpreter: {}".format(sys.executable))
    
        # configure environment for grid authentication
        proxy_path = configs["Runner"].get("proxy_path", "/afs/cern.ch/user/p/pinkaew/private/gridproxy.pem")
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
        
        transfer_input_filelist = [fileset_json_filename]
        path_proccessor_configs = [_ for _ in processor_config.keys() if _.endswith("path") or _.endswith("filelist")]
        for config in path_proccessor_configs:
            if processor_config[config] != None:
                if isinstance(processor_config[config], str):
                    transfer_input_filelist += [processor_config[config]]
                elif isinstance(processor_config[config], list):
                    transfer_input_filelist += processor_config[config]
                else:
                    raise ValueError("Processor config {} was recognized as path, but value {} is neither str or list"\
                                     .format(config, processor[config]))
        transfer_input_files = ",".join(transfer_input_filelist)

        port_number = configs["Runner"].getint("port_number", 8786)
        cern_cluster_config = {"cores": 1,
                               "memory": "2000MB",
                               "disk": "10GB",
                               "death_timeout":"60",
                               "lcg": True,
                               "nanny": False,
                               "container_runtime": "none",
                               "log_directory": "/eos/user/p/pinkaew/condor/log",
                               "scheduler_options": {"port": port_number, # port number to communicate with cluster
                                                     "host": socket.gethostname()
                                                    },
                               "job_extra": {
                                   "MY.JobFlavour": '"{}"'.format(configs["Runner"].get("job_flavour", "longlunch")),
                                   # only executables are transfer, heres are corrections
                                   "transfer_input_files": transfer_input_files
                                            },
                               "batch_name": configs["Runner"].get("batch_name", "dask-worker"),
                               "extra": ["--worker-port 10000:10100"],
                               "env_extra": env_extra

        }
        
        # xrootdstr xrootd redirector
        xrootdstr = "root://xrootd-cms.infn.it//" # for Europe and Asia
        #xrootdstr = "root://cmsxrootd.fnal.gov//"
        #xrootdstr = "root://xcache/" # for coffea.casa
        #xrootdstr = "root://cms-xrd-global.cern.ch//" # query all sites
        for dataset in fileset:
            # remove /eos/cms and prepend xrootd redirector
            fileset[dataset] = [xrootdstr + filename[8:] for filename in fileset[dataset]]
        
        # with defines the scope of cluster, client
        # this ensures that cluster.close() and client.close() are called at the end
        num_workers = configs["Runner"].getint("num_workers", 2)
        min_jobs = configs["Runner"].getint("min_jobs", 2)
        max_jobs = configs["Runner"].getint("max_jobs", 64)
        print("Initiating CernCluster")
        with CernCluster(**cern_cluster_config) as cluster:
            cluster.adapt(minimum=min_jobs, maximum=max_jobs)
            cluster.scale(num_workers)
            print("Initiating Client")
            with Client(cluster) as client:
                # uploading code, corrections were uploaded with transfer_input_files
                print("Uploading processor")
                # need to zip, so that processor retains directory structure
                # upload_file individually will lose directory structure
                shutil.make_archive("processor", "zip", base_dir="processor")
                client.upload_file("processor.zip")
                
                # define runner
                runner = processor.Runner(
                                    executor=processor.DaskExecutor(client=client, retries=6),
                                    schema=ScoutingJMENanoAODSchema,
                                    # size of each chunk to process (a unit of work)
                                    # approximately, grow linearly with memory usage
                                    chunksize=configs["Runner"].getint("chunksize", 100_000),

                                    # number of maximum chunks to process in each dataset, default to whole dataset.
                                    # do not set this when running the full analysis.
                                    # set this when testing
                                    maxchunks=eval(configs["Runner"].get("maxchunks", "None")),
                                    # other arguments
                                    skipbadfiles=True,
                                    xrootdtimeout=60
                                    )

                # processing
                processing(args, configs, runner, fileset_json_filename, treename="Events", 
                           processor_instance=OHProcessor(**processor_config))
