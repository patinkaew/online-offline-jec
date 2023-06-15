import argparse
import configparser
import glob
import json
import datetime
from functools import partial
from collections import defaultdict
import sys
import os
import shutil
import inspect

from coffea import processor  
from coffea import util as cutil

from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from processor.processor import OnlineOfflineProcessor #, SimpleProcessor
from processor.schema import JMENanoAODSchema, ScoutingJMENanoAODSchema
from util import *

import warnings
import time

# def get_default_dataset_name(data_dir):
#     end_index = data_dir.rfind("/")
#     if end_index +1 == len(data_dir):
#         end_index = data_dir[:-1].rfind("/")
#     start_index = data_dir[:end_index].rfind("/")
#     return data_dir[start_index+1: end_index]

def get_default_dataset_name(filename):
    # expect file format as /path_to_dataset/dataset/run/batch/job/_.root
    tokens = filename.split("/")
    if filename.endswith("/"):
        return tokens[-6]
    return tokens[-5]
    
def get_filelist(data_dir):
    filelist = glob.glob(os.path.join(data_dir, "*/*/*.root"))
    if len(filelist) == 0:
        filelist = glob.glob(os.path.join(data_dir, "*/*.root"))
    return filelist

def remove_badfiles(fileset):
    print("="*50)
    print("checking bad files")
    good_fileset = defaultdict(list)
    for dataset in fileset:
        badcounts = 0
        filecounts = len(fileset[dataset])
        for filename in fileset[dataset]:
            with uproot.open(filename) as f:
                if len(f.keys()) > 0:
                    #print("remove file: {} from dataset {}".format(filename, dataset))
                    good_fileset[dataset].append(filename)
                else:
                    badcounts += 1
        print("remove {} bad files of {} files from dataset {}".format(badcounts, filecounts, dataset))
    print("="*50)
    return good_fileset
    
# def build_fileset(data_dir, dataset_names=None):
#     if dataset_names is not None:
#         assert len(data_dir) == len(dataset_names)
#         dataset_names = [dataset_name if dataset_name != "*" else get_default_dataset_name(data_dir[i]) 
#                          for i, dataset_name in enumerate(dataset_names)]
#     else:
#         dataset_names = map(get_default_dataset_name, data_dir)
    
#     filelist = map(get_filelist, data_dir)
#     return dict(zip(dataset_names, filelist))

def build_fileset(input_paths, dataset_names=None, 
                  check_bad_files=False, max_files = None, 
                  xrootd_redirector=None, output_dir=None):
    # prepare dataset names
    input_paths = input_paths.split(",")
    
    if dataset_names is not None: # is directory
        dataset_names = dataset_names.split(",")
        assert len(input_paths) == len(dataset_names), "Number of provided dataset names ({}) must equal to input paths {}".format(len(dataset_names), len(input_paths))
    else:
        dataset_names = ["*"] * len(input_paths)
    
    # configure json output
    def get_json_output_path(suffix):
        nonlocal output_dir
        json_output_path = suffix # default to save in current working directory
        if output_dir:
            if output_dir.endswith("/"): # remove / if exist at the end
                output_dir = output_dir[:-1]
            json_output_filename = os.path.basename(output_dir) + "_" + suffix
            json_output_path = os.path.join(output_dir, json_output_filename)
        return json_output_path
    json_output_path = get_json_output_path("fileset.json")
    
    # already good, just copy json to output folder
    if (len(input_paths) == 1) and (input_paths[0].endswith(".json")) \
        and (not check_bad_files) and (max_files is None)\
        and (xrootd_redirector is None):
        
        shutil.copy(input_paths[0], json_output_path)
        return json_output_path
    
    # need some processing
    fileset = defaultdict(list)
    for i in range(len(input_paths)):
        input_path = input_paths[i]
        dataset_name = dataset_names[i]
        
        if os.path.isdir(input_path):
            dataset_name = dataset_name if dataset_name != "*" else get_default_dataset_name(filelist[0])
            filelist = get_filelist(input_path)
            fileset[dataset_name] += filelist
        elif os.path.isfile(input_path): # is file
            if input_path.endswith(".txt"):
                with open(filename) as file:
                    filelist = [line.rstrip() for line in file]
                dataset_name = dataset_name if dataset_name != "*" else get_default_dataset_name(filelist[0])
                fileset[dataset_name] += filelist
            elif input_path.endswith(".json"):
                for dataset_name, filelist in json2dict(input_path).items():
                    fileset[dataset_name] += filelist
            else:
                raise ValueError("Only txt and json are supported!")
        else:
            raise ValueError("input_path {} is invalid (neither director nor file). It might not exist.".format(input_path))
            
    # check that files can be open with uproot and contain at least one keys
    if check_bad_files:
        fileset = remove_badfiles(fileset)
    
    # truncated number of files, default to include all
    if max_files:
        for dataset in fileset:
            fileset[dataset] = fileset[dataset][:max_files]
        
    # save to fileset.json
    dict2json(fileset, json_output_path)
    
    # add redirector if any
    if xrootd_redirector: 
        for dataset in fileset:
            # remove path before /store (e.g. /eos/cms) and prepend xrootd redirector
            fileset[dataset] = [xrootd_redirector + filename[filename.find("/store"):] for filename in fileset[dataset]]        
        # save to fileset_redirector.json
        json_output_path = get_json_output_path("fileset_redirector.json")
        dict2json(fileset, json_output_path)
    
    return json_output_path

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
            
def processing(configs, runner, fileset, processor_instance, treename="Events"):
    print("="*50)
    print("Begin Processing")
    print("(Save file: {})".format(configs["IO"]["output_file"]))
    #mkdir_if_not_exist(os.path.dirname(args.out_file))
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

    print("Save to Output file: {}".format(configs["IO"]["output_file"]))
    cutil.save(out, configs["IO"]["output_file"])
    print("All Complete!")

# xrootdstr xrootd redirector
#xrootdstr = "root://xrootd-cms.infn.it//" # for Europe and Asia
#xrootdstr = "root://cmsxrootd.fnal.gov//" # for America
#xrootdstr = "root://xcache/" # for coffea.casa
#xrootdstr = "root://cms-xrd-global.cern.ch//" # query all sites

#     job flavor
#     espresso     = 20 minutes
#     microcentury = 1 hour
#     longlunch    = 2 hours
#     workday      = 8 hours
#     tomorrow     = 1 day
#     testmatch    = 3 days
#     nextweek     = 1 week

io_default_config = [("input_paths", ""),
                     ("dataset_names", None),
                     ("check_bad_files", False),
                     ("max_files", None),
                     ("xrootd_redirector", None),
                     ("base_output_dir", "")
                    ]

runner_default_config = [("executor", "iterative"),
                         ("num_workers", 1),
                         ("chunksize", 100_000),
                         ("maxchunks", None),
                         ("compression", None),
                         #("proxy_path", "/tmp/x509up_u{}".format(os.getuid())), # default in voms-proxy-init
                         ("proxy_path", os.path.join(os.path.expanduser("~"), "private/gridproxy.pem")),
                         #("proxy_path", "/afs/cern.ch/user/p/pinkaew/private/gridproxy.pem"),
                         ("port_number", 8786),
                         ("log_directory", "/eos/user/{}/{}/condor/log".format(os.path.join(os.getlogin())[0], os.path.join(os.getlogin()))),
                         ("job_flavour", "workday"),
                         ("min_jobs", 2),
                         ("max_jobs", 64),
                         ("batch_name", "dask-worker")
                        ]

if __name__ == "__main__":
    # parsing arguments and configurations
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, required=True)
    
     #will not allow replacing args -> configs anymore
#     parser.add_argument("--input_paths", type=str, nargs="+", required=False)
#     parser.add_argument("--dataset_names", type=str, nargs="+", required=False)
#     parser.add_argument("--out_dir", type=str, required=False)
#     parser.add_argument("--trigger_type", type=str, required=False)
#     parser.add_argument("--trigger_min_pt", type=float, required=False)
#     parser.add_argument('--off_jet_tag_probe', action=argparse.BooleanOptionalAction, required=False)
#     parser.add_argument('--on_jet_tag_probe', action=argparse.BooleanOptionalAction, required=False)
#     parser.add_argument("--off_jet_tag_min_pt", type=float, required=False)
#     parser.add_argument("--on_jet_tag_min_pt", type=float, required=False)
    
    # new version will have everything in config file (.cfg)
    # {dataset id}_{run, QCD}_offline_online_configname.cfg
    
    args = parser.parse_args()
    print_dict_json(vars(args), title="Arguments")
    
    assert os.path.exists(args.config_file), "config file: {} does not exist!".format(args.config_file)
    configs = configparser.ConfigParser()
    configs.read(args.config_file)
    
    #  add output_dir and output_file
    config_filename = os.path.splitext(os.path.basename(args.config_file))[0]
    configs["IO"]["output_dir"] = os.path.join(configs["IO"]["base_output_dir"], config_filename)
    mkdir_if_not_exist(configs["IO"]["output_dir"])
    shutil.copy(args.config_file, configs["IO"]["output_dir"]) # copy config file over
    configs["IO"]["output_file"] = os.path.join(configs["IO"]["output_dir"], config_filename + ".coffea")
    
    # build fileset
    build_fileset_args = dict()
    for parameter in inspect.signature(build_fileset).parameters.values():
        try:
            build_fileset_args[parameter.name] = eval(configs["IO"].get(parameter.name))
        except:
            build_fileset_args[parameter.name] = configs["IO"].get(parameter.name)
    fileset_json_path = build_fileset(**build_fileset_args)
    
    print("="*50)
    print("Process configuration to processor (Priority args > configs)")
    processor_config = build_processor_config(OnlineOfflineProcessor, configs, args)
    print("="*50)
    
    # change to list for printing
    configs._sections["Processor"]["off_jet_weight_filelist"] = eval(configs._sections["Processor"]["off_jet_weight_filelist"])
    configs._sections["Processor"]["on_jet_weight_filelist"] = eval(configs._sections["Processor"]["on_jet_weight_filelist"])
    print_dict_json(configs._sections, title="Configurations")
    
    # prepare runner
    print("Prepare Runner")
    runner_default_config_dict = dict(runner_default_config)
    def get_runner_config(field):
        try:
            value = eval(configs["Runner"].get(field, runner_default_config_dict[field]))
        except:
            value = configs["Runner"].get(field, runner_default_config_dict[field])
        return value
        
    executor_type = get_runner_config("executor")
    assert executor_type.lower() in ["iterative", "future", "dask"], "unrecongized executor: {}".format(executor_type)
    executor_type = executor_type.lower()
    if executor_type == "iterative" or executor_type == "future": # local
        # read configurations
        num_workers = get_runner_config("num_workers")
        if num_workers == 1 and executor_type != "iterative":
            warnings.warn("Get num_workers = 1, change to iterative executor")
            executor_type = "iterative"
        compression = get_runner_config("compression") # compression is either int or None
        assert compression == None or isinstance(compression, int), "invalid compression level"
        
        executor = processor.IterativeExecutor(compression=compression)
        if executor_type == "future":
            executor = processor.FuturesExecutor(compression=compression, workers=num_workers)

        runner = processor.Runner(
                executor=executor,
                schema=ScoutingJMENanoAODSchema,
                # size of each chunk to process (a unit of work), default to 100000
                # approximately, grow linearly with memory usage
                chunksize=get_runner_config("chunksize"),

                # number of maximum chunks to process in each dataset, default to whole dataset
                # do not set this when running the full analysis
                # set this when testing
                maxchunks=get_runner_config("maxchunks"),
                )
        
        # processing
        processing(configs, runner, fileset_json_path, treename="Events",
                   processor_instance=OnlineOfflineProcessor(**processor_config))
        
    else: # distributed
        # currently only for lxplus
        import socket
        from dask.distributed import Client 
        from dask_lxplus import CernCluster
    
        print("Current interpreter: {}".format(sys.executable))
    
        # configure environment for grid authentication
        proxy_path = get_runner_config("proxy_path")
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
        
        transfer_input_filelist = [fileset_json_path]
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

        port_number = get_runner_config("port_number")
        cern_cluster_config = {"cores": 1,
                               "memory": "2000MB",
                               "disk": "10GB",
                               "death_timeout":"60",
                               "lcg": True,
                               "nanny": False,
                               "container_runtime": "none",
                               "log_directory": get_runner_config("log_directory"),
                               "scheduler_options": {"port": port_number, # port number to communicate with cluster
                                                     "host": socket.gethostname()
                                                    },
                               "job_extra": {
                                   "MY.JobFlavour": '"{}"'.format(get_runner_config("job_flavour")),
                                   # only executables are transfer, heres are corrections
                                   "transfer_input_files": transfer_input_files
                                            },
                               "batch_name": get_runner_config("batch_name"),
                               "extra": ["--worker-port 10000:10100"],
                               "env_extra": env_extra

        }
        
        # with defines the scope of cluster, client
        # this ensures that cluster.close() and client.close() are called at the end
        num_workers = max(get_runner_config("num_workers"), 2)
        min_jobs = get_runner_config("min_jobs")
        max_jobs = get_runner_config("max_jobs")
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
                                    chunksize=get_runner_config("chunksize"),

                                    # number of maximum chunks to process in each dataset, default to whole dataset.
                                    # do not set this when running the full analysis.
                                    # set this when testing
                                    maxchunks=get_runner_config("maxchunks"),
                                    # other arguments
                                    skipbadfiles=True,
                                    xrootdtimeout=60
                                    )

                # processing
                processing(configs, runner, fileset_json_path, treename="Events", 
                           processor_instance=OHProcessor(**processor_config))
