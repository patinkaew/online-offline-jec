import sys
import os
import argparse
import inspect
from collections import OrderedDict
from process import io_default_config, runner_default_config
from processor.processor import OnlineOfflineProcessor
from util import update_config_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--replace", action=argparse.BooleanOptionalAction, required=False)
    args = parser.parse_args()
    
    default_config_dict = OrderedDict()
    default_config_dict["IO"] = io_default_config
    default_config_dict["Processor"] = [(parameter.name, parameter.default if parameter.default != inspect._empty else "") 
                                        for parameter in inspect.signature(OnlineOfflineProcessor).parameters.values()]
    config_filename = os.path.splitext(os.path.basename(args.config_file))[0]
    default_config_dict["Runner"] = runner_default_config[:-1] + [("batch_name", config_filename)]
    
    update_config_file(default_config_dict, args.config_file, args.replace)