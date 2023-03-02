import sys
from processor.processor import OHProcessor
from util import create_default_config_file

if __name__ == "__main__":
    config_file = sys.argv[1]
    create_default_config_file(OHProcessor, config_file)