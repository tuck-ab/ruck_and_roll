import argparse
import os

from .dir_paths import VIDEO_DIR

class CommandLineInterface:
    """Class for dealing with command line options.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Model used to predict actions in rugby"
        )
        
        self.parser.add_argument(
            "--run_tests", 
            help="run the tests in test.py", 
            action="store_true")
        
        self.parser.add_argument(
            "--u1903266", 
            help="Flag for running on dcs machines where files are stored in /dcs/large for u1903266", 
            action="store_true")

    def parse(self):
        self.args = self.parser.parse_args()
        
    def get_test_flag(self):
        return self.args.run_tests

    def get_vid_dir(self):
        if self.args.u1903266:
            return os.path.join("dcs", "large", "u1903266", "videos")
        

        return VIDEO_DIR