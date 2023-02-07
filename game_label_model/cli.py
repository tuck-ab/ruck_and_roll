import argparse

class CommandLineInterface:
    """Class for dealing with command line options.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Model used to predict actions in rugby"
        )
        
        self.parser.add_argument("--run_tests", help="Run the tests", action="store_true")
        
    def parse(self):
        self.args = self.parser.parse_args()
        
    def get_test_flag(self):
        return self.args.run_tests