import argparse
import os

from .dir_paths import VIDEO_DIR, YOLO_MODEL_DIR, LABEL_DIR, TEMP_DIR, GRAPH_MODEL_DIR

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

        self.parser.add_argument(
            "--u1921012", 
            help="Flag for running on dcs machines where files are stored in /dcs/large for u1921012", 
            action="store_true")
        
        self.parser.add_argument(
            "--run",
            help="What fuction to run"
        )

        self.parser.add_argument(
            "--model_num",
            help="Optional number to save model to different number file"
        )

    def parse(self):
        self.args = self.parser.parse_args()

        assert sum([self.args.u1903266, self.args.u1921012]) <= 1, "Only one DCS large file flag should be specified at once"
        
    def get_test_flag(self):
        return self.args.run_tests

    def get_vid_dir(self) -> str:
        """
        Uses the flags to work out if there is a different path for the videos
        needed.

        Returns:
            str: Path to the directory containing the videos
        """
        
        if self.args.u1903266:
            return os.path.join("/dcs", "large", "u1903266", "videos")

        if self.args.u1921012:
            return os.path.join("/dcs", "large", "u1921012", "videos")
        
        return VIDEO_DIR

    def get_yolo_model_dir(self) -> str:
        """Uses the flags to work out if there is a different path for the yolo
        models needed.

        Returns:
            str: Path to the directory containing the yolo models
        """
        
        if self.args.u1903266:
            return os.path.join("/dcs", "large", "u1903266", "yolo_models")

        if self.args.u1921012:
            return os.path.join("/dcs", "large", "u1921012", "yolo_models")

        return YOLO_MODEL_DIR

    def get_label_dir(self) -> str:
        """Uses the flags to work out if there is a different path for the label
        files needed.

        Returns:
            str: Path to the directory containing the label files
        """
        
        if self.args.u1903266:
            return os.path.join("/dcs", "large", "u1903266", "labels")

        if self.args.u1921012:
            return os.path.join("/dcs", "large", "u1921012", "labels")

        return LABEL_DIR
    
    def get_temp_dir(self) -> str:
        """Uses the flags to work out if there is a different path for a
        temporary directory

        Returns:
            str: Path to a temporary directory
        """

        if self.args.u1903266:
            return os.path.join("/dcs", "large", "u1903266", "temp")
        
        return TEMP_DIR
    
    def get_graph_model_dir(self) -> str:


        return GRAPH_MODEL_DIR
    
    def get_model_num(self) -> int:
        if self.args.model_num:
            return int(self.args.model_num)
        else:
            return 0

    def get_what_to_run(self) -> str:
        """Uses the flags to work out if there is a different function to be
        run instead of `main`

        Returns:
            str: A flag to be decoded in `__main__.py`
        """
        
        if self.args.run == "u1903266":
            return "u1903266"
        
        return "main"
