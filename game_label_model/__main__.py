from .cli import CommandLineInterface
from .tests import run_tests

if __name__ == "__main__":
    cli = CommandLineInterface()
    cli.parse()
    
    if cli.get_test_flag():
        run_tests()

    video_dir = cli.get_vid_dir()