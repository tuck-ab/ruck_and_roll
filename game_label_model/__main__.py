from .cli import CommandLineInterface
from .tests import run_tests
from .u1903266 import run as u1903266_run

def main():
    print("Running main")

if __name__ == "__main__":
    cli = CommandLineInterface()
    cli.parse()
    
    if cli.get_test_flag():
        run_tests()

    video_dir = cli.get_vid_dir()

    yolo_model_dir = cli.get_yolo_model_dir()

    temp_dir = cli.get_temp_dir()

    label_dir = cli.get_label_dir()
    
    what_to_run = cli.get_what_to_run()

    if what_to_run == "u1903266":
        u1903266_run(video_dir, yolo_model_dir, temp_dir, label_dir, main)
    else:
        main()