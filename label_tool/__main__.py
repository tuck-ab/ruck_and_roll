import argparse
import os
import pathlib

from .labeller import Labeller

## -- Defining useful directory paths
MODULE_DIR = pathlib.Path(__file__).parent
VIDEO_DIR = os.path.join(MODULE_DIR, "..", "videos")
FRAME_DIR = os.path.join(MODULE_DIR, "..", "images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--File", help="Input video file (do not use with -p)")
    parser.add_argument("-o", "--Output", help="Output file name")
    parser.add_argument("-p", "--Path", help="Path of the video file (do not use with -f)")
    parser.add_argument("--fromframe", help="Start from a given frame in a video")

    args = parser.parse_args()
    
    if args.File is None and args.Path is None:
        print("No file given. Use -f or --File to specify video in /videos/ or -p or --Path")
        exit(0)
        
    if args.File and args.Path:
        print("Both path and filename given. Only use one")
        exit(1)
        
    if args.File:
        vid_path = os.path.join(VIDEO_DIR, args.File)
    else:
        vid_path = args.Path
        
    if not os.path.exists(vid_path):
        print("Video file does not exist")
        
    labeller = Labeller(vid_path)
    
    label_tracker = labeller.run(start_frame=args.fromframe)
    
    output = args.Output
    if output is None:
        ## Work out what the start frame is for the label name
        if args.fromframe is None:
            start_frame = 0
        else:
            start_frame = int(args.fromframe)
            
        file_name = f"{vid_path.split(os.sep)[-1].split('.')[0]}from{start_frame}.lbl"
        output = os.path.join(".", file_name)
        
    label_tracker.write_to_file(output)
