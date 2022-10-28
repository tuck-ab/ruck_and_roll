import cv2
import os
import pathlib
import enum

## -- Defining useful directory paths
FILE_DIR = pathlib.Path(__file__).parent
VIDEO_DIR = os.path.join(FILE_DIR, "..", "videos")
FRAME_DIR = os.path.join(FILE_DIR, "..", "images")

## -- Labels for the game
LABELS = ["Nothing", "Forceful Tackle", "Absorbing Tackle",
            "Other Tackle", "Ruck", "Maul", "Lineout", "Scrum"]

Video_Label = enum.Enum("Video_Label", LABELS)


def save_video_frames(file_name, vid_path=VIDEO_DIR, frame_path=FRAME_DIR):
    """Saves every frame in a video as a .jpg in a given directory

    Args:
        file_name (str): The name of the video to save frames from
        vid_path (str / os.path, optional): Directory where the video is saved. 
        Defaults to VIDEO_DIR.
        frame_path (str / os.path, optional): Directory where the frame images
        are saved. Defaults to FRAME_DIR.
    """
    
    cam = cv2.VideoCapture(os.path.join(vid_path, file_name))
    
    ## Check if save directory exists and make one if not
    try:
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
    except OSError:
        print(f"Could not find directory to store frames\nCreating directory: {frame_path}")

    current_frame = 0

    try:
        while True:
            ## Read the image frame
            ret, frame = cam.read()
            
            if ret:
                frame_image_path = os.path.join(frame_path, f"frame{current_frame}.jpg")
                print(f"Creating: {frame_image_path}")
    
                ## Save the frame
                cv2.imwrite(frame_image_path, frame)

                current_frame += 1
            else:
                break
    except KeyboardInterrupt:
        pass 
    finally:
        ## Close and shut down objects properly
        cam.release()
        cv2.destroyAllWindows()
        
def play_video(file_name, vid_path=VIDEO_DIR):
    """Plays a video frame by frame. Allows the labeling of each frame.

    Args:
        file_name (_type_): The name of the video to play
        vid_path (_type_, optional): Directory where the video is saved.
        Defaults to VIDEO_DIR.
    """
    
    cam = cv2.VideoCapture(os.path.join(vid_path, file_name))
    
    frame = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    
    cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
    
    frame_number = 0
    
    while cam.isOpened():
        ret, frame = cam.read()
        
        ## Write labels to help user
        ## Frame number
        cv2.putText(frame, 
                    f"Frame: {frame_number}", 
                    (20, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 0), 2, 1)
        
        ## What key to press for label number
        for i, label in enumerate(LABELS):
            cv2.putText(frame, 
                        f"{i} - {label}", 
                        (20, 20 + (i+1) * 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 0), 2, 1)
        
        cv2.imshow("video", frame)
        key = cv2.waitKey(0)
        
        ## Valid inputs when viewing a frame
        VALID_KEYS = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'),
                      ord('5'), ord('6'), ord('7'), ord('q'), ord('n')]
        
        while key != VALID_KEYS:
            key = cv2.waitKey(0)
            
        ## Decode the input from user inputs
        if key == ord('q'):
            break
        elif key == ord('0'):
            pass
        elif key == ord('1'):
            pass
        elif key == ord('2'):
            pass
        elif key == ord('3'):
            pass
        elif key == ord('4'):
            pass
        elif key == ord('5'):
            pass
        elif key == ord('6'):
            pass
        elif key == ord('7'):
            pass
        
        frame_number += 1
        
    cam.release()
    
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    play_video("20220924dmpvcambridge.mp4")
    
        