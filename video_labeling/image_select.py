import cv2
import os
import pathlib
import enum

FILE_DIR = pathlib.Path(__file__).parent
VIDEO_DIR = os.path.join(FILE_DIR, "..", "videos")
FRAME_DIR = os.path.join(FILE_DIR, "..", "images")

LABELS = ["Nothing", "Forceful Tackle", "Absorbing Tackle",
            "Other Tackle", "Ruck", "Maul", "Lineout", "Scrum"]

Video_Label = enum.Enum("Video_Label", LABELS)


def save_video_frames(file_name, vid_path=VIDEO_DIR, frame_path=FRAME_DIR):
    cam = cv2.VideoCapture(os.path.join(vid_path, file_name))
    
    try:
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
    except OSError:
        print(f"Could not find directory to store frames\nCreating directory: {frame_path}")

    current_frame = 0

    try:
        while True:
            ret, frame = cam.read()
            if ret:
                frame_image_path = os.path.join(frame_path, f"frame{current_frame}.jpg")

                print(f"Creating: {frame_image_path}")
    
                cv2.imwrite(frame_image_path, frame)

                current_frame += 1
            else:
                break
    except KeyboardInterrupt:
        pass 
    finally:
        cam.release()
        cv2.destroyAllWindows()
        
def play_video(file_name, vid_path=VIDEO_DIR):
    cam = cv2.VideoCapture(os.path.join(vid_path, file_name))
    
    frame = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    
    cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
    
    frame_number = 0
    
    while cam.isOpened():
        ret, frame = cam.read()
        
        for i, label in enumerate(LABELS):
            cv2.putText(frame, 
                        f"{i} - {label}", 
                        (20, 20 + (i+1) * 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 0, 0), 
                        2, 
                        1)
        
        cv2.putText(frame, 
                    f"Frame: {frame_number}", 
                    (20, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 0), 
                    2, 
                    1)
        
        cv2.imshow("video", frame)
        key = cv2.waitKey(0)
        
        while key != ord('q') and key != ord('n'):
            key = cv2.waitKey(0)
            
        if key == ord('q'):
            break
        
        frame_number += 1
        
    cam.release()
    
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    play_video("20220924dmpvcambridge.mp4")
    
        