from moviepy import VideoFileClip
import os

# --- CONFIG ---
INPUT_FILE = "PyVista 2025-11-25 23-28-39.mp4"
OUTPUT_FILE = "brain_rotation.gif"
START_SEC = 6
END_SEC = 14
CROP_BOTTOM_FRACTION = 1/7

def create_clean_gif():
    if not os.path.exists(INPUT_FILE):
        print(f"Fehler: Datei '{INPUT_FILE}' nicht gefunden.")
        return

    clip = VideoFileClip(INPUT_FILE)
    
    clip_trimmed = clip.subclipped(START_SEC, END_SEC)
    
    w, h = clip.size
    new_height = int(h * (1 - CROP_BOTTOM_FRACTION))
    
    clip_cropped = clip_trimmed.cropped(x1=0, y1=0, width=w, height=new_height)
    
    clip_cropped.write_gif(OUTPUT_FILE, fps=15)
    
    print(f"Done! GIF saved as: {OUTPUT_FILE}")

if __name__ == "__main__":
    create_clean_gif()