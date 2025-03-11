import os
import cv2
import numpy as np


def generate_video_from_frames(frame_folder, output_video, fps=7, frame_skip=1):
    """
    Genera un video a partire da una cartella di frame.

    Args:
        frame_folder (str): Cartella contenente i frame.
        output_video (str): Percorso del file video di output.
        fps (int): Frame rate del video.
        frame_skip (int): Numero di frame da saltare (1 = nessun salto, 3-4 per downsampling a 7 FPS).
    """
    frames = sorted(os.listdir(frame_folder))
    frames = frames[::frame_skip]  # Seleziona solo i frame desiderati

    if not frames:
        print("Nessun frame trovato nella cartella.")
        return

    first_frame = cv2.imread(os.path.join(frame_folder, frames[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec video
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame_name in frames:
        frame_path = os.path.join(frame_folder, frame_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue  # Salta frame corrotti o non leggibili
        out.write(frame)

    out.release()
    print(f"Video salvato in {output_video}")


# Percorsi di input e output
frame_folder = r"D:\SVD_Xtend\prova\frames\9"  # Modifica con il percorso reale
output_video_1 = "video_full_7fps.mp4"  # Versione con tutti i frame a 7 FPS
output_video_2 = "video_sampled_7fps.mp4"  # Versione con sampling ogni 3-4 frame

# Genera i video
generate_video_from_frames(frame_folder, output_video_1, fps=7, frame_skip=1)  # Usa tutti i frame
generate_video_from_frames(frame_folder, output_video_2, fps=7, frame_skip=3)  # Prende un frame ogni 3-4