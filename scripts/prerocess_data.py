import json
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm


def save_frames(
    video_path: Path,
    save_root: Path,
    basename: str,
    image_width: int = 256,
    image_heigh: int = 256,
    laplacian_thr: int = 800,
    fps: int = 30,
    interval_sec: int = 60,
    img_format: str = "png",
):
    cpf = fps * interval_sec

    cap = cv2.VideoCapture(str(video_path))
    count = 0
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"{str(video_path)} finished.")
            break

        if count % cpf == 0:
            resize_frame = cv2.resize(frame, (image_width, image_heigh))
            laplacian = cv2.Laplacian(resize_frame, cv2.CV_64F)
            i += 1
            if ret and laplacian.var() >= laplacian_thr:
                img_name = str(save_root / f"{basename}_{i:08d}.{img_format}")
                write = cv2.imwrite(img_name, resize_frame)
                if not write:
                    raise RuntimeError("Fail to save")
        count = count + 1

    cap.release()


save_root = Path("/home/ayase/tmp/frames_test")
save_root.mkdir(exist_ok=True)

search_root = Path("/mnt/multimedia/nico_download/")
flist = list(search_root.glob("./*/*mp4"))


id_dict = {}
all_dict = defaultdict(dict)
prog_id = 0
video_id = 0
for f in tqdm(flist):
    f: Path
    prog_name = f.parent.name
    if prog_name not in id_dict:
        id_dict[prog_name] = prog_id
        prog_id += 1

    all_dict[prog_id][video_id] = str(f)
    save_dir = save_root / f"{prog_id:05d}"
    basename = f"{video_id:09d}"
    save_dir.mkdir(exist_ok=True)
    save_frames(video_path=f, save_root=save_dir, basename=basename, interval_sec=30)

with open("prog_id.json", "w") as f:
    json.dump(id_dict, f)

with open("all_mapping.json", "w") as f:
    json.dump(all_dict, f)
