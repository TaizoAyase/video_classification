import argparse
import hashlib
import json
from pathlib import Path

import cv2
from joblib import Parallel, delayed
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
            # print(f"{str(video_path)} finished.")
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


parser = argparse.ArgumentParser()
parser.add_argument("--input_root", type=str, help="input root directory for mp4 files")
parser.add_argument("--output_root", type=str, help="Directory to output save images")
parser.add_argument("--width", default=224, type=int, help="resize image width")
parser.add_argument("--height", default=224, type=int, help="reside image height")
parser.add_argument("--n_jobs", default=8, type=int, help="number of workers")
parser.add_argument("--interval", default=60, type=int, help="frame interval (sec)")
args = parser.parse_args()

save_root = Path(args.output_root)
save_root.mkdir(exist_ok=True)

search_root = Path(args.input_root)
flist = list(search_root.glob("./*/*mp4"))

id_dict = {}
prog_id = 0
for f in tqdm(flist):
    f: Path
    prog_name = f.parent.name
    if prog_name not in id_dict:
        id_dict[prog_name] = prog_id
        prog_id += 1

with open("prog_id.json", "w") as f:
    json.dump(id_dict, f)


def process(file: Path):
    prog_name = file.parent.name
    current_pid = id_dict[prog_name]
    save_dir = save_root / f"{current_pid:05d}"
    video_id = hashlib.md5(file.name.encode()).hexdigest()
    save_dir.mkdir(exist_ok=True)
    save_frames(
        video_path=file,
        save_root=save_dir,
        basename=video_id,
        interval_sec=args.interval,
        image_width=args.width,
        image_heigh=args.height,
    )


Parallel(n_jobs=args.n_jobs, verbose=10)(delayed(process)(f) for f in flist)
# [process(f) for f in tqdm(flist)]
