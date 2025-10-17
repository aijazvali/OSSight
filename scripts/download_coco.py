import os, zipfile, argparse, requests
from tqdm.auto import tqdm

URLS = {
    "val2017":  "http://images.cocodataset.org/zips/val2017.zip",
    "train2017":"http://images.cocodataset.org/zips/train2017.zip",
    "ann":      "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

def http_get(url, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        print(f"Already exists: {out_path}")
        return
    print(f"Downloading: {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def unzip_to(zip_path, dst_dir):
    print(f"Unzipping: {os.path.basename(zip_path)}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dst_dir)

def main(args):
    root = os.path.abspath(args.root)
    coco_dir = os.path.join(root, "coco")
    img_dir = os.path.join(coco_dir, "images")
    ann_dir = os.path.join(coco_dir, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # images
    if args.split not in ("val2017", "train2017"):
        raise ValueError("--split must be val2017 or train2017")

    zimg = os.path.join(coco_dir, f"{args.split}.zip")
    http_get(URLS[args.split], zimg)
    if not os.path.exists(os.path.join(img_dir, args.split)):
        unzip_to(zimg, img_dir)

    # annotations
    zann = os.path.join(coco_dir, "annotations_trainval2017.zip")
    http_get(URLS["ann"], zann)
    if not os.path.exists(os.path.join(coco_dir, "annotations", "captions_val2017.json")):
        unzip_to(zann, coco_dir)

    print("Done.")
    print("Images:", os.path.join(img_dir, args.split))
    print("Captions JSON:", os.path.join(coco_dir, "annotations", f"captions_{args.split}.json"
          if args.split == "train2017" else "captions_val2017.json"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--split", type=str, default="val2017", help="val2017 or train2017")
    main(p.parse_args())
