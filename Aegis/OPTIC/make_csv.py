import os, csv, glob

def write_csv(csv_path, image_dir, mask_dir):
    exts = ["*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp"]
    imgs = []
    for e in exts:
        imgs += glob.glob(os.path.join(image_dir, e))
    imgs = sorted(imgs)
    rows = []
    for im in imgs:
        base = os.path.splitext(os.path.basename(im))[0]
        # find any mask with same stem (any extension)
        cand = []
        for e in exts:
            cand += glob.glob(os.path.join(mask_dir, base + e[1:]))  # base + .ext
        if not cand:
            # fallback: pick the 1st file containing the base
            found = []
            for e in exts:
                found += glob.glob(os.path.join(mask_dir, f"*{base}*{e[1:]}"))
            if found:
                cand = [found[0]]
        if cand:
            rows.append([im, cand[0]])
        else:
            print(f"[WARN] No mask found for {im}")

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image","mask"])
        w.writerows(rows)
    print(f"Wrote {csv_path} with {len(rows)} pairs")

root = ""  # we will pass dataset_root="" so absolute paths in CSV are fine

# TRAIN (Domain1)
write_csv(
    "/user1/res/res2024/miu/shramanadey_r/Aegis/Aegis/OPTIC/csv/Domain1_train.csv",
    "/user1/res/res2024/miu/shramanadey_r/Data/A. Segmentation/4. Augmented Data/Train_Data/images",
    "/user1/res/res2024/miu/shramanadey_r/Data/A. Segmentation/4. Augmented Data/Train_Data/masks"
)

# VAL/TEST Dataset 1 (we'll call it D1)
write_csv(
    "/user1/res/res2024/miu/shramanadey_r/Aegis/Aegis/OPTIC/csv/D1_test.csv",
    "/user1/res/res2024/miu/shramanadey_r/Data/A. Segmentation/4. Augmented Data/Test_Data/images",
    "/user1/res/res2024/miu/shramanadey_r/Data/A. Segmentation/4. Augmented Data/Test_Data/masks"
)

# TEST Dataset 2 (REFUGE)
write_csv(
    "/user1/res/res2024/miu/shramanadey_r/Aegis/Aegis/OPTIC/csv/REFUGE_test.csv",
    "/user1/res/res2024/miu/shramanadey_r/Data/Refuge_Dataset/Images",
    "/user1/res/res2024/miu/shramanadey_r/Data/Refuge_Dataset/OD_GT"
)

# TEST Dataset 3 (Drishti)
write_csv(
    "/user1/res/res2024/miu/shramanadey_r/Aegis/Aegis/OPTIC/csv/Drishti_test.csv",
    "/user1/res/res2024/miu/shramanadey_r/Data/Drishti-GS1_files/Drishti-GS1_files/Training/Images",
    "/user1/res/res2024/miu/shramanadey_r/Data/Drishti-GS1_files/Drishti-GS1_files/Training/OD_GT"
)

# TEST Dataset 4 (DRIONS-DB)
write_csv(
    "/user1/res/res2024/miu/shramanadey_r/Aegis/Aegis/OPTIC/csv/DRIONS_test.csv",
    "/user1/res/res2024/miu/shramanadey_r/Data/DRION_DB/DRIONS-DB/images",
    "/user1/res/res2024/miu/shramanadey_r/Data/DRION_DB/DRIONS-DB/OD_GT"
)
