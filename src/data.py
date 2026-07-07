import os
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

class CustomStereoDataset(Dataset):
    """
    A custom dataset class to load stereo images and depth maps from separate folders, matching by filename stem.
    """

    def __init__(self, left_dir, right_dir, depth_dir, transform=None, depth_transform=None, focal_length=None, baseline=None, zero_is_invalid=False):
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.depth_transform = depth_transform
        self.focal_length = focal_length  # in pixels
        self.baseline = baseline          # in meters
        # CARLA: 0 = sky (valid). KITTI: 0 = no LiDAR return (invalid). Set True for KITTI.
        self.zero_is_invalid = zero_is_invalid

        # Accept any common image extension (CARLA is .jpg, KITTI is .png).
        img_exts = ('.jpg', '.jpeg', '.png')

        def stem_map(directory, exts):
            m = {}
            for f in os.listdir(directory):
                if f.lower().endswith(exts):
                    m[os.path.splitext(f)[0]] = f  # keep the real filename per stem
            return m

        left_map = stem_map(left_dir, img_exts)
        right_map = stem_map(right_dir, img_exts)
        depth_map = stem_map(depth_dir, ('.png',))  # disparity is always .png

        common_stems = sorted(set(left_map) & set(right_map) & set(depth_map))

        self.left_images = [left_map[stem] for stem in common_stems]
        self.right_images = [right_map[stem] for stem in common_stems]
        self.depth_maps = [depth_map[stem] for stem in common_stems]

        print(f"Loaded {len(self.left_images)} valid triplets (left, right, depth)")

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        left_img_path  = os.path.join(self.left_dir,  self.left_images[idx])
        right_img_path = os.path.join(self.right_dir, self.right_images[idx])
        disp_map_path  = os.path.join(self.depth_dir, self.depth_maps[idx])  # this is disparity now

        # --- Load RGB images ---
        left_pil  = Image.open(left_img_path).convert("RGB")
        right_pil = Image.open(right_img_path).convert("RGB")

        # Keep original sizes for disparity rescaling if we resize later
        orig_w, orig_h = left_pil.size

        disp_pil = Image.open(disp_map_path)

        # Convert to numpy float32
        disp_np = np.array(disp_pil).astype(np.float32)

        if np.nanmax(disp_np) > 512:
            disp_np = disp_np / 256.0

        # Mark truly invalid pixels with -1 (sentinel).
        # CARLA: zero is a valid disparity (sky / infinite depth) and must be preserved.
        # KITTI: zero means "no LiDAR return" and must be excluded (zero_is_invalid=True).
        if self.zero_is_invalid:
            invalid = ~np.isfinite(disp_np) | (disp_np <= 0)
        else:
            invalid = ~np.isfinite(disp_np) | (disp_np < 0)
        disp_np[invalid] = -1.0

        if self.depth_transform is not None:
            disp_pil_f = Image.fromarray(disp_np, mode="F")

            disp_pil_f = self.depth_transform(disp_pil_f)

            disp_np_resized = np.array(disp_pil_f, dtype=np.float32)

            new_h, new_w = disp_np_resized.shape

            scale_x = float(new_w) / float(orig_w)
            # Scale valid pixels; restore -1 sentinel (NEAREST keeps it exact,
            # but -1 * scale_x != -1, so we re-apply after scaling).
            invalid_resized = disp_np_resized < 0
            disp_np = disp_np_resized * scale_x
            disp_np[invalid_resized] = -1.0

        gt_disp = torch.from_numpy(disp_np).float()

        if self.transform is not None:
            left = self.transform(left_pil)
            right = self.transform(right_pil)
        else:
            from torchvision.transforms import ToTensor
            to_tensor = ToTensor()
            left = to_tensor(left_pil)
            right = to_tensor(right_pil)

        return left, right, gt_disp