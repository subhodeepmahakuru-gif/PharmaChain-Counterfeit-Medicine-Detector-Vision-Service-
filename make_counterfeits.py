
"""
make_counterfeits.py

PharmaChain - Member 4: Counterfeit Detector

Generates augmented genuine and synthetic counterfeit-style
images from genuine medicine packaging photographs.

IMPORTANT:
These generated images are synthetic augmentation examples.
They are not verified counterfeit medicine photographs.
"""

import argparse
import glob
import io
import os
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# Maximum image dimension
MAX_SIDE = 640


def load(path):
    """Load and resize an image while preserving aspect ratio."""

    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))

    return img


def blur(img, rng):
    """Apply Gaussian blur."""

    radius = rng.uniform(1.5, 3.0)

    return img.filter(
        ImageFilter.GaussianBlur(radius)
    )


def fade_colors(img, rng):
    """Reduce color saturation and contrast."""

    img = ImageEnhance.Color(img).enhance(
        rng.uniform(0.3, 0.7)
    )

    img = ImageEnhance.Contrast(img).enhance(
        rng.uniform(0.7, 0.9)
    )

    return img


def tint(img, rng):
    """Apply random color variation."""

    arr = np.array(img).astype(np.int16)

    shifts = np.array(
        [rng.randint(-30, 30) for _ in range(3)],
        dtype=np.int16
    )

    arr = arr + shifts

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def cover_patch(img, rng):
    """Cover a random image region with a sampled color."""

    w, h = img.size

    pw = rng.randint(max(1, w // 8), max(1, w // 3))
    ph = rng.randint(max(1, h // 8), max(1, h // 3))

    x = rng.randint(0, w - pw)
    y = rng.randint(0, h - ph)

    arr = np.array(img).copy()

    sample_x = rng.randint(0, w - 1)
    sample_y = rng.randint(0, h - 1)

    patch_color = img.getpixel((sample_x, sample_y))

    arr[y:y + ph, x:x + pw] = patch_color

    return Image.fromarray(arr)


def shear(img, rng):
    """Apply a small affine shear transformation."""

    k = rng.uniform(-0.12, 0.12)

    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, k, 0, 0, 1, 0),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255)
    )


def jpeg_damage(img, rng):
    """Simulate JPEG compression artifacts."""

    buf = io.BytesIO()

    quality = rng.randint(8, 22)

    img.save(
        buf,
        format="JPEG",
        quality=quality
    )

    buf.seek(0)

    with Image.open(buf) as compressed:
        return compressed.convert("RGB").copy()


def noise(img, rng):
    """Add random pixel noise."""

    arr = np.array(img).astype(np.float32)

    random_state = np.random.RandomState(
        rng.randint(0, 10**6)
    )

    noise_array = random_state.normal(
        0,
        14,
        arr.shape
    )

    arr = arr + noise_array

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


# Available augmentation functions
DEFECTS = [
    blur,
    fade_colors,
    tint,
    cover_patch,
    shear,
    jpeg_damage,
    noise
]


def make_fake(img, rng):
    """Create a synthetic altered image using random augmentations."""

    img = img.copy()

    number_of_defects = rng.choice([2, 3])

    selected_defects = rng.sample(
        DEFECTS,
        number_of_defects
    )

    for fn in selected_defects:
        img = fn(img, rng)

    return img


def mild_genuine(img, rng):
    """Create a mildly augmented genuine image."""

    img = img.copy()

    img = img.rotate(
        rng.uniform(-3, 3),
        expand=False,
        fillcolor=(255, 255, 255)
    )

    img = ImageEnhance.Brightness(img).enhance(
        rng.uniform(0.92, 1.08)
    )

    return img


def main():

    parser = argparse.ArgumentParser(
        description="Generate synthetic counterfeit-style medicine images."
    )

    parser.add_argument(
        "--src",
        default="real_genuine",
        help="Directory containing genuine medicine photographs"
    )

    parser.add_argument(
        "--out",
        default="data",
        help="Output dataset directory"
    )

    parser.add_argument(
        "--variants",
        type=int,
        default=6,
        help="Images per class per source photograph"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.variants < 1:
        parser.error("--variants must be at least 1")

    if not os.path.isdir(args.src):
        raise SystemExit(
            "Source directory does not exist: %s" % args.src
        )

    # Read source images
    supported_extensions = (
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.webp",
        "*.tif",
        "*.tiff"
    )

    files = []

    for extension in supported_extensions:
        files.extend(
            glob.glob(
                os.path.join(args.src, extension)
            )
        )

        files.extend(
            glob.glob(
                os.path.join(args.src, extension.upper())
            )
        )

    files = sorted(set(files))

    # Validate source images
    if len(files) < 10:
        raise SystemExit(
            "Need at least 10 genuine photos in %s/ (found %d)"
            % (args.src, len(files))
        )

    # Initialize random generator
    rng = random.Random(args.seed)

    # Shuffle source images
    rng.shuffle(files)

    n = len(files)

    # Split source photographs BEFORE augmentation
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    split_files = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    print("\nStarting dataset generation...\n")

    # Process each dataset split
    for split, flist in split_files.items():

        genuine_dir = os.path.join(
            args.out,
            split,
            "genuine"
        )

        counterfeit_dir = os.path.join(
            args.out,
            split,
            "counterfeit"
        )

        os.makedirs(
            genuine_dir,
            exist_ok=True
        )

        os.makedirs(
            counterfeit_dir,
            exist_ok=True
        )

        for fi, path in enumerate(flist):

            try:
                base = load(path)

            except Exception as error:
                print(
                    "Skipping unreadable image %s: %s"
                    % (path, error)
                )
                continue

            stem = "%s_%03d" % (split, fi)

            # Save original genuine image
            base.save(
                os.path.join(
                    genuine_dir,
                    stem + "_orig.jpg"
                ),
                quality=92
            )

            # Generate additional genuine images
            for v in range(args.variants - 1):

                genuine_img = mild_genuine(
                    base,
                    rng
                )

                genuine_img.save(
                    os.path.join(
                        genuine_dir,
                        "%s_g%d.jpg" % (stem, v)
                    ),
                    quality=92
                )

            # Generate synthetic altered images
            for v in range(args.variants):

                fake_img = make_fake(
                    base,
                    rng
                )

                fake_img.save(
                    os.path.join(
                        counterfeit_dir,
                        "%s_c%d.jpg" % (stem, v)
                    ),
                    quality=92
                )

        # Print split information
        print(
            "%-5s: %d source photos processed"
            % (split, len(flist))
        )

        print(
            "      Genuine images: %d"
            % (
                len([
                    f for f in os.listdir(genuine_dir)
                    if f.lower().endswith(".jpg")
                ])
            )
        )

        print(
            "      Synthetic altered images: %d"
            % (
                len([
                    f for f in os.listdir(counterfeit_dir)
                    if f.lower().endswith(".jpg")
                ])
            )
        )

    print("\n====================================")
    print("DATASET GENERATION COMPLETED")
    print("====================================")

    print("Source directory :", args.src)
    print("Output directory :", args.out)
    print("Variants         :", args.variants)
    print("Source photos    :", n)

    print("\nDataset structure:")
    print(args.out + "/")
    print("  train/")
    print("    genuine/")
    print("    counterfeit/")
    print("  val/")
    print("    genuine/")
    print("    counterfeit/")
    print("  test/")
    print("    genuine/")
    print("    counterfeit/")

    print("\nDone!")


if __name__ == "__main__":
    main()
