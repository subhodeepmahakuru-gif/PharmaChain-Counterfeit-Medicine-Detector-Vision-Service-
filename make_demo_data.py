import argparse
import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont


# Image dimensions
W, H = 320, 240

# Color palette
PALETTE = [
    (0, 122, 135),
    (0, 150, 110),
    (30, 90, 160)
]


def font(size):
    """Load a font safely."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_box(rng, genuine=True):
    """Draw a synthetic medicine box."""

    base = rng.choice(PALETTE)

    img = Image.new("RGB", (W, H), (245, 247, 247))
    d = ImageDraw.Draw(img)

    # Top colored section
    d.rectangle([0, 0, W, 62], fill=base)

    # Medicine brand name
    if genuine:
        name = "MEDIPURE"
    else:
        name = rng.choice([
            "MEDIPVRE",
            "MEDIPURF",
            "MED1PURE",
            "MEDIPURE"
        ])

    d.text(
        (16, 14),
        name,
        fill="white",
        font=font(32)
    )

    # Logo position
    if genuine:
        lx, ly = 262, 31
        r = 20
    else:
        lx = rng.randint(215, 280)
        ly = rng.randint(20, 45)
        r = rng.randint(14, 26)

    # Circular logo
    d.ellipse(
        [lx-r, ly-r, lx+r, ly+r],
        fill="white"
    )

    # Vertical logo line
    d.rectangle(
        [lx-3, ly-r+5, lx+3, ly+r-5],
        fill=base
    )

    # Horizontal logo line
    d.rectangle(
        [lx-r+5, ly-3, lx+r-5, ly+3],
        fill=base
    )

    # Medicine information
    d.text(
        (16, 85),
        "Vaccine X 10 ml",
        fill=(30, 40, 45),
        font=font(22)
    )

    # Batch number and expiry date
    batch_number = rng.randint(1000, 9999)

    d.text(
        (16, 120),
        f"Batch B{batch_number} Exp 06/2028",
        fill=(70, 80, 85),
        font=font(15)
    )

    # Barcode
    x = 18

    while x < 170:

        if genuine:
            bar_width = rng.choice([2, 3, 4])
        else:
            bar_width = rng.choice([2, 3, 4, 6, 7])

        if genuine or rng.random() > 0.12:
            d.rectangle(
                [x, 160, x + bar_width, 215],
                fill=(20, 20, 20)
            )

        x += bar_width + rng.choice([2, 3])

    # Hologram strip
    if genuine:

        for i in range(0, 60, 4):

            col = (
                (i * 7) % 255,
                (120 + i * 5) % 255,
                (255 - i * 6) % 255
            )

            d.rectangle(
                [225, 150 + i, 300, 153 + i],
                fill=col
            )

    else:
        d.rectangle(
            [225, 150, 300, 212],
            fill=(185, 185, 185)
        )

    return img


def make_genuine(rng):
    """Generate a synthetic genuine-class demo image."""

    img = draw_box(rng, genuine=True)

    img = img.rotate(
        rng.uniform(-2, 2),
        fillcolor=(245, 247, 247)
    )

    return img


def make_counterfeit(rng):
    """Generate a synthetic counterfeit-class demo image."""

    img = draw_box(rng, genuine=False)

    # Apply blur
    if rng.random() < 0.5:

        img = img.filter(
            ImageFilter.GaussianBlur(
                rng.uniform(0.8, 1.6)
            )
        )

    # Apply color variation
    if rng.random() < 0.4:

        px = img.load()

        shift = rng.choice([
            (25, -15, -20),
            (-20, 15, 25),
            (30, 20, -25)
        ])

        for y in range(H):
            for x in range(W):

                p = px[x, y]

                px[x, y] = tuple(
                    max(0, min(255, p[i] + shift[i]))
                    for i in range(3)
                )

    # Rotate image slightly
    img = img.rotate(
        rng.uniform(-2, 2),
        fillcolor=(245, 247, 247)
    )

    return img


def main():

    parser = argparse.ArgumentParser(
        description="Generate a synthetic medicine box dataset."
    )

    parser.add_argument(
        "--out",
        default="data",
        help="Output directory"
    )

    parser.add_argument(
        "--per-class",
        type=int,
        default=120,
        help="Number of images per class"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Validate input
    if args.per_class <= 0:
        parser.error("--per-class must be greater than zero")

    # Random generator
    rng = random.Random(args.seed)

    # Create output directories
    genuine_dir = os.path.join(
        args.out,
        "genuine"
    )

    counterfeit_dir = os.path.join(
        args.out,
        "counterfeit"
    )

    os.makedirs(genuine_dir, exist_ok=True)
    os.makedirs(counterfeit_dir, exist_ok=True)

    print("\nGenerating synthetic medicine dataset...\n")

    # Generate images
    for i in range(args.per_class):

        # Genuine image
        genuine_img = make_genuine(rng)

        genuine_path = os.path.join(
            genuine_dir,
            f"genuine_{i:04d}.png"
        )

        genuine_img.save(genuine_path)

        # Counterfeit image
        counterfeit_img = make_counterfeit(rng)

        counterfeit_path = os.path.join(
            counterfeit_dir,
            f"counterfeit_{i:04d}.png"
        )

        counterfeit_img.save(counterfeit_path)

        if (i + 1) % 20 == 0:
            print(
                f"Generated {i + 1}/{args.per_class} images per class"
            )

    # Final report
    total_images = args.per_class * 2

    print("\n================================")
    print("DATASET GENERATED SUCCESSFULLY")
    print("================================")

    print(f"Genuine images     : {args.per_class}")
    print(f"Counterfeit images : {args.per_class}")
    print(f"Total images       : {total_images}")

    print(f"\nOutput directory: {args.out}")

    print("\nDataset folders:")
    print(f"  {genuine_dir}/")
    print(f"  {counterfeit_dir}/")


if __name__ == "__main__":
    main()
