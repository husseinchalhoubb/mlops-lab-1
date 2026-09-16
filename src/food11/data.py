"""Prepare the Food-11 image dataset for ImageFolder-based training.

The raw dataset stores the class label in each filename. This script resizes
the images, places them in class directories, and creates a smaller development
dataset containing at most 100 images per class in each split.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


CLASS_NAMES = {
    0: "Bread",
    1: "Dairy product",
    2: "Dessert",
    3: "Egg",
    4: "Fried food",
    5: "Meat",
    6: "Noodles-Pasta",
    7: "Rice",
    8: "Seafood",
    9: "Soup",
    10: "Vegetable-Fruit",
}
SPLITS = ("training", "evaluation", "validation")
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_IMAGE_SIZE = 128
DEFAULT_MINI_LIMIT = 100


def class_name_from_filename(image_path: Path) -> str:
    """Return the Food-11 class name encoded at the start of a filename."""
    label_text = image_path.stem.split("_", maxsplit=1)[0]
    try:
        label = int(label_text)
        return CLASS_NAMES[label]
    except (ValueError, KeyError) as error:
        raise ValueError(
            f"Cannot determine a Food-11 label from {image_path.name!r}."
        ) from error


def raw_images(split_directory: Path) -> list[Path]:
    """Return supported images in a stable order for reproducible mini sets."""
    return sorted(
        (
            path
            for path in split_directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: path.name,
    )


def save_resized_image(source: Path, destination: Path, image_size: int) -> None:
    """Resize one image to a square RGB JPEG and save it at destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        image.save(destination, format="JPEG", quality=90)


def prepare_dataset(
    data_directory: Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    mini_limit: int = DEFAULT_MINI_LIMIT,
) -> tuple[Counter[tuple[str, str]], Counter[tuple[str, str]]]:
    """Build full and mini processed datasets and return their image counts."""
    raw_directory = data_directory / "food11_raw"
    processed_directory = data_directory / "food11_processed"
    mini_directory = data_directory / "food11_processed_mini"
    processed_temporary = data_directory / ".food11_processed.tmp"
    mini_temporary = data_directory / ".food11_processed_mini.tmp"

    if image_size <= 0:
        raise ValueError("image_size must be greater than zero")
    if mini_limit <= 0:
        raise ValueError("mini_limit must be greater than zero")

    missing_splits = [
        split for split in SPLITS if not (raw_directory / split).is_dir()
    ]
    if missing_splits:
        raise FileNotFoundError(
            f"Missing raw Food-11 split directories: {', '.join(missing_splits)}"
        )

    for temporary_directory in (processed_temporary, mini_temporary):
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)
        temporary_directory.mkdir(parents=True)

    processed_counts: Counter[tuple[str, str]] = Counter()
    mini_counts: Counter[tuple[str, str]] = Counter()

    try:
        for split in SPLITS:
            images = raw_images(raw_directory / split)
            print(f"Processing {split}: {len(images)} images")

            for index, source in enumerate(images, start=1):
                class_name = class_name_from_filename(source)
                count_key = (split, class_name)
                destination = (
                    processed_temporary / split / class_name / source.name
                )
                save_resized_image(source, destination, image_size)
                processed_counts[count_key] += 1

                if mini_counts[count_key] < mini_limit:
                    mini_destination = (
                        mini_temporary / split / class_name / source.name
                    )
                    mini_destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(destination, mini_destination)
                    mini_counts[count_key] += 1

                if index % 1000 == 0:
                    print(f"  completed {index}/{len(images)}")

        for output_directory in (processed_directory, mini_directory):
            if output_directory.exists():
                shutil.rmtree(output_directory)
        processed_temporary.rename(processed_directory)
        mini_temporary.rename(mini_directory)
    except Exception:
        for temporary_directory in (processed_temporary, mini_temporary):
            if temporary_directory.exists():
                shutil.rmtree(temporary_directory)
        raise

    return processed_counts, mini_counts


def print_summary(
    processed_counts: Counter[tuple[str, str]],
    mini_counts: Counter[tuple[str, str]],
) -> None:
    """Print split totals for the generated datasets."""
    print("\nGenerated dataset summary")
    for split in SPLITS:
        processed_total = sum(
            processed_counts[(split, class_name)]
            for class_name in CLASS_NAMES.values()
        )
        mini_total = sum(
            mini_counts[(split, class_name)]
            for class_name in CLASS_NAMES.values()
        )
        print(
            f"  {split}: {processed_total} processed, "
            f"{mini_total} mini images"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data",
        help="Directory containing food11_raw (default: project data directory)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Square output image size in pixels (default: 128)",
    )
    parser.add_argument(
        "--mini-limit",
        type=int,
        default=DEFAULT_MINI_LIMIT,
        help="Maximum images per class and split in the mini dataset (default: 100)",
    )
    return parser.parse_args()


def main() -> None:
    """Run Food-11 dataset preparation."""
    args = parse_args()
    processed_counts, mini_counts = prepare_dataset(
        args.data_dir.resolve(),
        image_size=args.image_size,
        mini_limit=args.mini_limit,
    )
    print_summary(processed_counts, mini_counts)


if __name__ == "__main__":
    main()
