import argparse
from pathlib import Path


def remove_multiples_of_60(directory):
    counter = 0
    for file_path in Path(directory).rglob("*"):
        try:
            num = int(file_path.stem)
            if num % 60 == 0:
                file_path.unlink()
                counter += 1
                # print(f"Removed: {file_path}")
        except ValueError:
            pass

    print(counter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove files that are multiples of 60."
    )
    parser.add_argument(
        "--dataset_path",
        help="Path to the directory to search for files.",
        default="/mnt/Bedo_CARLA/carla-capture-6/",
    )
    args = parser.parse_args()

    remove_multiples_of_60(args.dataset_path)
