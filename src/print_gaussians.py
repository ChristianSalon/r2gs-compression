import argparse
import pickle
from pathlib import Path
from pprint import pprint


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(prog="R2GS Print Gaussians")
    parser.add_argument("-i", "--input", type=str, required=True, help="Optimized gaussians (.pickle)")

    args = parser.parse_args()

    # Load gaussians as dict
    input_path = Path(args.input)
    with open(input_path, "rb") as file:
        gaussians = pickle.load(file)

    # Print compression and compaction metrics
    print(f"Number of Gaussians: {len(gaussians["density"])}")
    print(f"Input File Size: {input_path.stat().st_size} B")

    # Print gaussians dict
    print()
    pprint(gaussians)


if __name__ == "__main__":
    main()
