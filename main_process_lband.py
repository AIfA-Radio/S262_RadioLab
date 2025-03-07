import argparse
from preprocess import Preprocess
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Preprocess data for a given scan number.")
    parser.add_argument("scan_number", type=str, help="Scan number to process")
    args = parser.parse_args()

    preproc_data = Preprocess(scan_number=args.scan_number)
    preproc_data.save_data()

if __name__ == "__main__":
    main()
