import os
import glob
from lang_probing_src.config import UD_BASE_FOLDER
import argparse

# load in PUD files

def main(args):
    # load in PUD files
    for language in args.languages:
        pud_files = glob.glob(os.path.join(UD_BASE_FOLDER, f"UD_{language}-PUD", f"*_pud-ud-test.conllu"))

        print(pud_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--languages", type=str, nargs="+", default=["English"], help="A list of languages to collect input space for.")
    args = parser.parse_args()

    main(args)
