import argparse
import os
import pandas as pd
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

URL = "https://download.cncb.ac.cn/covid-ct/"

def download_and_unzip(args):
    if not os.path.exists(args.extract_to + "data"):
    # Create a new directory because it does not exist
        os.makedirs(args.extract_to + "data")
    df = pd.read_csv(args.file_names_csv)
    file_names = df["zip_file"].unique()
    filenames = []
    for file_name in file_names:
        if "NCP" in file_name:
            filenames.append(file_name)
    for file_name in filenames:
        if "NCP" in file_name:
            file_name = "COVID19" + file_name[3:] 
        print("downloading file {}", end="\r".format(file_name))
        print(URL + file_name)
        try:
            http_response = urlopen(URL + file_name)
            zipfile = ZipFile(BytesIO(http_response.read()))
            zipfile.extractall(path=args.extract_to + "data")
        except Exception as e:
            print("Could not extract file: {}".format(file_name))
            print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
	Utility script to convert commonvoice into wav and create the training and test json files for speechrecognition. """
	)
    parser.add_argument('--extract_to', type=str, default=None, required=True,
                        help='path where to extract the zip files to')
    parser.add_argument('--file_names_csv', type=str, default=None, required=True,
                        help='path to the zip_file_names.csv file. Check http://ncov-ai.big.ac.cn/download?lang=en')
    args = parser.parse_args()

    download_and_unzip(args)
    