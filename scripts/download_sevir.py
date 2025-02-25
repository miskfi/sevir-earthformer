import argparse

import os
import pathlib
import sys

import boto3
from botocore.handlers import disable_signing


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, required=True, default="/home/datalab/big-disk/SEVIR/", help="path to the download directory")
args = parser.parse_args()

resource = boto3.resource("s3")
resource.meta.client.meta.events.register("choose-signer.s3.*", disable_signing)
bucket = resource.Bucket("sevir")

# objs = bucket.objects.filter(Prefix='')

files = [
    "data/ir069/2018/SEVIR_IR069_RANDOMEVENTS_2018_0101_0430.h5",
    "data/ir069/2018/SEVIR_IR069_RANDOMEVENTS_2018_0501_0831.h5",
    "data/ir069/2018/SEVIR_IR069_RANDOMEVENTS_2018_0901_1231.h5",
    "data/ir069/2018/SEVIR_IR069_STORMEVENTS_2018_0101_0630.h5",
    "data/ir069/2018/SEVIR_IR069_STORMEVENTS_2018_0701_1231.h5",
    "data/ir069/2019/SEVIR_IR069_RANDOMEVENTS_2019_0101_0430.h5",
    "data/ir069/2019/SEVIR_IR069_RANDOMEVENTS_2019_0501_0831.h5",
    "data/ir069/2019/SEVIR_IR069_RANDOMEVENTS_2019_0901_1231.h5",
    "data/ir069/2019/SEVIR_IR069_STORMEVENTS_2019_0101_0630.h5",
    "data/ir069/2019/SEVIR_IR069_STORMEVENTS_2019_0701_1231.h5",
    "data/ir107/2018/SEVIR_IR107_RANDOMEVENTS_2018_0101_0430.h5",
    "data/ir107/2018/SEVIR_IR107_RANDOMEVENTS_2018_0501_0831.h5",
    "data/ir107/2018/SEVIR_IR107_RANDOMEVENTS_2018_0901_1231.h5",
    "data/ir107/2018/SEVIR_IR107_STORMEVENTS_2018_0101_0630.h5",
    "data/ir107/2018/SEVIR_IR107_STORMEVENTS_2018_0701_1231.h5",
    "data/ir107/2019/SEVIR_IR107_RANDOMEVENTS_2019_0101_0430.h5",
    "data/ir107/2019/SEVIR_IR107_RANDOMEVENTS_2019_0501_0831.h5",
    "data/ir107/2019/SEVIR_IR107_RANDOMEVENTS_2019_0901_1231.h5",
    "data/ir107/2019/SEVIR_IR107_STORMEVENTS_2019_0101_0630.h5",
    "data/ir107/2019/SEVIR_IR107_STORMEVENTS_2019_0701_1231.h5",
    "data/vil/2018/SEVIR_VIL_RANDOMEVENTS_2018_0101_0430.h5",
    "data/vil/2018/SEVIR_VIL_RANDOMEVENTS_2018_0501_0831.h5",
    "data/vil/2018/SEVIR_VIL_RANDOMEVENTS_2018_0901_1231.h5",
    "data/vil/2018/SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5",
    "data/vil/2018/SEVIR_VIL_STORMEVENTS_2018_0701_1231.h5",
    "data/vil/2019/SEVIR_VIL_RANDOMEVENTS_2019_0101_0430.h5",
    "data/vil/2019/SEVIR_VIL_RANDOMEVENTS_2019_0501_0831.h5",
    "data/vil/2019/SEVIR_VIL_RANDOMEVENTS_2019_0901_1231.h5",
    "data/vil/2019/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5",
    "data/vil/2019/SEVIR_VIL_STORMEVENTS_2019_0701_1231.h5",
    "data/lght/2018/SEVIR_LGHT_ALLEVENTS_2018_0201_0301.h5",
    "data/lght/2018/SEVIR_LGHT_ALLEVENTS_2018_0301_0401.h5",
    "data/lght/2018/SEVIR_LGHT_ALLEVENTS_2018_0401_0501.h5",
    "data/lght/2018/SEVIR_LGHT_ALLEVENTS_2018_0501_0601.h5",
    "data/lght/2018/SEVIR_LGHT_ALLEVENTS_2018_0601_0701.h5",
    "data/lght/2018/SEVIR_LGHT_ALLEVENTS_2018_0701_0801.h5",
    "data/lght/2018/SEVIR_LGHT_ALLEVENTS_2018_0801_0901.h5",
    "data/lght/2018/SEVIR_LGHT_ALLEVENTS_2018_0901_1001.h5",
    "data/lght/2018/SEVIR_LGHT_ALLEVENTS_2018_1001_1101.h5",
    "data/lght/2018/SEVIR_LGHT_ALLEVENTS_2018_1101_1201.h5",
    "data/lght/2018/SEVIR_LGHT_ALLEVENTS_2018_1201_0101.h5",
    "data/lght/2019/SEVIR_LGHT_ALLEVENTS_2019_0101_0201.h5",
    "data/lght/2019/SEVIR_LGHT_ALLEVENTS_2019_0201_0301.h5",
    "data/lght/2019/SEVIR_LGHT_ALLEVENTS_2019_0301_0401.h5",
    "data/lght/2019/SEVIR_LGHT_ALLEVENTS_2019_0401_0501.h5",
    "data/lght/2019/SEVIR_LGHT_ALLEVENTS_2019_0501_0601.h5",
    "data/lght/2019/SEVIR_LGHT_ALLEVENTS_2019_0601_0701.h5",
    "data/lght/2019/SEVIR_LGHT_ALLEVENTS_2019_0701_0801.h5",
    "data/lght/2019/SEVIR_LGHT_ALLEVENTS_2019_0801_0901.h5",
    "data/lght/2019/SEVIR_LGHT_ALLEVENTS_2019_0901_1001.h5",
    "data/lght/2019/SEVIR_LGHT_ALLEVENTS_2019_1001_1101.h5",
    "data/lght/2019/SEVIR_LGHT_ALLEVENTS_2019_1101_1201.h5",
]

download_dir = args.dir

for i, f in enumerate(files):
    file_path = download_dir + f

    if not os.path.exists(file_path):
        print(f"Downloading {f} [{i+1}/{len(files)}]")
        pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        bucket.download_file(f, file_path)
    else:
        print(f"Skipping {f} (already downloaded) [{i+1}/{len(files)}]")
    sys.stdout.flush()

print("Download finished")
