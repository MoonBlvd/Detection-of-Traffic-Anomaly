# When, Where, and What? A New Dataset for Anomaly Detection in Driving Videos
Yu Yao, Xizi Wang, Mingze Xu, Zelin Pu, Ella Atkins, David Crandall

:boom: This repo contains the **Detection of Traffic Anomaly (DoTA) dataset** and the **code** of our paper.

## DoTA dataset
Install dependencies by `pip install -r requirements.txt`

Get the DoTA dataset with following steps:

    cd dataset
    unzip DoTA_annotations.zip
    python download_DoTA.py --url_file DoTA_urls.txt --download_dir PATH_TO_SAVE_RAW_VIDEO --to_images True --img_dir PATH_TO_SAVE_FRAMES --anno_dir annotations/

The `download_DoTA.py` script first downloads the original long videos from YouTube, write to `*PATH_TO_SAVE_RAW_VIDEO*`. Then it extracts annotated frames for each video clip and write to `*PATH_TO_SAVE_FRAMES*`.

**NOTE**, four long videos (see `dataset/broken_urls.txt`) in DoTA were **REMOVED** by YouTube. To use these four videos, please download here and put them in the user-defined *PATH_TO_SAVE_RAW_VIDEO* before running the above script.
