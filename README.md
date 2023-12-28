# When, Where, and What? A New Dataset for Anomaly Detection in Driving Videos
Yu Yao, Xizi Wang, Mingze Xu, Zelin Pu, Ella Atkins, David Crandall

<p align="center"><img width="100%" src="img/DoTA.png"/></p>

:boom: This repo contains the **Detection of Traffic Anomaly (DoTA) dataset** and the **code** of our [paper](https://arxiv.org/pdf/2004.03044.pdf).

## DoTA dataset
**UPDATE 12/27/2023:** We further provide the [splitted zip files](https://drive.google.com/drive/folders/1_WzhwZC2NIpzZIpX7YCvapq66rtBc67n) for researchers who have difficult to download the 55GB large file from the blow link. They provide the same data, except that in this new link the 55GB data are speratly archived into 5 10GB files and a 5GB file for easier downloading experience.

**UPDATE 11/13/2021:** Due to high ratio of lost video URL, we have connected with the original channel authors to get the authority to share all the video clips we downloaded and extracted. Please find the download link [here](https://drive.google.com/file/d/1RQp4hOP9X7TW6S3_vbqZvFSkrbbFzrRj/view?usp=sharing). After downloading frames from the above link, users can skip the datadownloading and frame extraction steps below.

Install `ffmpeg`.

Install python dependencies by `pip install -r requirements.txt`.

Get your youtube cookies to be able to download age restricted videos. Please find how to generate cookie based on this [github issue](https://github.com/blackjack4494/yt-dlc/issues/7). Chrome users can use [this app](https://chrome.google.com/webstore/detail/get-cookiestxt/bgaddhkoddajcdgocldbbfleckgcbcid) to generate cookie file.

First, download the orginal videos from YouTube:
```
cd dataset
unzip DoTA_annotations.zip
python download_DoTA.py --url_file DoTA_urls.txt --download_dir PATH_TO_SAVE_RAW_VIDEO --cookiefile PATH_TO_COOKIE/cookies.txt
```

**NOTE**: 6 long videos (see `dataset/broken_urls.txt`) in DoTA were **REMOVED** by YouTube. Please download these videos [here](https://drive.google.com/drive/folders/1uSdyNu8XM_QV6y9r4tpxN7iNNNAgWy9P?usp=sharing) and put them in the user-defined `PATH_TO_SAVE_RAW_VIDEO` **before** running the next step.

Second, extract annotated frames from original videos:
```    
python video2frames.py -v PATH_TO_SAVE_RAW_VIDEO -a annotations -f 10 -o PATH_TO_SAVE_FRAMES -n NUM_OF_PROCESSES
```
The `video2frames.py` script extracts annotated frames for each video clip and writes to `PATH_TO_SAVE_FRAMES`. This will take minitues to hours depending on your machine. It took us around 35 minutes to extract all clips with `NUM_OF_PROCESSES=8`.

Now the annotated clips are extracted and ready to use!

### Annotations for VAD 
Besides the per-video '.json' files, we also provide a `metadata_train.json` and a `metadata_val.json` which contains the video metadata in the following format:

    {
    video_id: {
        "video_start": int,
        "video_end": int,
        "anomaly_start": int,
        "anomaly_end": int,
        "anomaly_class": str,
        "num_frames": int,
        "subset": "train" or "test"
    },

### Extracted data for FOL model training and testing
We also provide the extracted object bounding box tracks and corresponding optical flow features that we used to train our FOL models.
To use these extracted data, please [download](https://drive.google.com/drive/folders/1IVCedrlPg03Fsg4tqDA2cWYlcdrsKUsp?usp=sharing) and extract the zip files to your data directory.
We provide an example of how to run our dataloader:
```
python dataloader_example.py --load_config config/config_example.yaml
```
Please make sure the data directories match your data directory on your machine.

## Citation
If you found this repo is useful, please cite our paper:
```
@article{yao2022dota,
  title={DoTA: unsupervised detection of traffic anomaly in driving videos},
  author={Yao, Yu and Wang, Xizi and Xu, Mingze and Pu, Zelin and Wang, Yuchen and Atkins, Ella and Crandall, David},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2022},
  publisher={IEEE}
}
```
