# When, Where, and What? A New Dataset for Anomaly Detection in Driving Videos
Yu Yao, Xizi Wang, Mingze Xu, Zelin Pu, Ella Atkins, David Crandall

<p align="center"><img width="100%" src="img/DoTA.png"/></p>

:boom: This repo contains the **Detection of Traffic Anomaly (DoTA) dataset** and the **code** of our paper.

## DoTA dataset
Install dependencies by `pip install -r requirements.txt`

Get the DoTA dataset with following steps:

    cd dataset
    unzip DoTA_annotations.zip
    python download_DoTA.py --url_file DoTA_urls.txt --download_dir PATH_TO_SAVE_RAW_VIDEO --to_images True --img_dir PATH_TO_SAVE_FRAMES --anno_dir annotations/

The `download_DoTA.py` script first downloads the original long videos from YouTube, write to `PATH_TO_SAVE_RAW_VIDEO`. Then it extracts annotated frames for each video clip and write to `PATH_TO_SAVE_FRAMES`. This will take several hours depending on your internet speed.

**NOTE**, four long videos (see `dataset/broken_urls.txt`) in DoTA were **REMOVED** by YouTube. To use these four videos, please download [here](https://drive.google.com/open?id=1uSdyNu8XM_QV6y9r4tpxN7iNNNAgWy9P) and put them in the user-defined `PATH_TO_SAVE_RAW_VIDEO` **before** running the above script.

## Citation
If you found this repo is useful, please feel free to cite our paper:

    @article{yao2020dota,
            title={When, Where, and What? A New Dataset for Anomaly Detection in Driving Videos},
            author={Yao, Yu and Wang, Xizi and Xu, Mingze and Pu, Zelin and Atkins, Ella and Crandall, David},
            journal={arXiv preprint arXiv:2004.03044},
            year={2020}
            }