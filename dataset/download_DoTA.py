import sys
import os
import glob
import argparse
import yaml
import cv2
import youtube_dl
import json
from tqdm import tqdm
from multiprocessing import Pool
import time
import pdb

def download_videos(args):
    DOWNLOAD_DIR = args.download_dir
    COOKIES_FILE = args.cookiefile

    # Download videos
    if not os.path.isdir(DOWNLOAD_DIR):
        print("The indicated download directory does not exist!")
        print("Directory made!")
        os.makedirs(DOWNLOAD_DIR)

    '''Download videos'''
    ydl_opt = {'cookiefile': COOKIES_FILE, 'outtmpl': os.path.join(DOWNLOAD_DIR, '%(id)s.%(ext)s'),
            'format': 'mp4'}
    ydl = youtube_dl.YoutubeDL(ydl_opt)

    url_list = open(args.url_file,'r').readlines()
    
    ydl.download(url_list)
    print("Download finished!")
    
# def videos_to_frame(args, file_name):
#     # load annotations
#     vid = file_name.split('/')[-1].split('.')[0]
#     all_annos = sorted(glob.glob(os.path.join(args.anno_dir, vid+'*.json')))
#     starts = []
#     ends = []
#     for anno_file in all_annos:
#         anno = json.load(open(anno_file, 'r'))
#         starts.append(anno['video_start'])
#         ends.append(anno['video_end'])
#     # load video
#     cap = cv2.VideoCapture(file_name)
#     cap.set(cv2.CAP_PROP_FPS, 10)
#     downsample_rate = 3
#     length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = float(cap.get(cv2.CAP_PROP_FPS))
#     print("Number of frames: ", length)
#     print("FPS: ", fps)
#     i, j, k = 0, 0, 0
#     pdb.set_trace()
#     while True:
#         # Capture frame-by-frame
#         ret, image = cap.read()
#         if not ret:
#             break
#         if i % downsample_rate == 0:
#             if j >= starts[0] and j <= ends[0]:
#                 if j == starts[0]:
#                     # create a new video clip directory
#                     sub_folder_name = vid + '_' + str(j).zfill(6)
#                     img_dir = os.path.join(args.img_dir, sub_folder_name)
#                     if not os.path.exists(img_dir):
#                         os.makedirs(img_dir)
#                         print('Writing to ', img_dir)
#                 img_name = str(k).zfill(6) + '.jpg'
#                 cv2.imwrite(os.path.join(img_dir, img_name), image)
#                 k += 1
#             if j == ends[0]:
#                 starts.pop(0)
#                 ends.pop(0)
#                 if len(starts) == 0:
#                     break
#                 k = 0
#             j += 1
#         i += 1

def main():
    parser = argparse.ArgumentParser(description='AnAnXingChe video downloader parameters.')
    parser.add_argument('--download_dir', required=True, help='target directory to save downloaded videos')
    parser.add_argument('--url_file', required=True, help='a .txt file saving urls to all videos')
    parser.add_argument('--cookiefile', required=True, help='a .txt file with youtube cookies')
    # parser.add_argument('--to_images', type=bool, default=False, help='downsample the video and save image frames, default is false')
    # parser.add_argument('--img_dir', default='frames', help='target directory to save downsampled')
    # parser.add_argument('--anno_dir', default='annotations', help='directory where all annotation files are saved')
    # parser.add_argument('-n', type=int, default=1, help='number of threads when extracting frames')

    args = parser.parse_args()

    # download videos using youtubeDL
    download_videos(args)
    
    # extract annotated frames from videos
    all_videos = sorted(glob.glob(os.path.join(args.download_dir, '*.mp4')))
    print("Number of videos: ", len(all_videos))

    # start = time.time()
    # pool = Pool(args.n)
    # if args.to_images:
    #     # pool.starmap(videos_to_frame, [(args, vid) for vid in all_videos])
    #     for video_idx, file_name in tqdm(enumerate(all_videos)):
    #         videos_to_frame(args, file_name)
    # print("Elapse of frame extraction:", time.time()-start)
if __name__ == '__main__':
    main()
