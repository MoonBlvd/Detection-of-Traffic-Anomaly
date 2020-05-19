import os
import subprocess
import argparse
import glob
import json
from multiprocessing import Pool
import shutil
import time
import pdb


def videos_to_frames(args, video_dir):
    vid = video_dir.split('/')[-1].split('.')[0]
    output_dir = os.path.join(args.out_dir, vid)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        print(video_dir + " has been processed!")
        return output_dir

    # query = "ffmpeg -i " + video_dir + " -vf fps=" + str(args.fps) + " -qscale:v 0 " + output_dir + "/%06d.jpg" 
    query = "ffmpeg -i " + video_dir + " -vf fps=" + str(args.fps) + " " + output_dir + "/%06d.jpg" 
    response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
    s = str(response).encode('utf-8')

    return output_dir

def extract_clips(args, frames_dir):
    # load annotations
    vid = frames_dir.split('/')[-1]
    all_annos = sorted(glob.glob(os.path.join(args.anno_dir, vid+'*.json')))
    starts = []
    ends = []
    for anno_file in all_annos:
        anno = json.load(open(anno_file, 'r'))
        starts.append(anno['video_start'])
        ends.append(anno['video_end'])
    
    # get all frames
    all_frames = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))

    print("Number of frames: ", len(all_frames))
    # print("FPS: ", args.fps)

    # extract annotated clips
    for start, end in zip(starts, ends):
        clip_path = os.path.join(args.out_dir, vid+'_'+str(start).zfill(6))
        if not os.path.exists(clip_path): os.makedirs(clip_path)

        for i, old_img_path in enumerate(all_frames[start-1:end]):
            shutil.move(old_img_path, os.path.join(clip_path, str(i).zfill(6)+'.jpg'))
        print('Clip path: {}, Length: {}'.format(clip_path, len(glob.glob(os.path.join(clip_path, '*')))))
    # remove the rest frames since they are not useful
    print('Delete the rest frame in ', frames_dir)
    shutil.rmtree(frames_dir)

def extract_frames_from_videos(args, video_dir):
    frames_dir = videos_to_frames(args, video_dir)
    extract_clips(args, frames_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_dir", required=True, help="the directory to the video or videos")
    # parser.add_argument("--video_key_file", help="the directory to the file of a list of video keys")
    parser.add_argument('-a', '--anno_dir', default='annotations', help='directory where all annotation files are saved')
    parser.add_argument("-f", "--fps", default=10, type=int, required=True, help="the target fps of the extracted frames")
    parser.add_argument("-o", "--out_dir", required=True, help="the output directory")
    parser.add_argument('-n', type=int, default=1, help='number of threads when extracting frames')
    args = parser.parse_args()

    input_video_dir = args.video_dir
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    all_video_names = sorted(glob.glob(os.path.join(input_video_dir, '*')))
    print("Number of video: ", len(all_video_names))
    start = time.time()
    pool = Pool(args.n)
    pool.starmap(extract_frames_from_videos, [(args, video_name) for video_name in all_video_names])
    print("Elapse of frame extraction:", time.time()-start)

    # for video_name in all_video_names:
    #     extract_frames_from_videos(args, video_name)
        