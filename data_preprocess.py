from tqdm import tqdm
import json
import codecs
import requests
import pandas as pd
from transformers import BertTokenizer, AutoTokenizer
from os import listdir
from os.path import isfile, join
import torch
import nltk
import numpy as np
import random
import re

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def draw_samples(lis, ratio):
    samples = ratio if ratio > 1 else int(ratio * len(lis))

    if samples > len(lis):
        new_lis = np.random.choice(len(lis), samples, replace=True)
    else:
        new_lis = np.random.choice(len(lis), samples, replace=False)

    n_lis = [lis[i] for i in new_lis]

    return n_lis


def preprocess_trafficqa_to_tensor_dataset():
    import clip
    import torch

    import pickle

    image_dir = 'data/frames/'

    train_metadata_dir = 'data/annotations/R3_train.json'
    val_metadata_dir = 'data/annotations/R3_test.json'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-L/16", device=device)

    torch.random.manual_seed(0)

    def read_image_and_qas(metadata_dir, split='train'):
        metadata = json_load(metadata_dir)

        all_images, all_questions, all_option_1, all_option_2, all_option_3, all_option_4, all_ans = [], [], [], [], [], [], []
        for ind, md in enumerate(tqdm(metadata[1:])):
            '''
            Abandoned due to significant use of memory
            '''
            # all_frames = []
            # frame_index = sorted(draw_samples([i for i in range(20)], 10))
            # for ind in frame_index:
            #     frame = preprocess(Image.open('{}{}_{}.jpg'.format(image_dir, key, str(ind))))
            #     all_frames.append(frame)
            # all_frames = torch.cat(all_frames, dim=0)
            all_frames = torch.tensor(md[1], dtype=torch.int).unsqueeze(0)  # video_id

            question = md[4]
            opt1 = md[5]
            opt2 = md[6]
            opt3 = md[7]
            opt4 = md[8]
            ans = torch.tensor(md[9], dtype=torch.long).unsqueeze(0)  # answer: option index

            t_question = clip.tokenize(question, context_length=77, truncate=True)
            # t_opt1 = clip.tokenize('Question: {}, Answer: {}'.format(question, opt1), context_length=77, truncate=True)
            # t_opt2 = clip.tokenize('Question: {}, Answer: {}'.format(question, opt2), context_length=77, truncate=True)
            # t_opt3 = clip.tokenize('Question: {}, Answer: {}'.format(question, opt3), context_length=77, truncate=True)
            # t_opt4 = clip.tokenize('Question: {}, Answer: {}'.format(question, opt4), context_length=77, truncate=True)

            t_opt1 = clip.tokenize('Answer: {}'.format(opt1), context_length=77, truncate=True)
            t_opt2 = clip.tokenize('Answer: {}'.format(opt2), context_length=77, truncate=True)
            t_opt3 = clip.tokenize('Answer: {}'.format(opt3), context_length=77, truncate=True)
            t_opt4 = clip.tokenize('Answer: {}'.format(opt4), context_length=77, truncate=True)

            all_images.append(all_frames)
            all_questions.append(t_question)
            all_option_1.append(t_opt1)
            all_option_2.append(t_opt2)
            all_option_3.append(t_opt3)
            all_option_4.append(t_opt4)
            all_ans.append(ans)

        pickle.dump(
            [all_images, all_questions, all_option_1, all_option_2, all_option_3, all_option_4, all_ans],
            open('data/{}_only_option.cache'.format(split), "wb"), protocol=4)

    read_image_and_qas(train_metadata_dir, split='train')
    read_image_and_qas(val_metadata_dir, split='val')


def sample_frames_from_video_trafficqa():
    # Importing all necessary libraries
    import cv2

    path = 'data/compressed_videos/'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    video_names = json_load("data/annotations/vid_filename_to_id.json")

    frames_per_video = 25
    for ind, f in enumerate(tqdm(onlyfiles)):
        # Read the video from specified path
        cam = cv2.VideoCapture(path + f)

        # frame
        currentframe = 0
        all_frames = []
        while True:
            # reading from frame
            ret, frame = cam.read()

            if ret:
                all_frames.append(frame)
                currentframe += 1
            else:
                break
        lens = len(all_frames)
        if lens >= frames_per_video:
            interval = lens // frames_per_video

            frame_ind = [i * interval for i in range(frames_per_video)]
            for i in range(len(frame_ind)):
                if frame_ind[i] >= lens:
                    frame_ind[i] = lens - 1
            sampled_frames = [all_frames[i] for i in frame_ind]
        else:
            sampled_frames = sorted(draw_samples([i for i in range(len(all_frames))], frames_per_video))
            sampled_frames = [all_frames[i] for i in sampled_frames]

        for ind, frame in enumerate(sampled_frames):
            cv2.imwrite('data/frames/{}_{}.jpg'.format(f.split('.')[0], str(ind)), frame)

        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    import math
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def reformat_jsonl_to_json():
    dir = 'data/annotations/R2_train.jsonl'

    with open(dir, 'r') as f:
        lines = f.readlines()

    new_js = []
    for line in lines:
        js = json.loads(line)
        new_js.append(js)

    json_dump(new_js, 'data/annotations/R2_train.json')


def resize_images():
    from PIL import Image

    path = 'data/frames/'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    # onlyfiles = set([f.split('_')[0] for f in onlyfiles])

    t = 0
    for f in tqdm(onlyfiles):
        image = Image.open(path + f)
        image.thumbnail((224, 224))
        image.save(path.replace('frames', 'frames_resize') + f)
    # print(t)


def concat_images(frame_num=9):
    import cv2

    dataset_name = 'msrvtt'
    path = 'data/frames_resize/'
    path_out = 'data/frames_resize_{}/'.format(str(frame_num))
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = set([f.split('.')[0].split('_')[0] for f in onlyfiles])

    _path = 'data/compressed_videos/'
    onlyfiles = [f for f in listdir(_path) if isfile(join(_path, f))]
    onlyfiles = [vn.replace('.mp4', '') for vn in onlyfiles]


    indices_dict = {
        4: [[0, 8], [16, 24]],
        9: [[0, 3, 6], [9, 12, 15], [17, 20, 24]],
        16: [[0, 2, 3, 5], [7, 9, 10, 12], [14, 16, 17, 19], [20, 21, 22, 24]],
        25: [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19], [20, 21, 22, 23, 24]],
    }

    def concat_vh(list_2d):

        # return final image
        return cv2.vconcat([cv2.hconcat(list_h)
                            for list_h in list_2d])

    for f in tqdm(onlyfiles):
        img_list = []
        for ind in indices_dict[frame_num]:
            im_lis = []
            for i in ind:
                image = cv2.imread('{}{}_{}.jpg'.format(path, f, str(i)))
                im_lis.append(image)
            img_list.append(im_lis)
        img_tile = concat_vh(img_list)
        out_path = '{}{}.jpg'.format(path_out, f)
        cv2.imwrite(out_path, img_tile)


def sample_frames_from_video_msrvtt():
    # Importing all necessary libraries
    import cv2

    path = 'data/msrvtt/MSRVTT/videos/all/'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = [f for f in onlyfiles if '._' not in f]
    frames_per_video = 25
    for ind, f in enumerate(tqdm(onlyfiles)):
        # Read the video from specified path
        cam = cv2.VideoCapture(path + f)

        # frame
        currentframe = 0
        all_frames = []
        while True:
            # reading from frame
            ret, frame = cam.read()

            if ret:
                all_frames.append(frame)
                currentframe += 1
            else:
                break
        lens = len(all_frames)
        if lens >= frames_per_video:
            interval = lens // frames_per_video

            frame_ind = [i * interval for i in range(frames_per_video)]
            for i in range(len(frame_ind)):
                if frame_ind[i] >= lens:
                    frame_ind[i] = lens - 1
            sampled_frames = [all_frames[i] for i in frame_ind]
        else:
            sampled_frames = sorted(draw_samples([i for i in range(len(all_frames))], frames_per_video))
            sampled_frames = [all_frames[i] for i in sampled_frames]

        for ind, frame in enumerate(sampled_frames):
            cv2.imwrite('data/msrvtt/frames/{}_{}.jpg'.format(f.split('.')[0], str(ind)), frame)

        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()


def process_msrvtt_to_tensor_dataset():
    data_dirs = ['data/msrvtt/anno_downstream/msrvtt_ret_train7k.json',
                 'data/msrvtt/anno_downstream/msrvtt_mc_test.json']

    import clip
    import pickle

    def tokenize_dataset(data_dir, split='train'):
        js = json_load(data_dir)

        all_images, all_captions, all_ans = [], [], []

        all_video_names = []
        for ind, e in enumerate(tqdm(js)):
            all_video_names.append(e['video'])

            all_images.append(torch.tensor([ind], dtype=torch.int))
            t_caption = clip.tokenize(e['caption'], context_length=77, truncate=True)

            all_captions.append(t_caption)
            if split == 'val':
                all_ans.append(torch.tensor([e['answer']], dtype=torch.long))

        if split == 'val':
            pickle.dump(
                [all_images, all_captions, all_ans], open('data/msrvtt/{}.cache'.format(split), "wb"),
                protocol=4)
        else:
            pickle.dump(
                [all_images, all_captions], open('data/msrvtt/{}.cache'.format(split), "wb"),
                protocol=4)

        video_names = {'split': split, 'data': all_video_names}
        json_dump(video_names, 'data/msrvtt/{}_video_names.json'.format(split))

    tokenize_dataset(data_dirs[0], 'train')
    tokenize_dataset(data_dirs[1], 'val')


def concat_images_via_different_arrangement(frame_arr=''):
    import cv2
    import os
    dataset_name = 'msrvtt'
    path = 'data/msrvtt/frames_resize/'
    path_out = 'data/msrvtt/frames_resize_{}/'.format(frame_arr)

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = set([f.split('.')[0].split('_')[0] for f in onlyfiles])

    # _path = 'data/compressed_videos/'
    # onlyfiles = [f for f in listdir(_path) if isfile(join(_path, f))]
    # onlyfiles = [vn.replace('.mp4', '') for vn in onlyfiles]

    #
    # video_names = json_load("data/annotations/vid_id_to_filename.json").values()
    # onlyfiles = [vn.replace('.mp4', '') for vn in video_names]

    # indices = [[0, 3, 6], [9, 12, 15], [17, 20, 24]]
    indices_dict = {
        'vertical': [[0], [3], [6], [9], [12], [15], [17], [20], [24]],
        'vertical-descent': [[24], [20], [17], [15], [12], [9], [6], [3], [0]],
        'horizontal': [[0, 3, 6, 9, 12, 15, 17, 20, 24]],
        'horizontal-descent': [[24, 20, 17, 15, 12, 9, 6, 3, 0]],
        'matrix-v-ascent': [[0, 9, 17], [3, 12, 20], [6, 15, 24]],
        'matrix-v-decent': [[24, 15, 6], [20, 12, 3], [17, 9, 0]],
        # 'matrix-h-ascent': [0, 3, 6, 9, 12, 15, 17, 20, 24],
        'matrix-h-decent': [[24, 20, 17], [15, 12, 9], [6, 3, 0]],
        'matrix-random': [[24, 12, 17], [9, 20, 3], [15, 6, 0]],
    }
    # [[0, 2, 3, 5], [7, 9, 10, 12], [14, 16, 17, 19], [20, 21, 22, 24]]
    indices_dict_ = {
        'vertical': [[0], [2], [3], [5], [7], [9], [10], [12], [14], [16], [17],
                     [19], [20], [21], [22], [24]],
        'vertical-descent': [[24], [22], [21], [20], [19], [17], [16], [14], [12],
                             [10], [9], [7], [5], [3], [2], [0]],
        'horizontal': [[0, 2, 3, 5, 7, 9, 10, 12, 14, 16, 17, 19, 20, 21, 22, 24]],
        'horizontal-descent': [[24, 22, 21, 20, 19, 17, 16, 14, 12, 10, 9, 7, 5, 3, 2, 0]],
        'matrix-v-ascent': [[0, 7, 14, 20], [2, 9, 16, 21], [3, 10, 17, 22], [5, 12, 19, 24]],
        'matrix-v-decent': [[24, 19, 12, 5], [22, 17, 10, 3], [21, 16, 9, 2], [20, 14, 7, 0]],
        # 'matrix-h-ascent': [0, 3, 6, 9, 12, 15, 17, 20, 24],
        'matrix-h-decent': [[24, 22, 21, 20], [19, 17, 16, 14], [12, 10, 9, 7], [5, 3, 2, 0]],
        'matrix-random': [[24, 5, 12, 17], [9, 20, 3, 10], [22, 15, 6, 0], [2, 21, 14, 16]],
    }

    def concat_vh(list_2d):

        # return final image
        return cv2.vconcat([cv2.hconcat(list_h)
                            for list_h in list_2d])

    for f in tqdm(onlyfiles):
        img_list = []
        for ind in indices_dict[frame_arr]:
            im_lis = []
            for i in ind:
                image = cv2.imread('{}{}_{}.jpg'.format(path, f, str(i)))
                im_lis.append(image)
            img_list.append(im_lis)
        img_tile = concat_vh(img_list)
        out_path = '{}{}.jpg'.format(path_out, f)
        cv2.imwrite(out_path, img_tile)


def export_arrrangement_mstvtt():
    indices_dict = {
        'vertical': [[0, 3, 6, 9, 12, 15, 17, 20, 24]],
        'vertical-descent': [[0, 3, 6, 9, 12, 15, 17, 20, 24]],
        'horizontal': [0, 3, 6, 9, 12, 15, 17, 20, 24],
        'horizontal-descent': [[24, 22, 21, 20, 19, 17, 16, 14, 12, 10, 9, 7, 5, 3, 2, 0]],
        'matrix-v-ascent': [[0, 9, 17], [3, 12, 20], [6, 15, 24]],
        'matrix-v-decent': [[24, 15, 6], [20, 12, 3], [17, 9, 0]],
        'matrix-h-ascent': [0, 3, 6, 9, 12, 15, 17, 20, 24],
        'matrix-h-decent': [[24, 20, 17], [15, 12, 9], [6, 3, 0]],
        'matrix-random': [[24, 12, 17], [9, 20, 3], [15, 6, 0]],
    }

    for k in tqdm(indices_dict):
        concat_images_via_different_arrangement(k)


if __name__ == '__main__':
    sample_frames_from_video_trafficqa()
    preprocess_trafficqa_to_tensor_dataset()
    sample_frames_from_video_msrvtt()
    process_msrvtt_to_tensor_dataset()

    concat_images(9)
    concat_images(16)

    export_arrrangement_mstvtt()
