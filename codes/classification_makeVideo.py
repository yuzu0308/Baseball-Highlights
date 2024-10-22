from PIL import Image
import keras
import sys, os
import numpy as np
from keras.models import load_model

import cv2
import os
import glob
import shutil

import tkinter

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import moviepy.editor as mp

import csv

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def load_image(img):
    #img = Image.open(path)
    img = img.convert('RGB')
    imsize = (224, 224)
    img = img.resize(imsize)
    img = np.asarray(img)
    return img

def write_plt(audience_list, in_play_list, pitch_list, result_dir_name):
    print("write_plt ...")
    x = np.arange(audience_list.shape[0])

    fig, ax = plt.subplots(figsize=(15.0, 9.6))

    ax.set_xlim(0.0, audience_list.shape[0])
    ax.plot(x, audience_list)
    ax.plot(x, in_play_list)
    ax.plot(x, pitch_list)
    ax.legend(['audience', 'in_play', 'pitch'],
        bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.savefig(result_dir_name + "/image_classification.jpg")

def classification(keras_param, path, result_dir_name): #動画(画像)を４つのクラスに分類
    model = load_model(keras_param)

    print("##################")
    print(path)
    print("classification ...")

    video_capture = cv2.VideoCapture(path)
    n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    audience_list = np.array([])
    in_play_list = np.array([])
    pitch_list = np.array([])
    result_pic_num = np.array([])

    in_play_flag = 0
    pitch_flag = 0
    pitch_start_num = 0
    pitch_count = 0
    in_play_count = 0

    in_play_TR_or_FA_list = []

    for frame_idx in range(n_frames):
        success, frame = video_capture.read() #success=フレームが正常に読み込まれたか
        # frame ＝動画の１フレーム（画像データ：numpy)
        if success:
            print(frame_idx, "/", n_frames)
            file = cv2pil(frame)
            img = load_image(file)
            prd = model.predict(np.expand_dims(img, axis=0) / 255.0) #各クラスである確率
            #print(prd)
            prelabel = np.argmax(prd, axis=1)
            #print(prelabel)

            audience_list = np.append(audience_list, prd[0][0])
            in_play_list = np.append(in_play_list, prd[0][1])
            pitch_list = np.append(pitch_list, prd[0][3])

            #pitch
            if int(prelabel) == 3:
                if in_play_flag == 1:
                    if pitch_count > 50:
                        result_pic_num = np.append(result_pic_num, np.arange(pitch_start_num, frame_idx-1))
                        if in_play_count > 41:
                            in_play_TR_or_FA_list.append(True)
                        else:
                            in_play_TR_or_FA_list.append(False)
                        #print("pitch_count:", pitch_count, "in_play_count:", in_play_count)
                    in_play_flag = 0
                    pitch_flag = 0
                    pitch_start_num = 0
                    pitch_count = 0
                    in_play_count = 0
                if pitch_flag == 0:
                    pitch_flag = 1
                    pitch_start_num = frame_idx
                elif pitch_flag == 1:
                    pitch_count += 1
            #in_play
            elif int(prelabel) == 1:
                if pitch_flag == 1:
                    in_play_flag = 1
                    in_play_count += 1
            #audience
            elif int(prelabel) == 0:
                if pitch_flag == 1:
                    in_play_flag = 1
                    in_play_count += 1
            #no_play
            elif int(prelabel) == 2:
                if pitch_flag == 1:
                    if pitch_count > 50:
                        result_pic_num = np.append(result_pic_num, np.arange(pitch_start_num, frame_idx-1))
                        #print("pitch_count:", pitch_count, "in_play_count:", in_play_count)
                        if in_play_flag == 1:
                            if in_play_count > 41:
                                in_play_TR_or_FA_list.append(True)
                            else:
                                in_play_TR_or_FA_list.append(False)
                        elif in_play_flag == 0:
                            in_play_TR_or_FA_list.append(False)
                in_play_flag = 0
                pitch_flag = 0
                pitch_start_num = 0
                pitch_count = 0
                in_play_count = 0

    video_capture.release()
    write_plt(audience_list, in_play_list, pitch_list, result_dir_name)

    return result_pic_num, in_play_TR_or_FA_list

def make_video(path, video_num, result_dir_name, result_pic_num, in_play_TR_or_FA_list):
    print("make_video ...")
    video_capture = cv2.VideoCapture(path)  #元動画読み込み
    n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) #動画のフレーム数

    vid_num = 1
    vid_filename = video_num + "_{}.mp4".format(str(vid_num).zfill(6))
    basename_without_ext = os.path.splitext(os.path.basename(vid_filename))[0]
    dir_name = result_dir_name + '/clip_videos'
    os.makedirs(dir_name, exist_ok=True)
    vid_writer = cv2.VideoWriter(
        '{}/{}'.format(dir_name, vid_filename),
        cv2.VideoWriter_fourcc(*'MP4V'),
        30,
        (1280, 720),
        )

    pitch_flag = 0
    selected_pic_num = []
    start_num = 0
    j = 0

    for frame_idx in range(n_frames):
        success, frame = video_capture.read()
        if success:
            if frame_idx in result_pic_num:
                vid_writer.write(frame)
                if pitch_flag == 0:
                    pitch_flag = 1
                    start_num = frame_idx
            else:
                if pitch_flag == 1:
                    vid_writer.release()
                    selected_pic_num.append([basename_without_ext, start_num, frame_idx-1, in_play_TR_or_FA_list[j]])
                    pitch_flag = 0
                    vid_num += 1
                    j += 1
                    vid_filename = video_num + "_{}.mp4".format(str(vid_num).zfill(6))
                    basename_without_ext = os.path.splitext(os.path.basename(vid_filename))[0]
                    vid_writer = cv2.VideoWriter(
                        '{}/{}'.format(dir_name, vid_filename),
                        cv2.VideoWriter_fourcc(*'MP4V'),
                        30,
                        (1280, 720),
                        )

    if pitch_flag == 1:
        vid_writer.release()
        selected_pic_num.append([basename_without_ext, start_num, n_frames-1, in_play_TR_or_FA_list[j]])
        j += 1
    else:
        video_paths = glob.glob(dir_name+"/*.mp4")
        len_video_paths = len(video_paths)
        video_paths.sort()
        vid_name = video_paths[len_video_paths-1]
        os.remove(vid_name)

    video_capture.release()

    if j != len(in_play_TR_or_FA_list):
        print("################### length_error!! ###################")

    save_path = os.path.join(result_dir_name, "selected_frame.csv")
    f = open(save_path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerows(selected_pic_num)
    f.close()

def extract_wav(path, result_dir_name):
    print("extract_wav ...")
    clip_input = mp.VideoFileClip(path).subclip()
    basename_without_ext = os.path.splitext(os.path.basename(path))[0]
    clip_input.audio.write_audiofile(result_dir_name + '/' +
        basename_without_ext + '.wav')

if __name__ == '__main__':
    number = 3

    while 1:
        video_num = "video_" + str(number)
        result_dir_name = "../../" + video_num
        os.makedirs(result_dir_name, exist_ok=True)
        path = '../../baseball_video/' + video_num + '.mp4'
        keras_param = "../../Preparation/weights/vgg16_baseball_4class_200epoch_1e5.h5"

        result_pic_num, in_play_TR_or_FA_list = classification(keras_param, path, result_dir_name)

        make_video(path, video_num, result_dir_name, result_pic_num, in_play_TR_or_FA_list)

        #extract_wav(path, result_dir_name)
        
        print("##################")
        break
        number += 1
        if number >= 12:
            break