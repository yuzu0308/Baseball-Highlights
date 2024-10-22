# This script trains a neural network model from font-based number images.
# Image data are augmented by changing their shape.
# The trained model is evaluated by validation data of MNIST.
import gc, keras, math, os, re
import glob
import numpy as np
import cv2
import csv
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Reshape
from keras.layers.convolutional import Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout
from keras.datasets import mnist
from keras.utils import plot_model
from PIL import Image, ImageDraw, ImageFont # For drawing font images

def check_strike_ball(img_thresh, model, basename_without_ext):
    ret, img_thresh = cv2.threshold(img_thresh, 125, 255, cv2.THRESH_BINARY)#video_1 ~ video_12, 104
    #ret, img_thresh = cv2.threshold(img_thresh, 105, 255, cv2.THRESH_BINARY)#video_101,102
    #img_thresh = cv2.bitwise_not(img_thresh)#video_101

    ball_shot = img_thresh[134 : 154, 154 : 166]#video_1 ~ video_12, 104
    #ball_shot = img_thresh[8 : 36, 39 : 59]#video_101
    ball_shot = cv2.copyMakeBorder(ball_shot, 0, 0, 4, 4, cv2.BORDER_CONSTANT, (0,0,0))#video_1 ~ video_12,video_101, 104
    #ball_shot = img_thresh[27 : 47, 93 : 105]#video_102
    #ball_shot = cv2.copyMakeBorder(ball_shot, 0, 0, 4, 4, cv2.BORDER_CONSTANT, (0,0,0))#video_102
    if np.all(ball_shot == 0):
        print("ball_count: 0 (no_count)")
        ball = 0
    else:
        ball_shot = cv2.resize(ball_shot, dsize=(28, 28))
        ball_shot = (ball_shot.astype(np.float32))/255.0
        ball_shot = ball_shot.reshape((28, 28, 1))
        pred = np.squeeze(model.predict(ball_shot[np.newaxis]))
        pred = pred[:4]
        #print(pred)
        ball = np.argmax(pred)
        print("ball_count: ", ball)
    #cv2.imshow("img_th", ball_shot)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    strike_shot = img_thresh[134 : 154, 172 : 184]#video_1 ~ video_12, 104
    #strike_shot = img_thresh[41 : 69, 39 : 59]#video_101
    strike_shot = cv2.copyMakeBorder(strike_shot, 0, 0, 4, 4, cv2.BORDER_CONSTANT, (0,0,0))#video_1 ~ video_12,video_101, 104
    #strike_shot = img_thresh[27 : 47, 114: 126]#video_102
    #strike_shot = cv2.copyMakeBorder(strike_shot, 0, 0, 4, 4, cv2.BORDER_CONSTANT, (0,0,0))#video_102
    if np.all(strike_shot == 0):
        print("strike_count: 0 (no_count)")
        strike = 0
    else:
        strike_shot = cv2.resize(strike_shot, dsize=(28, 28))
        strike_shot = (strike_shot.astype(np.float32))/255.0
        strike_shot = strike_shot.reshape((28, 28, 1))
        pred = np.squeeze(model.predict(strike_shot[np.newaxis]))
        pred = pred[:3]
        #print(pred)
        strike = np.argmax(pred)
        print("strike_count: ", strike)
    #cv2.imshow("img_th", strike_shot)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    return ball, strike

def check_score(img_thresh, model, basename_without_ext):
    ret, img_thresh = cv2.threshold(img_thresh, 145, 255, cv2.THRESH_BINARY)#video_1 ~ video_12, 104
    #ret, img_thresh = cv2.threshold(img_thresh, 105, 255, cv2.THRESH_BINARY)#video_101,102
    
    top_score_shot = img_thresh[63 : 97, 89 : 123]#video_1 ~ video_12, 104
    #top_score_shot = img_thresh[49 : 77, 122 : 150]#video_101
    #top_score_shot = img_thresh[5 : 25, 55 : 75]#video_102
    top_score_shot = cv2.resize(top_score_shot, dsize=(28, 28))
    top_score_shot = (top_score_shot.astype(np.float32))/255.0
    top_score_shot = top_score_shot.reshape((28, 28, 1))
    pred = np.squeeze(model.predict(top_score_shot[np.newaxis]))
    #print(pred)
    top_score = np.argmax(pred)
    print("top_score: ", top_score)
    #cv2.imshow("img_th", top_score_shot)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    
    bottom_score_shot = img_thresh[96 : 130, 89 : 123]#video_1 ~ video_12, 104
    #bottom_score_shot = img_thresh[76 : 104, 122 : 150]#video_101
    #bottom_score_shot = img_thresh[26 : 46, 55 : 75]#video_102
    bottom_score_shot = cv2.resize(bottom_score_shot, dsize=(28, 28))
    bottom_score_shot = (bottom_score_shot.astype(np.float32))/255.0
    bottom_score_shot = bottom_score_shot.reshape((28, 28, 1))
    pred = np.squeeze(model.predict(bottom_score_shot[np.newaxis]))
    #print(pred)
    bottom_score = np.argmax(pred)
    print("bottom_score: ", bottom_score)
    #cv2.imshow("img_th", bottom_score_shot)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    return top_score, bottom_score

def check_out(img_thresh, model, basename_without_ext):
    ret, img_thresh = cv2.threshold(img_thresh, 125, 255, cv2.THRESH_BINARY)#video_1 ~ video_12, 104
    #ret, img_thresh = cv2.threshold(img_thresh, 105, 255, cv2.THRESH_BINARY)#video_101,102
    #img_thresh = cv2.bitwise_not(img_thresh)#video_101

    out_shot = img_thresh[132 : 156, 66 : 88]#video_1 ~ video_12, 104
    out_shot = cv2.copyMakeBorder(out_shot, 0, 0, 1, 1, cv2.BORDER_CONSTANT, (0,0,0))#video_1 ~ video_12, 104
    #out_shot = img_thresh[76 : 104, 39 : 59]#video_101
    #out_shot = cv2.copyMakeBorder(out_shot, 0, 0, 4, 4, cv2.BORDER_CONSTANT, (0,0,0))#video_101
    #out_shot = img_thresh[6 : 26, 88 : 104]#video_102
    #out_shot = cv2.copyMakeBorder(out_shot, 0, 0, 2, 2, cv2.BORDER_CONSTANT, (0,0,0))#video_102
    out_shot = cv2.resize(out_shot, dsize=(28, 28))
    out_shot = (out_shot.astype(np.float32))/255.0
    out_shot = out_shot.reshape((28, 28, 1))
    pred = np.squeeze(model.predict(out_shot[np.newaxis]))
    pred = pred[:4]
    #print(pred)
    out = np.argmax(pred)
    print("out: ", out)
    #cv2.imshow("img_th", out_shot)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    return out

def evaluate():
    np.set_printoptions(suppress=True)
    keras_param = "../../Preparation/weights/number_font_30epoch.h5"
    model = load_model(keras_param)
    model.summary()
    """
    plot_model(
        model,
        show_shapes=True,
    )
    """

    number = 2
    video_num = "video_" + str(number)
    dir_name = "../../" + video_num + "/score"
    pic_paths = glob.glob(dir_name + "/*.jpg")
    len_pic_paths = len(pic_paths)
    pic_paths.sort()

    csv_path = "../../" + video_num + "/selected_frame.csv"
    with open(csv_path) as fp:
        csv_lst = list(csv.reader(fp))
    csv_num = 0

    for filename in pic_paths:
        basename_without_ext = os.path.splitext(os.path.basename(filename))[0]
        print(basename_without_ext)
        pic = cv2.imread(filename, 0)
        pic = cv2.blur(pic, (2, 2))
        
        top_score, bottom_score = check_score(pic, model, basename_without_ext)
        out = check_out(pic, model, basename_without_ext)
        ball, strike = check_strike_ball(pic, model, basename_without_ext)

        for lst_idx, lst_element in enumerate(csv_lst):
            if lst_element[0] == basename_without_ext:
                csv_num = lst_idx
        csv_lst[csv_num].extend([top_score, bottom_score, out, ball, strike])
        print("############")
        #break

    csv_num_list = []
    for lst_idx, lst_element in enumerate(csv_lst):
        if lst_element[4] == "False":
            csv_num_list.append(lst_idx)
    for i in csv_num_list:
        csv_lst[i].extend(["None","None","None","None","None"])

    #"""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_lst)
    #"""

if __name__ == '__main__':
    evaluate()