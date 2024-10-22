import cv2
import os
import os.path
import glob
import shutil
import numpy as np
import csv
from PIL import Image
import chainer

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from keras.models import Model, Sequential, load_model
from keras.layers import Input, LSTM, Dense, TimeDistributed, Lambda, Dropout
from keras import backend as K
from keras import regularizers

import time

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import argparse

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import sys
sys.path.append('../../Preparation/codes')
from data_utils import DataSet
sys.path.append('../../Preparation/codes')
from utils import Params
sys.path.append('../../Preparation/codes')
from utils import set_logger

def lstm(num_features=1024, hidden_units=256, dense_units=256, reg=1e-1, dropout_rate=0.5, seq_length=60, num_classes=4):
	model = Sequential()

	model.add(LSTM(1024, input_shape=(seq_length, num_features),return_sequences=False,dropout=dropout_rate))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(dropout_rate))
	model.add(Dense(num_classes, activation='softmax'))

	return model

def video_feat_extract(video):
	googlenet = chainer.links.GoogLeNet()

	video_capture = cv2.VideoCapture(video)
	n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
	basename_without_ext = os.path.splitext(os.path.basename(video))[0]
	video_feat = None

	for frame_idx in range(n_frames):
		success, frame = video_capture.read()
		if success:
			x = chainer.links.model.vision.googlenet.prepare(frame)
			x = x[np.newaxis]
			result = googlenet(x, layers=['pool5'])
			frame_feat = result['pool5'].data.squeeze()
			if video_feat is None:
				video_feat = frame_feat
			else:
				video_feat = np.vstack((video_feat, frame_feat))

	video_capture.release()

	return video_feat, n_frames, basename_without_ext


if __name__ == "__main__":
	json_path = os.path.join('../../Preparation/weights', 'params.json')
	assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
	params = Params(json_path)

	learning_rate = 0.0001#params.learning_rate#0.001
	decay = params.decay#0.0
	hidden_units = params.hidden_units#128
	dense_units = params.dense_units#128
	reg = params.reg#0.0
	dropout_rate = 0.5#params.dropout_rate#0.3
	batch_size = 64#params.batch_size#128
	nb_epoch = 100#params.nb_epoch#300
	train_size = params.train_size#0.8
	num_classes = 5#params.num_classes#6
	seq_length = 30#params.seq_length#16

	a = Input(shape=(1,))
	b = Dense(1)(a)
	model = Model(inputs=a, outputs=b)

	cnn_model = Model(inputs=a, outputs=b)

	rnn_model = lstm(hidden_units=hidden_units, dense_units=dense_units, 
		reg=reg, dropout_rate=dropout_rate,
		seq_length=seq_length, num_classes=num_classes)

	optimizer = Adam(lr=learning_rate, decay=decay)

	metrics = ['categorical_accuracy'] # ['accuracy']  # if using 'top_k_categorical_accuracy', must specify k
	rnn_model.compile(loss='categorical_crossentropy', optimizer=optimizer,
		metrics=metrics)

	rnn_model.summary()

	folder_path = '../../Preparation/weights/checkpoints/'

	saved_weights = os.path.join(folder_path, 'lstm_weights.0100-1.194.hdf5')
	rnn_model.load_weights(saved_weights)

	classes = ['ball', 'strike', 'swing', 'other', 'pickoff']
	np.set_printoptions(suppress=True)
	np.set_printoptions(precision=3)

	number = 2

	while 1:
		video_num = "video_" + str(number)
		#video_num = "video_2"
		dir_name = "../../" + video_num + "/clip_videos"
		npy_dir_name = dir_name + "_npy/"
		os.makedirs(npy_dir_name, exist_ok=True)
		plt_dir_name = dir_name + "_plt/"
		os.makedirs(plt_dir_name, exist_ok=True)
		video_paths = glob.glob(dir_name + "/*.mp4")
		seq_length = 30
		i = 0
		len_video_paths = len(video_paths)
		video_paths.sort()

		csv_path = "../../" + video_num + "/selected_frame.csv"
		with open(csv_path) as fp:
			csv_lst = list(csv.reader(fp))
		csv_num = 0

		for video in video_paths:
			video_feat, n_frames, basename_without_ext = video_feat_extract(video)
			print(video_feat.shape, n_frames, basename_without_ext)
			if n_frames < 30:
				move_dir_name = "../../" + video_num + "/pickoff_or_other_vid/"
				os.makedirs(move_dir_name, exist_ok=True)
				new_path = shutil.move(video, move_dir_name)
				csv_lst[csv_num].append(False)
				csv_num += 1
				print("skip!!!!!!!!!!!!!")
				print("***********************")
				continue

			batch_size = 30
			stride_size = 10
			start_idx = 0#
			end_idx = start_idx + batch_size#

			ball_list = np.zeros(video_feat.shape[0])
			strike_list = np.zeros(video_feat.shape[0])
			swing_list = np.zeros(video_feat.shape[0])
			other_list = np.zeros(video_feat.shape[0])
			pickoff_list = np.zeros(video_feat.shape[0])
			count_list = np.zeros(video_feat.shape[0])

			move_flag = 1

			while 1:
				X_test = None
				X_test = video_feat[start_idx:end_idx]
				X_test = np.expand_dims(X_test, axis=0)
				#print(start_idx,end_idx,X_test.shape)
				Y_pred_class = rnn_model.predict(X_test)
				#print(Y_pred_class)
				prelabel = int(np.argmax(Y_pred_class, axis=1))
				#print(classes[prelabel])
				if prelabel == 0 or prelabel == 1 or prelabel == 2:
					move_flag = 0

				for temp_num in range(start_idx, end_idx):
					ball_list[temp_num] = (ball_list[temp_num] * count_list[temp_num] + Y_pred_class[0][0]) / (count_list[temp_num] + 1)
					strike_list[temp_num] = (strike_list[temp_num] * count_list[temp_num] + Y_pred_class[0][1]) / (count_list[temp_num] + 1)
					swing_list[temp_num] = (swing_list[temp_num] * count_list[temp_num] + Y_pred_class[0][2]) / (count_list[temp_num] + 1)
					other_list[temp_num] = (other_list[temp_num] * count_list[temp_num] + Y_pred_class[0][3]) / (count_list[temp_num] + 1)
					pickoff_list[temp_num] = (pickoff_list[temp_num] * count_list[temp_num] + Y_pred_class[0][4]) / (count_list[temp_num] + 1)
					count_list[temp_num] += 1

				start_idx += stride_size
				end_idx = start_idx + batch_size

				if end_idx >= video_feat.shape[0]:
					end_idx = video_feat.shape[0]
					start_idx = end_idx - batch_size
					X_test = None
					X_test = video_feat[start_idx:end_idx]
					X_test = np.expand_dims(X_test, axis=0)
					#print(start_idx,end_idx,X_test.shape)
					Y_pred_class = rnn_model.predict(X_test)
					#print(Y_pred_class)
					prelabel = np.argmax(Y_pred_class, axis=1)
					#print(classes[int(prelabel)])

					for temp_num in range(start_idx, end_idx):
						ball_list[temp_num] = (ball_list[temp_num] * count_list[temp_num] + Y_pred_class[0][0]) / (count_list[temp_num] + 1)
						strike_list[temp_num] = (strike_list[temp_num] * count_list[temp_num] + Y_pred_class[0][1]) / (count_list[temp_num] + 1)
						swing_list[temp_num] = (swing_list[temp_num] * count_list[temp_num] + Y_pred_class[0][2]) / (count_list[temp_num] + 1)
						other_list[temp_num] = (other_list[temp_num] * count_list[temp_num] + Y_pred_class[0][3]) / (count_list[temp_num] + 1)
						pickoff_list[temp_num] = (pickoff_list[temp_num] * count_list[temp_num] + Y_pred_class[0][4]) / (count_list[temp_num] + 1)
						count_list[temp_num] += 1

					break
			"""
			np.set_printoptions(threshold=np.inf)
			for test_num in range(video_feat.shape[0]):
				print(test_num, ball_list[test_num], count_list[test_num])
			"""

			print("ave:  ball", np.average(ball_list), "strike", np.average(strike_list),
				"swing", np.average(swing_list), "other", np.average(other_list),
				"pickoff", np.average(pickoff_list))
			print("max:  ball", np.amax(ball_list), "strike", np.amax(strike_list),
				"swing", np.amax(swing_list), "other", np.amax(other_list),
				"pickoff", np.amax(pickoff_list))
			print("***********************")
			
			#"""
			if move_flag == 1:
				x = np.arange(ball_list.shape[0])
				fig, ax = plt.subplots(figsize=(8.0, 6.0))

				ax.set_xlim(0.0, ball_list.shape[0])
				ax.plot(x, ball_list)
				ax.plot(x, strike_list)
				ax.plot(x, swing_list)
				ax.plot(x, other_list)
				ax.plot(x, pickoff_list)
				ax.legend(['ball', 'strike', 'swing', 'other', 'pickoff'])#, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)#['back_bench', 'in_play', 'no_play', 'pitch', 'runner']
				fig_name = basename_without_ext + ".jpg"
				plt.savefig(plt_dir_name + fig_name)
				plt.close()

				move_dir_name = "../../" + video_num + "/pickoff_or_other_vid/"
				os.makedirs(move_dir_name, exist_ok=True)
				new_path = shutil.move(video, move_dir_name)
				csv_lst[csv_num].append(False)
				csv_num += 1
			else:
				path = os.path.join(npy_dir_name, basename_without_ext)
				np.save(path, video_feat)
				csv_lst[csv_num].append(True)
				csv_num += 1
			#"""
		
		with open(csv_path, 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerows(csv_lst)
		
		break
		number += 1
		if number >= 12:
			break