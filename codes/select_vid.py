import numpy as np
import cv2
import os
import glob
import shutil
import csv
from knapsack import knapsack_dp
import math
import h5py
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.inf)

def evaluate_summary(machine_summary, user_summary, eval_metric='avg'):
    """Compare machine summary with user summary (keyshot-based).
    Args:
    --------------------------------
    machine_summary and user_summary should be binary vectors of ndarray type.
    eval_metric = {'avg', 'max'}
    'avg' averages results of comparing multiple human summaries.
    'max' takes the maximum (best) out of multiple comparisons.
    """
    machine_summary = machine_summary.astype(np.float32)
    user_summary = user_summary.astype(np.float32)
    n_users,n_frames = user_summary.shape

    # binarization
    machine_summary[machine_summary > 0] = 1
    user_summary[user_summary > 0] = 1

    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        zero_padding = np.zeros((n_frames - len(machine_summary)))
        machine_summary = np.concatenate([machine_summary, zero_padding])

    f_scores = []
    prec_arr = []
    rec_arr = []

    for user_idx in range(n_users):
        gt_summary = user_summary[user_idx,:]
        overlap_duration = (machine_summary * gt_summary).sum()
        precision = overlap_duration / (machine_summary.sum() + 1e-8)
        recall = overlap_duration / (gt_summary.sum() + 1e-8)
        if precision == 0 and recall == 0:
            f_score = 0.
        else:
            f_score = (2 * precision * recall) / (precision + recall)
        f_scores.append(f_score)
        prec_arr.append(precision)
        rec_arr.append(recall)

    if eval_metric == 'avg':
        final_f_score = np.mean(f_scores)
        final_prec = np.mean(prec_arr)
        final_rec = np.mean(rec_arr)
    elif eval_metric == 'max':
        final_f_score = np.max(f_scores)
        max_idx = np.argmax(f_scores)
        final_prec = prec_arr[max_idx]
        final_rec = rec_arr[max_idx]
    
    return final_f_score, final_prec, final_rec

def select_batter_result(np_csv_lst):
	selected_videos = []

	lst_idx = 0
	while 1:
		if lst_idx+1 == np_csv_lst.shape[0]:
			selected_videos.append(lst_idx)
			break

		#add score
		if(np_csv_lst[lst_idx][5] != np_csv_lst[lst_idx+1][5] or
			np_csv_lst[lst_idx][6] != np_csv_lst[lst_idx+1][6]):
			selected_videos.append(lst_idx)
			lst_idx += 1
			continue

		#add out
		if np_csv_lst[lst_idx][7] != np_csv_lst[lst_idx+1][7]:
			selected_videos.append(lst_idx)
			lst_idx += 1
			continue

		#move runner
		if(np_csv_lst[lst_idx][10] != np_csv_lst[lst_idx+1][10] or
			np_csv_lst[lst_idx][11] != np_csv_lst[lst_idx+1][11] or
			np_csv_lst[lst_idx][12] != np_csv_lst[lst_idx+1][12]):
			selected_videos.append(lst_idx)
			lst_idx += 1
			continue

		lst_idx += 1

	return selected_videos

def scoring(selected_videos, np_csv_lst):
	seg_score = []
	nfps = []
	for i in selected_videos:
		print(np_csv_lst[i])
		video_score = 0.0
		substract_score = 0
		runnner_bias = (0.1*int(np_csv_lst[i][11])) + (0.1*int(np_csv_lst[i][12])*2)
		#last batter
		if i+1 == np_csv_lst.shape[0]:
			# pitch => audience, in_play
			if np_csv_lst[i][3] == "True":
				video_score = 0.1 + runnner_bias
				print("grounder / fly: ", video_score)
			# only pitch
			elif np_csv_lst[i][3] == "False":
				video_score = 0.55 + runnner_bias
				print("strikeout: ", video_score)
			seg_score.append(round(float(video_score), 2))
			nfps.append(int(int(np_csv_lst[i][2]) - int(np_csv_lst[i][1]) + 1))
			break

		# pitch => audience, in_play
		if np_csv_lst[i][3] == "True":
			# add score
			if(np_csv_lst[i][5] != np_csv_lst[i+1][5] or
				np_csv_lst[i][6] != np_csv_lst[i+1][6]):
				substract_score = int(np_csv_lst[i+1][5]) - int(np_csv_lst[i][5])
				if substract_score == 0:
					substract_score = int(np_csv_lst[i+1][6]) - int(np_csv_lst[i][6])
				#add out
				if np_csv_lst[i][7] != np_csv_lst[i+1][7]:
					video_score = 0.48 + (0.02*substract_score)
					print("out & add score: ", video_score)
				else:
					# Presence of runners
					if(int(np_csv_lst[i+1][10]) == 0 or int(np_csv_lst[i+1][11]) == 0
						or int(np_csv_lst[i+1][12]) == 0):
						video_score = 0.92 + (0.02*substract_score)
						print("homerun: ", video_score)
					else:
						video_score = 0.86 + (0.02*substract_score)
						print("clutch hit: ", video_score)
			else:
				#add out
				if np_csv_lst[i][7] != np_csv_lst[i+1][7]:
					video_score = 0.1 + runnner_bias
					print("grounder / fly: ", video_score)
				else:
					video_score = 0.75
					print("hit / miss: ", video_score)
		# only pitch
		elif np_csv_lst[i][3] == "False":
			# add score
			if(np_csv_lst[i][5] != np_csv_lst[i+1][5] or
				np_csv_lst[i][6] != np_csv_lst[i+1][6]):
				substract_score = int(np_csv_lst[i+1][5]) - int(np_csv_lst[i][5])
				if substract_score == 0:
					substract_score = int(np_csv_lst[i+1][6]) - int(np_csv_lst[i][6])
				video_score = 0.48 + (0.02*substract_score)
				print("bases-full walk / passed ball: ", video_score)
			else:
				#add out
				if np_csv_lst[i][7] != np_csv_lst[i+1][7]:
					video_score = 0.55 + runnner_bias
					print("strikeout: ", video_score)
				else:
					video_score = 0.2
					print("walk / hit by pitch: ", video_score)
		seg_score.append(round(float(video_score), 2))
		nfps.append(int(int(np_csv_lst[i][2]) - int(np_csv_lst[i][1]) + 1))
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

	return seg_score, nfps

def frm2video(video_capture, n_frames, summary, vid_writer):
	for frame_idx in range(n_frames):
		success, frame = video_capture.read()
		if success:
			if summary[frame_idx] == 1:
				vid_writer.write(frame)

def print_graph(pic_name, machine_summary):
	print(machine_summary.shape)
	
	probs = np.zeros_like(machine_summary).astype(float) + 0.5
	left = np.array(range(probs.shape[0]))
	plt.figure(figsize=(16.8,4.8))
	for i in range(len(left)):
		plt.bar(left[i], probs[i],
			width=1.0, color= 'lightgray' if machine_summary[i] == 0 else 'blue')
	plt.savefig(pic_name)
	plt.close()
	"""
    summary_15 = None
    for machine_summary_idx, machine_summary_num in enumerate(machine_summary):
        if machine_summary_idx % 5 == 0:
            if summary_15 is None:
                summary_15 = machine_summary_num
            else:
                summary_15 = np.vstack((summary_15, machine_summary_num))
    summary_15 = np.squeeze(summary_15)

    probs = np.zeros_like(summary_15).astype(float) + 0.5
    left = np.array(range(probs.shape[0]))
    
    for i in range(len(left)):
        plt.bar(left[i], probs[i],
        	width=1.0, color= 'lightgray' if summary_15[i] == 0 else 'blue')
    plt.savefig(pic_name)
    plt.close()
    """

if __name__ == '__main__':
	number = 2
	video_num = "video_" + str(number)
	output_dir = "../../" + video_num + "/result"
	os.makedirs(output_dir, exist_ok=True)

	csv_path = "../../" + video_num + "/selected_frame.csv"
	with open(csv_path) as fp:
		csv_lst = list(csv.reader(fp))
	#csv_num = 0

	np_csv_lst = np.array(csv_lst)
	np_csv_lst = np_csv_lst[np_csv_lst[:, 4] != 'False']
	print(np_csv_lst)

	selected_videos = select_batter_result(np_csv_lst)
	print("###########################################")

	seg_score, nfps = scoring(selected_videos, np_csv_lst)
	print("###########################################")

	print("seg_score:", seg_score)
	print("nfps:", nfps)
	n_segs = len(nfps)
	print("n_segs:", n_segs)
	
	video_name = "../../baseball_video/" + video_num + ".mp4"
	video_capture = cv2.VideoCapture(video_name)
	n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
	limits = int(math.floor(n_frames * 0.03))
	print("limits:", limits)
	picks = knapsack_dp(seg_score, nfps, n_segs, limits)
	#print(picks)

	predict_summary = np.zeros(n_frames)
	for i in picks:
		print(np_csv_lst[selected_videos[i]])
		start = int(np_csv_lst[selected_videos[i]][1])
		end = int(np_csv_lst[selected_videos[i]][2]) + 1
		predict_summary[start:end] = 1

	pic_name = output_dir + "/LONG_predict_summary.png"
	print_graph(pic_name, predict_summary)
	
	#make predict video
	result_video_name = output_dir + "/predict.mp4"
	vid_writer = cv2.VideoWriter(
		result_video_name,
		cv2.VideoWriter_fourcc(*'MP4V'),
		30,
		(1280, 720),
	)
	frm2video(video_capture, n_frames, predict_summary, vid_writer)
	vid_writer.release()
	video_capture.release()
	
	#"""
	print("###########################################")
	dataset = h5py.File('../../Preparation/datasets/baseball_chainer_35_3.h5', 'r')
	dataset_keys = dataset.keys()
	for key_idx, key in enumerate(dataset_keys):
		name = str(dataset[key]['video_name'][...])
		if video_num == name[2:-1]:
			user_summary = dataset[key]['user_summary2'][...]
	dataset.close()

	eval_metric = 'avg'#'max'
	fms = []
	for i in range(user_summary.shape[0]):
		pic_name = output_dir + "/LONG_user_summary_" + str(i) + ".png"
		print_graph(pic_name, user_summary[i])
		fm, _, _ = evaluate_summary(predict_summary,
			np.expand_dims(user_summary[i], axis = 0), eval_metric)
		print(i, " F-score ", "{:.1%}".format(fm))
		fms.append(fm)
		
		#make user video
		video_name = "../../baseball_video/" + video_num + ".mp4"
		video_capture = cv2.VideoCapture(video_name)
		n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
		result_video_name = output_dir + "/user_summary_" + str(i) + ".mp4"
		vid_writer = cv2.VideoWriter(
			result_video_name,
			cv2.VideoWriter_fourcc(*'MP4V'),
			30,
			(1280, 720),
		)
		frm2video(video_capture, n_frames, user_summary[i], vid_writer)
		vid_writer.release()
		video_capture.release()
		
	mean_fm = np.mean(fms)
	print("Average F-score {:.1%}".format(mean_fm))
	#"""