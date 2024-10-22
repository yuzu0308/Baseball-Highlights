import cv2
import os
import glob
import shutil
import numpy as np
import csv

def select_replay(video, template_pic, score_dir_name):
	video_capture = cv2.VideoCapture(video)
	n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
	basename_without_ext = os.path.splitext(os.path.basename(video))[0]
	video_feat = None
	max_num = -100000

	for frame_idx in range(n_frames):
		success, frame = video_capture.read()
		if success:
			frame = frame[497 : 658, 1015 : 1227]#video_1 ~ video_12, 104
			#frame = frame[577 : 687, 1060 : 1219]#video_101
			#frame = frame[45 : 114, 70 : 260]#video_102,103
			comp_num = np.count_nonzero(template_pic == frame) / template_pic.size

			if comp_num > max_num:
				max_num = comp_num
				cv2.imwrite(score_dir_name + basename_without_ext + '.jpg', frame)

	video_capture.release()

	print(max_num, basename_without_ext)
	return max_num, basename_without_ext

#Main
number = 2
threshold = 0.09#video_1 ~ video_12, video_101, 104
#threshold = 0.04#video_102,103

while 1:
	video_num = "video_" + str(number)
	template_pic = cv2.imread("../../" + video_num + "/" + video_num + "_score.jpg")#video_1 ~ video_12, 104
	#template_pic = cv2.imread("video_" + str(number) + "_score.jpg")#video_101~103
	score_dir_name = "../../" + video_num + "/score/"
	os.makedirs(score_dir_name, exist_ok=True)
	dir_name = "../../" + video_num + "/clip_videos"
	video_paths = glob.glob(dir_name + "/*.mp4")
	len_video_paths = len(video_paths)
	video_paths.sort()

	csv_path = "../../" + video_num + "/selected_frame.csv"
	with open(csv_path) as fp:
		csv_lst = list(csv.reader(fp))
	csv_num = 0

	for path in video_paths:
		max_num, basename_without_ext = select_replay(path, template_pic, score_dir_name)

		if max_num < threshold:
			
			move_dir_name = "../../" + video_num + "/pickoff_or_other_vid/"
			os.makedirs(move_dir_name, exist_ok=True)
			shutil.move(path, move_dir_name)
			npy_path = "../../" + video_num + "/clip_videos_npy/" + basename_without_ext + ".npy"
			shutil.move(npy_path, move_dir_name)
			pic_path = score_dir_name + basename_without_ext + '.jpg'
			shutil.move(pic_path, move_dir_name)

			for lst_idx, lst_element in enumerate(csv_lst):
				if lst_element[0] == basename_without_ext:
					csv_num = lst_idx
			csv_lst[csv_num][4] = False
			
			print("move!", basename_without_ext)

		print("#####################")

	
	with open(csv_path, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerows(csv_lst)
	

	break
	number += 1
	if number >= 12:
		break