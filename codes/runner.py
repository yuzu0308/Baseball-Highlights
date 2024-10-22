import cv2
import os
import glob
import shutil
import numpy as np
import csv

def make_mask(origin_pic, mask_npy):
	#cv2.polylines(base_shot, [pts_2nd], True, (255, 0, 0), thickness=1)
	mask = np.zeros_like(origin_pic, dtype=np.float64)
	mask_pic = cv2.fillConvexPoly(mask, points=mask_npy, color=(255, 255, 255))
	result_pic = origin_pic * mask_pic

	return result_pic

number = 2
video_num = "video_" + str(number)
dir_name = "../../" + video_num + "/score"
pic_paths = glob.glob(dir_name + "/*.jpg")
len_pic_paths = len(pic_paths)
pic_paths.sort()

threshold = 125#video1 ~ 12, 101, 104
#threshold = 155#video102 ~ 103
i = 0

np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.inf)

csv_path = "../../" + video_num + "/selected_frame.csv"
with open(csv_path) as fp:
	csv_lst = list(csv.reader(fp))
csv_num = 0

while 1:
	if i == len_pic_paths:
		break

	template_pic = cv2.imread(pic_paths[i], 0)
	template_pic = cv2.blur(template_pic, (2, 2))
	#template_pic = cv2.imread("video_12/score/video_12_000019.jpg", 0)
	basename_without_ext = os.path.splitext(os.path.basename(pic_paths[i]))[0]
	print(basename_without_ext)
	ret, img_thresh = cv2.threshold(template_pic, threshold, 255, cv2.THRESH_BINARY)
	#img_thresh = img_thresh.astype(np.float64)#video_102
	#img_thresh = cv2.bitwise_not(img_thresh)
	#cv2.imwrite(dir_name + '/' + str(i) + '.jpg', img_thresh)
	
	base_shot = img_thresh[63 : 130, 127 : 208].astype(np.float64)#video1 ~ 12, 104

	pts_1st = np.array(((43, 42), (59, 26), (75, 42), (59, 58)))#video1 ~ 12, 104
	pic_1st = make_mask(base_shot, pts_1st)#video1 ~ 12,104
	#pts_1st = np.array(((168, 19), (175, 13), (182, 19), (175, 25)))#video102
	#pic_1st = make_mask(img_thresh, pts_1st)#video102
	runner_1st = 0 if np.all(pic_1st == 0) else 1

	pts_2nd = np.array(((24, 23), (40, 7), (56, 23), (40, 39)))#video1 ~ 12, 104
	pic_2nd = make_mask(base_shot, pts_2nd)#video1 ~ 12, 104
	#pts_2nd = np.array(((156, 11), (163, 5), (170, 11), (163, 17)))#video102
	#pic_2nd = make_mask(img_thresh, pts_2nd)#video102
	runner_2nd = 0 if np.all(pic_2nd == 0) else 1

	pts_3rd = np.array(((5, 42), (21, 26), (37, 42), (21, 58)))#video1 ~ 12, 104
	pic_3rd = make_mask(base_shot, pts_3rd)#video1 ~ 12, 104
	#pts_3rd = np.array(((144, 19), (151, 13), (158, 19), (151, 25)))#video102
	#pic_3rd = make_mask(img_thresh, pts_3rd)#video102
	runner_3rd = 0 if np.all(pic_3rd == 0) else 1

	#im_v = cv2.vconcat([base_shot, pic_1st, pic_2nd, pic_3rd])
	print("runner_1st:", runner_1st, " runner_2nd:", runner_2nd,
		" runner_3rd:", runner_3rd)

	for lst_idx, lst_element in enumerate(csv_lst):
		if lst_element[0] == basename_without_ext:
			csv_num = lst_idx
	csv_lst[csv_num].extend([runner_1st, runner_2nd, runner_3rd])
	#cv2.imshow("img_th", im_v)
	#cv2.waitKey()
	#cv2.destroyAllWindows()#esc

	print("#####################")
	i += 1
	#break

csv_num_list = []
for lst_idx, lst_element in enumerate(csv_lst):
	if lst_element[4] == "False":
		csv_num_list.append(lst_idx)

for i in csv_num_list:
	csv_lst[i].extend(["None","None","None"])

with open(csv_path, 'w', newline='') as f:
	writer = csv.writer(f)
	writer.writerows(csv_lst)
