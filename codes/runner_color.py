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

number = 101
video_num = "video_" + str(number)
dir_name = "../../" + video_num + "/score"
pic_paths = glob.glob(dir_name + "/*.jpg")
len_pic_paths = len(pic_paths)
pic_paths.sort()

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

	basename_without_ext = os.path.splitext(os.path.basename(pic_paths[i]))[0]
	print(basename_without_ext)

	img = cv2.imread(pic_paths[i])
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	blur = cv2.blur(hsv, (2, 2))

	hsv_min = np.array([0, 64, 0])
	hsv_max = np.array([30, 255, 255])
	mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

	hsv_min = np.array([150, 64, 0])
	hsv_max = np.array([179, 255, 255])
	mask2 = cv2.inRange(hsv, hsv_min, hsv_max)

	mask = mask1 + mask2

	masked_img = cv2.bitwise_and(img, img, mask=mask)
	img_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

	ret, img_thresh = cv2.threshold(img_gray, 55, 255, cv2.THRESH_BINARY)
	#cv2.imwrite(dir_name + '/' + str(i) + '.jpg', img_thresh)

	pts_1st = np.array(((130, 37), (141, 29), (152, 37), (141, 45)))
	pic_1st = make_mask(img_thresh, pts_1st)
	runner_1st = 0 if np.all(pic_1st == 0) else 1

	pts_2nd = np.array(((98, 19), (109, 11), (120, 19), (109, 27)))
	pic_2nd = make_mask(img_thresh, pts_2nd)
	runner_2nd = 0 if np.all(pic_2nd == 0) else 1

	pts_3rd = np.array(((66, 37), (77, 29), (88, 37), (77, 45)))
	pic_3rd = make_mask(img_thresh, pts_3rd)
	runner_3rd = 0 if np.all(pic_3rd == 0) else 1

	#print(img_thresh.shape, pic_1st.shape, pic_2nd.shape, pic_3rd.shape)
	#im_v = cv2.vconcat([img_thresh.astype(np.float64), pic_1st.astype(np.float64), pic_2nd.astype(np.float64), pic_3rd.astype(np.float64)])
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
