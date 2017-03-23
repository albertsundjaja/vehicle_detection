from sklearn.svm import SVC
from helper_functions import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import pickle
from scipy.ndimage.measurements import label
from keras.models import load_model

svc_model = pickle.load(open("svm_model.p", "rb"))
svc_scaler = pickle.load(open("svm_scaler.p", "rb"))
cnn_model = load_model("model.h5")
#load parameters
#load_params = pickle.load(open("svm_param.p", "rb"))

# Parameters
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9 # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "gray" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop_32 = [400, 450] # Min and max in y to search in slide_window()
y_start_stop_64 = [400, 600]
y_start_stop_80 = [400, None]
y_start_stop_128 = [400, None]
xy_overlap_32 = [0.75,0.75]
xy_overlap_64 = [0.75,0.75]
xy_overlap_80 = [0.5,0.5]
xy_overlap_128 = [0.5, 0.5]

# placeholder to save frames from video
frames = []
heatmap_glob = None

def process_image(image):
	# load image
	#image = mpimg.imread('test6.jpg')
	# crop the sky
	# image_cropped = image[400:960, 0:1280]

	# image copied to be drawn into
	draw_image = np.copy(image)

	# create the sliding windows for individual image
	window_list_32 = slide_window(image, y_start_stop=y_start_stop_32, xy_window = (32,32), xy_overlap=xy_overlap_32)
	window_list_64 = slide_window(image, y_start_stop=y_start_stop_64, xy_window = (64,64), xy_overlap=xy_overlap_64)
	window_list_80 = slide_window(image, y_start_stop=y_start_stop_64, xy_window = (80,80), xy_overlap=xy_overlap_80)
	window_list_128 = slide_window(image, y_start_stop=y_start_stop_128, xy_window = (128,128), xy_overlap=xy_overlap_128)

	# placeholder for detected window
	window_detected_list = []

	# iterate through the windows and detect vehicle
	for window in window_list_32:
		window_img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64)) 
		window_features = single_extract_features(window_img, color_space=color_space, 
			spatial_size=spatial_size, hist_bins=hist_bins, 
			orient=orient, pix_per_cell=pix_per_cell, 
			cell_per_block=cell_per_block, 
			hog_channel=hog_channel, spatial_feat=spatial_feat, 
			hist_feat=hist_feat, hog_feat=hog_feat)
		# Reshape and apply scaling
		reshaped = window_features.reshape(1, -1)
		window_features_scaled = svc_scaler.transform(reshaped)
		# Predict using your classifier
		prediction = svc_model.predict(window_features_scaled)
		if prediction == 1:
			window_detected_list.append(window)

	# iterate through the windows and detect vehicle
	for window in window_list_64:
		window_img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64)) 
		window_features = single_extract_features(window_img, color_space=color_space, 
			spatial_size=spatial_size, hist_bins=hist_bins, 
			orient=orient, pix_per_cell=pix_per_cell, 
			cell_per_block=cell_per_block, 
			hog_channel=hog_channel, spatial_feat=spatial_feat, 
			hist_feat=hist_feat, hog_feat=hog_feat)
		# Reshape and apply scaling
		reshaped = window_features.reshape(1, -1)
		window_features_scaled = svc_scaler.transform(reshaped)
		# Predict using your classifier
		prediction = svc_model.predict(window_features_scaled)
		if prediction == 1:
			window_detected_list.append(window)

		# iterate through the windows and detect vehicle
	for window in window_list_80:
		window_img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64)) 
		window_features = single_extract_features(window_img, color_space=color_space, 
			spatial_size=spatial_size, hist_bins=hist_bins, 
			orient=orient, pix_per_cell=pix_per_cell, 
			cell_per_block=cell_per_block, 
			hog_channel=hog_channel, spatial_feat=spatial_feat, 
			hist_feat=hist_feat, hog_feat=hog_feat)
		# Reshape and apply scaling
		reshaped = window_features.reshape(1, -1)
		window_features_scaled = svc_scaler.transform(reshaped)
		# Predict using your classifier
		prediction = svc_model.predict(window_features_scaled)
		if prediction == 1:
			window_detected_list.append(window)

	# iterate through the windows and detect vehicle
	for window in window_list_128:
		window_img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64)) 
		window_features = single_extract_features(window_img, color_space=color_space, 
			spatial_size=spatial_size, hist_bins=hist_bins, 
			orient=orient, pix_per_cell=pix_per_cell, 
			cell_per_block=cell_per_block, 
			hog_channel=hog_channel, spatial_feat=spatial_feat, 
			hist_feat=hist_feat, hog_feat=hog_feat)
		# Reshape and apply scaling
		reshaped = window_features.reshape(1, -1)
		window_features_scaled = svc_scaler.transform(reshaped)
		# Predict using your classifier
		prediction = svc_model.predict(window_features_scaled)
		if prediction == 1:
			window_detected_list.append(window)

	# Create a copy placeholder for heatmap
	heat = np.zeros_like(image[:,:,0]).astype(np.float)

	# Add heat to each box in window list
	heat = add_heat(heat, window_detected_list)
	    
	# Apply threshold to help remove false positives
	heat = apply_threshold(heat, 4)

	# Visualize the heatmap when displaying    
	heatmap = np.clip(heat, 0, 255)

	# Check if this is first init, initialise global heatmap
	global heatmap_glob
	if (heatmap_glob == None):
		print('yes')
		heatmap_glob = heatmap

	new_frame_factor = 0.3
	heatmap = new_frame_factor * heatmap + (1 - new_frame_factor) * heatmap_glob
	heatmap = apply_threshold(heatmap, 4)

	#update heatmap glob
	heatmap_glob = heatmap

	# Find final boxes from heatmap using label function
	labels = label(heatmap)

	# Get bounding box of the heatmap labels to get the image to feed into our cnn
	bboxes = get_bboxes_heatmap(labels)
	# Placeholder for CNN classification
	valid_bboxes = []

	# Feed each bbox image into CNN
	for bbox in bboxes:
		potential_bbox = cv2.resize(image[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]], (64, 64)) 
		prediction = cnn_model.predict(potential_bbox[None,:,:,:])
		print(prediction)
		if prediction > 0.5:
			valid_bboxes.append(bbox)

	# Draw box for validated bbox by CNN
	draw_img = draw_bboxes(np.copy(image), valid_bboxes)

	# draw boxes for detected window
	img_drawn = draw_boxes(draw_image, window_detected_list)

	draw_img = draw_labeled_bboxes(np.copy(image), labels)

	return draw_img



