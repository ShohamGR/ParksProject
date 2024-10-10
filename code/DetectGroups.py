import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from ultralytics import YOLO
import csv
import scipy.io
import imageio

# Load YOLOv8 model
model = YOLO('yolov8l.pt')

# Function to detect objects using YOLOv8
def detect_objects(image):
    results = model(image)
    detections = results[0].boxes.data.cpu().numpy()  # Get detected bounding boxes

    boxes = []
    for detection in detections:
        x1, y1, x2, y2, score, class_id = detection
        if class_id == 0:  # Filter for person class
            width = x2 - x1
            height = y2 - y1
            boxes.append([int(x1), int(y1), int(width), int(height)])

    return boxes

def cluster_objects(boxes, epsilon):
    boxes = np.array(boxes)
    if boxes.shape[0] == 0:
        return [], [], []

    min_samples = 1

    clustered_boxes = []
    group_sizes = []
    radii = []

    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels = dbscan.fit_predict(boxes)
    unique_labels, counts = np.unique(labels, return_counts=True)

    for label, count in zip(unique_labels, counts):
        if label != -1:
            cluster_indices = np.where(labels == label)[0]
            cluster_boxes = boxes[cluster_indices, :]
            min_x = np.min(cluster_boxes[:, 0])
            max_x = np.max(cluster_boxes[:, 0] + cluster_boxes[:, 2])
            min_y = np.min(cluster_boxes[:, 1])
            max_y = np.max(cluster_boxes[:, 1] + cluster_boxes[:, 3])
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            radius = np.sqrt(((max_x - min_x) / 2)**2 + ((max_y - min_y) / 2)**2)  # Euclidean distance
            clustered_boxes.append((center_x, center_y))
            group_sizes.append(count)
            radii.append(radius)

    return clustered_boxes, group_sizes, radii

import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def detectTime(image, park):
    
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_COLOR)
    elif isinstance(image, np.ndarray):
        # The image is already provided as an ndarray, no need to read
        pass
    else:
        raise TypeError("Unsupported image format")
    
    clock = sp.io.loadmat("./GroupProjectFiles/clock.mat")
    clock = clock["clock"]
    
    if park == 'kat':
        loc = clock['kat'][0][0][0]
    else:
        loc = clock['remez'][0][0][0]

    digit = np.zeros(clock["digits"][0][0][:, :, :6].shape)
    for i in range(6):
        cropped_image = cv2.resize(cv2.cvtColor(image[loc[i][0][1]-1:loc[i][0][1] + loc[i][0][3], loc[i][0][0]-1:loc[i][0][0] + loc[i][0][2]], cv2.COLOR_BGR2GRAY), clock['digits'][0][0][:, :, 0].T.shape)
        digit[:, :, i] = cropped_image
    

    digit_b = np.array(digit < 40).astype(np.uint8)
    digit_w = np.array(digit > 215).astype(np.uint8)

    ncc = np.zeros((2, 10))
    res = np.zeros(6)
    for i in range(6):
        for j in range(10):
     
            ncc[0, j] = cv2.matchTemplate(digit_b[:, :, i], clock['digits'][0][0][:, :, j], cv2.TM_CCOEFF_NORMED)[0][0]
            ncc[1, j] = cv2.matchTemplate(digit_w[:, :, i], clock['digits'][0][0][:, :, j], cv2.TM_CCOEFF_NORMED)[0][0]
            
        
        indb = np.argmax(ncc[0, :])
        indw = np.argmax(ncc[1, :])
        
        maxb = ncc[0, indb]
        maxw = ncc[1, indw]
        
        if maxb >= maxw:
            res[i] = indb
        else:
            res[i] = indw


    time = f"{int(res[0])}{int(res[1])}:{int(res[2])}{int(res[3])}:{int(res[4])}{int(res[5])}"

    return time

# Function to flatten 2D array to list of tuples
def flatten_2d_array_to_tuples(arr_2d):
    arr_2d = np.array(arr_2d)
    if len(arr_2d.shape) != 2:
        raise ValueError("Input must be a 2D array")
    flattened = [tuple(row) for row in arr_2d.tolist()]
    out = np.empty(len(flattened), dtype=object)
    out[:] = flattened
    return out

# Function to detect the segment number based on bounding box, parking lot, and camera information
def detect_mapping(bbox, park, cam):
    mapping = scipy.io.loadmat("./GroupProjectFiles/mapping.mat")["mapping"]
    bbox = np.round(bbox).astype(int)

    if park == 'kat':
        loc = mapping["kat"][0][0][0][cam-1]
    else:
        loc = mapping["remez"][0][0][0][cam-1]

    segMap = loc["segments"][0][0]
    colorTable = loc["table"][0][0]
    BottomPixNum = int(np.ceil(bbox[2] * bbox[3] * 0.1))

    if BottomPixNum > 10:
        bottomPix = [bbox[0], bbox[1] + int(np.floor(bbox[3] * 0.9)), bbox[2], int(np.ceil((bbox[3] + 1) * 0.1))]
    else:
        bottomPix = [bbox[0], bbox[1] + bbox[3] - 9, bbox[2], 10]

    bottomPix = np.uint16(bottomPix)
    segCrop = segMap[bottomPix[1]:bottomPix[1] + bottomPix[3], bottomPix[0]:bottomPix[0] + bottomPix[2]]
    segCrop = segCrop.reshape(-1, 3)
    colors = np.unique(segCrop, axis=0)
    temp_colors = flatten_2d_array_to_tuples(colors)
    temp_table = flatten_2d_array_to_tuples(colorTable[:, :3])
    nonvalid = np.setdiff1d(temp_colors, temp_table, assume_unique=True)
    validColors = np.setxor1d(temp_colors, nonvalid, assume_unique=True)
    validColors = np.array(validColors.tolist())
    count = np.zeros(validColors.shape[0])

    for i in range(validColors.shape[0]):
        count[i] = np.sum(np.all(segCrop == validColors[i], axis=1))

    ind = np.argmax(count)
    selection = validColors[ind]
    ind = np.argmax(np.all(colorTable[:, :3] == selection, axis=1))
    segN = colorTable[ind, 3]

    return segN

def save_video_frames(video_path, sample_interval =25):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    frames = []

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % sample_interval == 0:
            frames.append(frame)
            print(f"Frame {frame_number} saved")
        frame_number += 1

    cap.release()
    return frames, frame_number


def process_video(frames, frame_number, start_frame_num, output_file, park, cam, epsilon):
    #This is just to check that the whole video was loaded
    print(frame_number)
    #If the run stopped in the middle somewhere last time, set min_frames to the last frame that was written to the csv and this will continue smoothly
    min_frames = 0
    frame_idx = start_frame_num + min_frames

    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        #This makes sure we write the title row only if we're starting a new file :)
        if frame_idx == 0:
            writer.writerow(['Frame', 'Time', 'Group Center X', 'Group Center Y', 'Group Radius', 'Group Size', 'Size Category', 'Segment'])

        for frame in frames:
            print(f"Processing frame {frame_idx} out of {frame_number + start_frame_num}")
            object_boxes = detect_objects(frame)
            clustered_boxes, group_sizes, radii = cluster_objects(object_boxes, epsilon)
            time = detectTime(frame, park)

            # Write all detected groups to the file
            for i, group in enumerate(clustered_boxes):
                size_category = '1' if group_sizes[i] == 1 else '2' if group_sizes[i] == 2 else '3-6' if group_sizes[i] <= 6 else '7+'
                bbox = object_boxes[i]
                try:
                    segment = detect_mapping(bbox, park, cam)
                except ValueError as e:
                    print(f"Error detecting mapping for bbox {bbox}: {e}")
                    segment = -1  # Assign a default value or handle as needed

                writer.writerow([frame_idx, time, group[0], group[1], radii[i], group_sizes[i], size_category, segment])
                f.flush()
            frame_idx += 25


def extract_frame_times(video_path, park, fps):
    cap = cv2.VideoCapture(video_path)
    
    frame_times = {}
    frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_idx in range(0, total_frames, fps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        time = detectTime(frame, park)
        frame_times[frame_idx] = time
        frame_number += fps
    
    cap.release()
    return frame_times


output_file = "./DetectedGroups/Groups/Remez/2304/0812/cam3/raw_groups_remez_2304_3_new.csv"
fps = 25  # Replace with your actual frames per second value
park = 'remez'  # Replace with your actual park identifier
cam = 3
epsilon = 150

#frame_times = extract_frame_times(video_path, park, fps)
#update_csv_times(input_file, output_file, frame_times)

'''
start_frame_num = 0
for video_path in [ video_path_1, video_path_2, video_path_4, video_path_5, video_path_6, video_path_7, video_path_8, video_path_9]: 
    frames, num_frames = save_video_frames(video_path)   
    process_video(frames, num_frames, start_frame_num, output_file, park, cam, epsilon)
    start_frame_num += num_frames
'''
