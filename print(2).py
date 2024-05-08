import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Load YOLOv4-tiny model
net = cv2.dnn.readNet("/content/drive/MyDrive/photos_Anat/yolov4-tiny.weights", "/content/drive/MyDrive/photos_Anat/yolov4-tiny.cfg")

# Load COCO class labels
with open("/content/drive/MyDrive/photos_Anat/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Function to detect objects using YOLO
def detect_objects(image):
    # Define minimum confidence threshold and NMS threshold
    min_confidence = 0.5
    nms_threshold = 0.3

    # Preprocess input image for YOLO
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass through the network to perform object detection
    layer_names = net.getLayerNames()
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)

    # Initialize lists to store detected objects
    boxes = []

    # Iterate through each detection
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter detections based on confidence threshold
            if confidence > min_confidence and class_id == 0:  # Filter for person class
                # Scale bounding box coordinates to image size
                box = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")

                # Calculate top-left corner coordinates
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])

    return boxes

# Function to cluster object bounding boxes using DBSCAN
def cluster_objects(boxes, image_height):
    # Convert bounding boxes to NumPy array
    boxes = np.array(boxes)

    # Split the image into top and bottom halves
    top_half_boxes = boxes[boxes[:, 1] < image_height // 2]
    bottom_half_boxes = boxes[boxes[:, 1] >= image_height // 2]

    # Initialize DBSCAN with epsilon and min_samples parameters
    epsilon_top = 80  # Epsilon for top half
    epsilon_bottom = 100  # Epsilon for bottom half
    min_samples = 3  # Minimum number of points in a neighborhood

    # Initialize lists to store clustered object bounding boxes, group sizes, and radii
    clustered_boxes = []
    group_sizes = []
    radii = []

    # Cluster object bounding boxes in the top half using DBSCAN
    if len(top_half_boxes) > 0:
        dbscan_top = DBSCAN(eps=epsilon_top, min_samples=min_samples)
        labels_top = dbscan_top.fit_predict(top_half_boxes)
        unique_labels_top, counts_top = np.unique(labels_top, return_counts=True)
        for label, count in zip(unique_labels_top, counts_top):
            if label != -1:
                cluster_indices = np.where(labels_top == label)[0]
                cluster_boxes = top_half_boxes[cluster_indices, :]
                min_x = np.min(cluster_boxes[:, 0])
                max_x = np.max(cluster_boxes[:, 0] + cluster_boxes[:, 2])
                min_y = np.min(cluster_boxes[:, 1])
                max_y = np.max(cluster_boxes[:, 1] + cluster_boxes[:, 3])
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                radius = max((max_x - min_x) / 2, (max_y - min_y) / 2)
                clustered_boxes.append((center_x, center_y))
                group_sizes.append(count)
                radii.append(radius)

    # Cluster object bounding boxes in the bottom half using DBSCAN
    if len(bottom_half_boxes) > 0:
        dbscan_bottom = DBSCAN(eps=epsilon_bottom, min_samples=min_samples)
        labels_bottom = dbscan_bottom.fit_predict(bottom_half_boxes)
        unique_labels_bottom, counts_bottom = np.unique(labels_bottom, return_counts=True)
        for label, count in zip(unique_labels_bottom, counts_bottom):
            if label != -1:
                cluster_indices = np.where(labels_bottom == label)[0]
                cluster_boxes = bottom_half_boxes[cluster_indices, :]
                min_x = np.min(cluster_boxes[:, 0])
                max_x = np.max(cluster_boxes[:, 0] + cluster_boxes[:, 2])
                min_y = np.min(cluster_boxes[:, 1])
                max_y = np.max(cluster_boxes[:, 1] + cluster_boxes[:, 3])
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                radius = max((max_x - min_x) / 2, (max_y - min_y) / 2)
                clustered_boxes.append((center_x, center_y))
                group_sizes.append(count)
                radii.append(radius)

    return clustered_boxes, group_sizes, radii


# Example usage
# Load input image
image = cv2.imread("/content/drive/MyDrive/photos_Anat/park.jpg")

# Detect objects in the image
object_boxes = detect_objects(image)
image_height = image.shape[0]

# Cluster object bounding boxes using DBSCAN
clustered_boxes, group_sizes, radii = cluster_objects(object_boxes, image_height)

# Output clustered object bounding box centroids, group sizes, and radii
print("Clustered Object Bounding Box Centroids:")
print(clustered_boxes)
print("Group Sizes:")
print(group_sizes)
print("Radii:")
print(radii)

# Function to plot clustered object bounding box centroids and detected person bounding boxes on the input image
def plot_clusters(image, clustered_boxes, person_boxes, group_sizes, radii):
    # Plot clustered object bounding box centroids and circles around them on the input image
    for i, centroid in enumerate(clustered_boxes):
        cv2.circle(image, (int(centroid[0]), int(centroid[1])), int(radii[i]), (0, 165, 255), 2)  # Orange circle around each centroid
        # Display group size near the centroid
        cv2.putText(image, f"Group Size: {group_sizes[i]}", (int(centroid[0]) - 50, int(centroid[1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Plot detected person bounding boxes on the input image
    for box in person_boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Plot bounding boxes of all objects in blue
    for box in object_boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Convert image from BGR to RGB (for displaying with matplotlib)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image with clustered object centroids and person bounding boxes
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title('DBSCAN Clustering of Object Bounding Box Centroids')
    plt.axis('off')
    plt.show()

# Split object boxes into person and other objects
person_boxes = [box for box in object_boxes if box[2] == 0]  # Added line

# Plot clustered object bounding box centroids and detected person bounding boxes on the input image
plot_clusters(image, clustered_boxes, person_boxes, group_sizes, radii)  # Updated line
