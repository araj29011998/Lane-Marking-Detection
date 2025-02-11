import os
import cv2
from src.preprocess import preprocess_image, region_of_interest
from src.hough_transform import detect_lanes_hough_transform, draw_lines
from src.visualize import visualize
import json

# Function to load images from the dataset
def load_tusimple_data(data_dir, json_file):
    """
    Load TuSimple dataset images and labels from the provided directory and JSON file.
    Parse the JSON line by line to handle multiple JSON objects.
    """
    image_paths = []
    labels = []

    # Read JSON file for lane labels
    json_path = os.path.join(data_dir, json_file)
    
    with open(json_path, 'r') as file:
        for line in file:
            # Parse each line as a separate JSON object
            data = json.loads(line)
            
            # Fix the image path by ensuring 'clips' is only included once
            image_path = os.path.join(data_dir, data['raw_file'])
            image_paths.append(image_path)
            labels.append(data['lanes'])  # Assuming 'lanes' contains lane data

    print(f"Loaded {len(image_paths)} images and {len(labels)} labels from {data_dir}")
    return image_paths, labels

def main():
    # Paths to dataset
    train_set_path = "data/TuSimple/train_set/"
    test_set_path = "data/TuSimple/test_set/"
    
    # Load training images and labels
    image_paths, labels = load_tusimple_data(train_set_path, "label_data_0313.json")

    # Loop through each image and process it
    for idx, (image_path, label) in enumerate(zip(image_paths, labels)):
        print(f"Processing image {idx + 1}/{len(image_paths)}: {image_path}")

        # Read the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Preprocess the image (grayscale, blur, edge detection)
        edges = preprocess_image(image)

        # Apply region of interest masking
        roi_edges = region_of_interest(edges)

        # Use Hough Transform to detect lanes
        lines = detect_lanes_hough_transform(roi_edges)

        # Draw the detected lanes on the image
        line_image = draw_lines(image, lines)

        # Visualize the result and save it
        result_image_path = f"output/result_image_{idx + 1}.jpg"
        cv2.imwrite(result_image_path, line_image)
        print(f"Result saved at {result_image_path}")

        # Optionally show the result (comment out if not needed for every image)
        visualize(image, line_image, image)

if __name__ == "__main__":
    main()
