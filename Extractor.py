import cv2
import numpy as np
import random
import os

def extract(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    DIST_THRESHOLD = 89 
    def contours_are_close(cnt1, cnt2, threshold):
        cnt1 = cnt1.reshape(-1, 2)  
        for p in cnt2.reshape(-1, 2):  
            if cv2.pointPolygonTest(cnt1, (int(p[0]), int(p[1])), True) > -threshold:
                return True
        return False
    merged_contours = [] 
    used = set()
    for i, c1 in enumerate(contours):
        if i in used:
            continue
        merged = [c1]
        for j, c2 in enumerate(contours):
            if i != j and j not in used and (contours_are_close(c1,c2,DIST_THRESHOLD) or contours_are_close(c2,c1,DIST_THRESHOLD)):
                merged.append(c2)
                used.add(j)
        merged_points = np.vstack(merged)
        hull = cv2.convexHull(merged_points)
        merged_contours.append(hull)






    output_image = image.copy()
    for contour in merged_contours:
        color = [random.randint(0, 255) for _ in range(3)]  
        cv2.drawContours(output_image, [contour], -1, color, 2)
    cv2.imshow("Merged Contours", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    output_image = image.copy()


    for contour in merged_contours:
        x_coords = contour[:, 0, 0]
        y_coords = contour[:, 0, 1]  

        min_x = np.min(x_coords)
        max_x = np.max(x_coords)
        min_y = np.min(y_coords)
        max_y = np.max(y_coords)


        points = np.array([[max_x, max_y],
                      [max_x, min_y],
                      [min_x, min_y],
                      [min_x, max_y]])

        contourBox = [points.reshape((4, 1, 2))]

        color = [random.randint(0, 255) for _ in range(3)]
        cv2.drawContours(output_image, contourBox, -1, color, 2)




    cv2.imshow("Merged Contours with Boxes", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_dataset_folder(dataset_path):
    """Process all images in the dataset folder and its subfolders"""
    # Counter for processed images
    total_processed = 0
    total_failed = 0
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            # Check if the file is an image (based on common extensions)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                print(f"Processing: {image_path}")
                
                # Invert the image
                if extract(image_path):
                    total_processed += 1
                else:
                    total_failed += 1

    # Print summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {total_processed} images")
    print(f"Failed to process: {total_failed} images")

if __name__ == "__main__":
    dataset_path = "datasetTemp"  # Replace with your dataset folder path
    process_dataset_folder(dataset_path)
