import cv2
import pandas as pd
import numpy as np
import math, csv

net = cv2.dnn.readNetFromCaffe("MobileNet-SSD-master/MobileNet-SSD-master/deploy.prototxt", "MobileNet-SSD-master/MobileNet-SSD-master/mobilenet_iter_73000.caffemodel")
def detect_vehicles_dnn(frame, net):
    """
    Detect vehicles (cars, buses, and trucks) in the given frame using a pre-trained DNN model.
    Args:
    - frame (numpy array): The image frame to process.
    - net (cv2.dnn_Net): The DNN network loaded with pre-trained weights and configuration.

    Returns:
    - list of rectangles: Each rectangle corresponds to a detected vehicle.
    """
    # Pre-process the image

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    
    # Get detections
    detections = net.forward()
    
    # Extract bounding boxes for detected vehicles
    vehicle_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.1 :  # Confidence threshold
            class_id = int(detections[0, 0, i, 1])
            # Check for class IDs corresponding to car, bus, and truck
            if class_id in [2, 5, 7]:  # Class IDs for bus, car, and truck in COCO dataset
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                vehicle_boxes.append(box.astype("int"))
                
    return vehicle_boxes

def update_headlight_matrix(frame, detected_vehicles, pixel_matrix):
    updated_pixel_matrix = pixel_matrix.copy()
    # Calculate the size of individual pixels in the pixel matrix
    px_height, px_width = frame.shape[0] // pixel_matrix.shape[0], frame.shape[1] // pixel_matrix.shape[1]
    
    # For each detected vehicle, determine which pixels to turn off for the top half
    for (startX, startY, endX, endY) in detected_vehicles:
        midpointY = (startY + endY) // 2  # Calculate the midpoint of the vertical coordinates
        start_col = startX // px_width
        end_col = endX // px_width
        start_row = midpointY // px_height  # Start from the midpoint to represent the top half
        end_row = endY // px_height
        
        # Turn off the LEDs in the corresponding region of the top half
        updated_pixel_matrix[start_row:end_row, start_col:end_col] = 0
        
    return updated_pixel_matrix

evaluation_file = pd.DataFrame()
print(evaluation_file)


# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 09:37:04 2023

@author: marcosh
"""
# import participant_script as ps

# Function to simulate the black out of specific set of pixels from pixel_matrix
def update_pixel_matrix(frame, pixel_matrix, color=(230, 230, 230), thickness=-1):    
    h, w, _ = frame.shape
    rows, cols = pixel_matrix.shape
        
    # High Beam ROI:
    black_border = 125
    low_beam_height = int(0.42*h)               # 42% of frame height
    sky_area_height = int(0.40*h)               # 40% of frame height
    side_area = int(0.05*w) + black_border      # 5% fo frame width of each side 
    
    pixel_matrix_width = w-(2*side_area) 
    pixel_matrix_height = h-sky_area_height-low_beam_height
    py, px = math.floor(pixel_matrix_height / rows), math.floor(pixel_matrix_width / cols) # pixel size

    pixel_columns = np.linspace(start = side_area, stop = w-side_area, num = cols, dtype = int)
    pixel_rows = np.linspace(start = sky_area_height, stop = h-low_beam_height, num = rows, dtype = int)
    for j, x in enumerate (pixel_columns):
        for i, y in enumerate (pixel_rows):
            if pixel_matrix[i,j] == 0:
                # black out simulation
                cv2.rectangle(frame, (x, y), (x + px, y + py), color=color, thickness=thickness) 

# Function to simulate the pixel_matrix
def draw_pixel_matrix(frame, pixel_matrix_shape, color=(230, 230, 230), thickness=1):
    h, w, _ = frame.shape
    rows, cols = pixel_matrix_shape

    # High Beam ROI:
    black_border = 125
    low_beam_height = int(0.42*h)               # 42% of frame height
    sky_area_height = int(0.40*h)               # 40% of frame height
    side_area = int(0.05*w) + black_border      # 5% fo frame width of each side 
    
    # Draw Columns
    x_coords = np.linspace(start = side_area, stop = w-side_area, num = cols, dtype = int)
    for x in x_coords:
         cv2.line(frame, (x, sky_area_height), (x, h-low_beam_height), color=color, thickness=thickness)
         
    # Draw Rows 
    y_coords = np.linspace(start = sky_area_height, stop = h-low_beam_height, num = rows, dtype = int)
    for y in y_coords:
        cv2.line(frame, (side_area, y), (w-side_area, y), color=color, thickness=thickness)

# Function to save each key_pixel_matrix
def save_key_pixel_matrix(pixel_matrix, filename, key_frame):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Key Frame:", key_frame])
        csv_writer.writerows(pixel_matrix)
        csvfile.write('\n')
        
# Initialize a list to store evaluation data

data_frames_with_details = []
### MAIN CODE ###

evaluation_file = pd.DataFrame()  # Initialize to empty dataframe
def main():
    print('Starting main function...')
    evaluation_file = pd.DataFrame()  # Ensure initialization at the start
    # Open video:
    video_name = "Test Video Level 1.mp4" # set name of video to test
    print('Attempting to open video...')
    video = cv2.VideoCapture(video_name)
    print('Video opened successfully.')
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total frames in the video: {total_frames}')
    
    # Create Window
    
    cv2.namedWindow("Testing Window", cv2.WINDOW_NORMAL)
    
    # Create Pixel Matrix
    prows = 64
    pcolumns = 256
    pixel_matrix_shape = [prows, pcolumns]
    pixel_matrix = np.full((prows, pcolumns), 1)
    
    # Key Frame parameters
    fr = 0 # frame counter
    # key_frames = [60, 125, 240, 275, 593, 635, 1000, 1095, 1136, 1450, 1520, 
    #               1552, 2060, 2130, 2140, 2310, 2340, 2420, 2800, 3060, 3165,
    #               3246, 3265, 3305, 3543, 3605, 3740, 3794, 4355, 4752, 4865,
    #               5050, 5230, 5255, 5455, 5800, 5960, 6595] # Key Frames for Video Phase 1
    # evaluation_frames = [125]
    # evaluation_frames = [1095, 1150, 1450, 1552, 2130, 2340,  240, 2420, 3060, 3265, 3605, 4355, 4865, 5230, 5255, 5455, 5960, 60, 635, 6595]   
    evaluation_frames = [1000, 1136,  125, 1520, 2060, 2140, 2310,  275, 2800, 3165, 3246, 3305,3543, 3740, 3794, 4752, 5050,5800,593]
    while video.isOpened():  
        # Read frame by frame
        success, frame = video.read()

        # Check if the frame was read successfully
        if not success:
            print(f'Failed to read frame at frame number: {fr}')
            break
        # print(fr)
        # # Print every 100 frames for brevity
        # if fr % 100 == 0:
        #     print(f'Processing frame number: {fr}')
        # Sample frame processing
        if fr in evaluation_frames:
            print(f'Processing sample frame: {fr}')
            try:
                print('Starting sample frame processing...')
                detected_vehicles = detect_vehicles_dnn(frame, net)
                print('Updating pixel matrix...')
                pixel_matrix = update_headlight_matrix(frame, detected_vehicles, pixel_matrix)
                # Uncomment the line below if you want to use the participant function
                # pixel_matrix = ps.participant_function(frame, pixel_matrix)
                print('Completed sample frame processing.')
            except Exception as e:
                print(f'Error during sample frame processing: {e}')

                    
                draw_pixel_matrix(frame, pixel_matrix_shape)            
                update_pixel_matrix(frame, pixel_matrix)                                      
                    
                # Display results
                cv2.imshow("Testing Window", frame)           
                    
                # save the generated pixel_matrix according to the respective key_frame
                print(f'Processing evaluation frame: {fr}')
            if fr in evaluation_frames: 
                save_key_pixel_matrix(pixel_matrix, f"key_pixel_matrix_{fr}.csv", fr)
                
                # Get the dimensions of the data
                num_rows, num_cols = pixel_matrix.shape

                # Create a dataframe with details for the current file
                details_df = pd.DataFrame({
                    'Frame': fr,
                    'Row': [row_idx + 1 for row_idx in range(num_rows) for _ in range(num_cols)],
                    'Column': [col_idx + 1 for _ in range(num_rows) for col_idx in range(num_cols)],
                    'Value': pd.DataFrame(pixel_matrix).values.flatten()
                })

                # Generate the composite key column
                
                details_df['CompositeKey'] = details_df['Frame'].astype(str) + '-' + (details_df['Row']).astype(str) + '-' +  details_df['Column'].astype(str)

                data_frames_with_details.append(details_df)
                # Concatenate all the dataframes into a single dataframe
                print('Populating evaluation_file DataFrame...')
                evaluation_file = pd.concat(data_frames_with_details, ignore_index=True)[['CompositeKey', 'Value']]
                    
            # Stop video
            quitButton = cv2.waitKey(25) & 0xFF == ord('q')
            closeButton = cv2.getWindowProperty('Testing Window', cv2.WND_PROP_VISIBLE) < 1
            if quitButton or closeButton: 
                break
        fr += 1
            
        # Wrong reading
    # else: break
    #write evaluation file to CSV    
    if not evaluation_file.empty:
        print('Attempting to write evaluation_file to CSV...')
    evaluation_file.to_csv('Evaluation_Data.csv', index=False)
    print('Completed main function execution.')
    
    video.release()
    cv2.destroyAllWindows()    

if __name__ == '__main__':
    main()
