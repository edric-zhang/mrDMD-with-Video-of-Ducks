import cv2
import numpy as np
import scipy.io
import time

# 1. Point to your local file
# Use the full, absolute path to your Downloads folder
local_filename = r"C:\Users\edric\Downloads\ducks_take_off_420_720p50.y4m"
mat_filename = r"C:\Users\edric\Downloads\ducks_snapshot_matrix.mat"


start_time = time.time()
cap = cv2.VideoCapture(local_filename)
frames_list = []
max_frames = 500  

print("Processing frames...")
while len(frames_list) < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 1. Convert from BGR to standard RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. ADD THIS LINE: Resize to 640x360 (Drops RAM requirements drastically!)
    resized_frame = cv2.resize(rgb_frame, (160, 90))
    
    # 3. Flatten the resized frame instead
    flattened_vector = resized_frame.flatten()
    frames_list.append(flattened_vector)


cap.release()

# 2. Stack all 500 vectors side-by-side to make the 2D matrix
ducks = np.column_stack(frames_list)

print("Saving to .mat file...")
# 3. Save as a MATLAB-compatible file with variable name 'X'
scipy.io.savemat(mat_filename, {"ducks": ducks})

end_time = time.time()
print(f"Done! Total runtime: {end_time - start_time:.2f} seconds.")
print(f"Matrix shape: {ducks.shape}")
