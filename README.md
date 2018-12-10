# 3D-DeepBox-for-EECS545
Step 0:
Install python3, Tensorflow >= 1.4, and some popular libraries in python3 (OpenCV, numpy, tqdm...)

Step 1:
Modify the training set paths defined in "config.py", create folders when needed

Step 2:
Run "python3 data.py", make sure "label_crop.txt" and "label_stats.txt" are generated successfully

Step 3:
Run "python3 train.py", wait patiently... It should converge fast (10 epochs would be enough)

Step 4:
Modify the paths in "predict.sh", create folders when needed, and run "sh predict.sh"

Step 5:
Modify the paths in "visualize.sh", take the prediction as label here, and run "visualize.py" to see the result!
