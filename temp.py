import os
import pickle
from natsort import natsorted

keypoint_precision_score = 0.9
keypoint_recall_score = 0.3

keypoint_f1_score = 2 * (keypoint_precision_score * keypoint_recall_score) / (keypoint_precision_score + keypoint_recall_score)

keypoint_f1_score_2 = 2 * (keypoint_precision_score * keypoint_recall_score / (keypoint_precision_score + keypoint_recall_score))

print(keypoint_f1_score)
print(keypoint_f1_score_2)