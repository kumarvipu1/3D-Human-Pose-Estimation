import cv2
import detector
import numpy as np
import estimator
import datetime


def posefromvid(videofile):
    now = datetime.datetime.now()
    cap = cv2.VideoCapture(videofile)
    property_id = int(cv2.CAP_PROP_FRAME_COUNT)
    length = int(cv2.VideoCapture.get(cap, property_id))
    keypoint_list = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detect = detector.DetectionModule(frame)
        cropidx = np.array(detect.getdetectedFrames())
        kpestimate = estimator.EstimatorModule(frame, cropidx)
        keypoints = np.array(kpestimate.coordtransform())
        if not isinstance(keypoints, np.ndarray):
            # failed to detect human
            keypoint_list.append(None)
        else:
            keypoint_list.append(keypoints)
            count += 1
            print(f'Estimating frame {count} of frames {length}')
    cap.release()
    pose2d = keypoint_list
    pose2d_file = 'output2d/pose2d_{}_{}_{}.npy'.format(str(now.hour),str(now.minute),str(now.day))
    np.save(pose2d_file, pose2d)
    return pose2d




