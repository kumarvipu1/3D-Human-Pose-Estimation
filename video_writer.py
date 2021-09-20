import cv2
import numpy as np
import datetime

def writeSkeleton(videofile, kparray):
    now = datetime.datetime.now()
    cap = cv2.VideoCapture(videofile)
    pose2d = np.array(kparray)
    POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4],
                  [5, 6], [6, 7], [8, 9], [8, 12], [9, 10], [12, 13], [13, 14],
                  [10, 11]]
    hasFrame, frame = cap.read()
    output_path = 'output2d/outputvid_{}_{}_{}.avi'.format(str(now.hour),str(now.minute),str(now.day))
    vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24,
                                 (frame.shape[1], frame.shape[0]))

    for i in range(len(pose2d)):
        ret, frame = cap.read()
        if not ret or (len(pose2d[i][j]) != 15 for j in range(len(pose2d[i]))):
            pass

        kparr = pose2d[i]
        for j in range(len(kparr)):
            keypoints = np.array(kparr[j])
            x = keypoints[..., 0]
            y = keypoints[..., 1]
            for k in range(len(keypoints)):
                cv2.circle(frame, (int(x[k]), int(y[k])),
                           3, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)

            for l in POSE_PAIRS:
                try:
                    cv2.line(frame, (int(x[l[0]]), int(y[l[0]])), (int(x[l[1]]), int(y[l[1]])),
                             (0, 0, 255), 1, lineType=cv2.LINE_AA)
                except:
                    pass

        vid_writer.write(frame)
        print('frame written')

    cap.release()