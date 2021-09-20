import numpy as np
import cv2


class EstimatorModule:

    def __init__(self, image, croppedarray):
        self.croppedarray = croppedarray
        self.est_model = '2d_model/pose_deploy.prototxt'
        self.est_weight = '2d_model/pose_iter_584000.caffemodel'
        self.est_net = cv2.dnn.readNetFromCaffe(self.est_model, self.est_weight)
        print('model loaded...')
        self.image = image
        self.keypointarray = self.keypointEstimator()
        print('keypoint estimated')

    # Estimate keypoints in local frame
    def keypointEstimator(self):
        croppedarray = self.croppedarray
        keypoint_array = []
        for i in range(len(croppedarray)):
            img = croppedarray[i][1]
            padleft = croppedarray[i][0]
            wc, hc, cc = img.shape
            wn = 250
            hn = wn * int(hc / wc)
            inpBlob_est = cv2.dnn.blobFromImage(img, 1.0 / 255, (wn, hn), (0, 0, 0), swapRB=False, crop=False)
            est_net = self.est_net
            est_net.setInput(inpBlob_est)
            output = est_net.forward()
            keypoints = []
            for i in range(15):
                probMap = output[0, i, :, :]
                probMap = cv2.resize(probMap, (wc, hc))
                threshold = 0.08
                mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
                mapMask = np.uint8(mapSmooth > threshold)
                # find blobs
                contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

                # find maxima for each blob

                for cnt in contours:
                    blobMask = np.zeros(mapMask.shape)
                    blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
                    maskedProbMap = mapSmooth * blobMask
                    _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
                    keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

            keypoint_array.append(keypoints)

        return keypoint_array

    # transform coordinates to global frame
    def coordtransform(self):
        keypointarray = np.array(self.keypointarray)
        croppedarray = self.croppedarray
        bbox = croppedarray[..., 2]
        padding = croppedarray[..., 0]
        allKeypoints = []
        for i in range(len(keypointarray)):
            padleft = padding[i]
            xmin = bbox[i][0]
            ymin = bbox[i][2]
            kpts = np.array(keypointarray[i])
            keypointTransformed = []
            xut = kpts[..., 0]
            yut = kpts[..., 1]
            prob = kpts[..., 2]
            for i in range(len(kpts)):
                xt = xut[i] + xmin - padleft
                yt = yut[i] + ymin
                p = prob[i]
                keypointTransformed.append((xt, yt, p))
            allKeypoints.append(keypointTransformed)
        return np.array(allKeypoints)