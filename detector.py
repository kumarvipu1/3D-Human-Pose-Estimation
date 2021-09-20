import numpy as np
import cv2

class DetectionModule:

    def __init__(self, image):
        # self.img = cv2.imread(image)
        self.img = image
        self.modeldef = 'detector_model/deploy.prototxt'
        self.det_weight = 'detector_model/VGG_MPII_COCO14_SSD_500x500_iter_60000.caffemodel'
        self.bbox = self.getbbox()

    # code referenced from https://github.com/Fang-Haoshu/RMPE
    def getbbox(self):
        img = self.img
        det_net = cv2.dnn.readNetFromCaffe(self.modeldef, self.det_weight)
        inpBlob = cv2.dnn.blobFromImage(img, 1, (250, 250), (0, 0, 0), swapRB=True, crop=False)
        det_net.setInput(inpBlob)
        detections = det_net.forward()
        print('prediction complete.....')

        # extracting params from bbbox
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        configThred = 0.3

        top_indices = [m for m, conf in enumerate(det_conf) if conf > configThred]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()

        top_labels = det_label[top_indices]
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        bb_box = []
        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1])-10)
            ymin = int(round(top_ymin[i] * img.shape[0])-25)
            xmax = int(round(top_xmax[i] * img.shape[1])+10)
            ymax = int(round(top_ymax[i] * img.shape[0])+25)
            bb_box.append([xmin, xmax, ymin, ymax])

        return bb_box

    # code referenced from https://github.com/Fang-Haoshu/RMPE
    def getdetectedFrames(self):
        croppedImgs = []
        b_box = self.bbox
        image = self.img
        for k in range(len(b_box)):
            cropimg = image[b_box[k][2]:b_box[k][3], b_box[k][0]:b_box[k][1]]
            paddedimg, padleft = self.padding(cropimg)
            croppedImgs.append((np.array(padleft), np.array(paddedimg), np.array(self.bbox[k])))
        return croppedImgs

    def padding(self, cropped_img):
        h, w, c = cropped_img.shape
        dim = max(h, w)  # giving extra padding
        padded_array = np.empty([dim, dim, 3], dtype=np.uint8)
        padded_array[:, :] = np.array([0, 0, 0])
        padup = (dim - h) / 2
        paddown = dim - padup
        padleft = (dim - w) / 2
        padright = dim - padleft
        padded_array[int(padup):int(paddown), int(padleft):int(padright)] = cropped_img
        return padded_array, int(padleft)