import numpy as np
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras import backend as K
from tensorflow.keras import models
import datetime

class Inference():

    def __init__(self, model_json, model_weight, kp2d):

        model_graph = open(model_json, 'r')
        model_loaded = model_graph.read()
        model_graph.close()

        model = model_from_json(model_loaded)
        model.load_weights(model_weight)
        model.compile(loss = self.euc_dist_keras, optimizer = 'adam')
        self.model = model
        self.data = kp2d

    def euc_dist_keras(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))

    def inference(self):
        now = datetime.datetime.now()
        model = self.model
        data = self.data

        output_list = []
        for i in range(len(data)):
            print('estimating for frame {} of frames {}....'.format(i, len(data)))
            kparr = data[i]
            output_personwise = []
            for j in range(len(kparr)):
                x_std, y_std = [], []
                keypoints = np.array(kparr[j])
                x1 = keypoints[..., 0]
                y1 = keypoints[..., 1]
                try:
                    x = [x1[8], x1[1], x1[0],
                        x1[2], x1[3], x1[4],
                        x1[5], x1[6], x1[7],
                        x1[9], x1[10], x1[11],
                        x1[12], x1[13], x1[14]]

                    y = [y1[8], y1[1], y1[0],
                        y1[2], y1[3], y1[4],
                        y1[5], y1[6], y1[7],
                        y1[9], y1[10], y1[11],
                        y1[12], y1[13], y1[14]]

                    xm = np.mean(x)
                    ym = np.mean(y)
                    sigma_x = np.std(x)
                    sigma_y = np.std(y)
                    for l in range(len(keypoints)):
                        xs = (x[l] - xm) / ((sigma_x + sigma_y) / 2)
                        ys = (y[l] - ym) / ((sigma_x + sigma_y) / 2)
                        x_std.append(xs)
                        y_std.append(ys)
                    inpt = np.concatenate((x_std, y_std))
                    inpt = inpt.reshape(1, len(inpt))
                    output = model.predict(inpt)
                    z = output[0]
                    for k in range(len(z)):
                        z[k] = abs((z[k] * ((sigma_x + sigma_y) / 2)))
                    kpfin = np.stack((x, y, z), axis=1)
                    output_personwise.append(kpfin)
                except:
                    continue

            output_list.append(output_personwise)

        file = 'output3d/prediction_{}_{}_{}.npy'.format(str(now.hour),str(now.minute),str(now.day))
        np.save(file, np.array(output_list), allow_pickle = True)

        return output_list

