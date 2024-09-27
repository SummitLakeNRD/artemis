import os
import cv2
import json
from datetime import datetime


class output:
    def __init__(self, sitename, output_directory, conf_threshold, 
                 latitude, longitude):
        self.site = sitename
        self.output = os.path.abspath(os.path.join(output_directory, self.site))
        os.makedirs(self.output, exist_ok=True)
        self.conf = conf_threshold
        self.latitude = latitude
        self.longitude = longitude
        self.color = (0, 255, 255)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def imageOut(self, frame, bboxes, confidence, 
                 predicted_class, bbox_output):
        time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        file_prefix = time + '_' + self.site
        filename = os.path.join(self.output, str(file_prefix + '.png'))
        if bbox_output == 'yes' and max(confidence, default = 0) >= self.conf:
            for box, conf, pred_class in zip(bboxes, confidence, predicted_class):
                label = "%s" % (pred_class + ": " + str(round(conf, 2)))
                cv2.putText(frame, label, (round(box[0]), round(box[1]) - 10), 
                            self.font, 1, self.color, 1) 
                cv2.rectangle(frame, (round(box[0]), round(box[1])), 
                              (round(box[2]), round(box[3])),
                              self.color, 2)
            cv2.imwrite(filename, frame)
        elif max(confidence, default = 0) >= self.conf:
            cv2.imwrite(filename, frame)
        else:
            pass
        return file_prefix

    def jsonOut(self, file_prefix, bboxes, confidence, predicted_class):
        filename = os.path.join(self.output, str(file_prefix + '.json'))
        for boxes, conf, pred_class in zip(bboxes, confidence, predicted_class):
            json_data = {
                "datetime":str(datetime.now()),
                "site":self.site,
                "class":pred_class,
                "confidence":conf,
                "x1":boxes[0],
                "y1":boxes[1],
                "x2":boxes[2],
                "y2":boxes[3],
                "latitude":self.latitude,
                "longitude":self.longitude,
                "image_path":self.output + '/' + file_prefix + '.png'
            }
            with open("{}".format(filename), 'w') as f:
                json.dump(json_data, f)



