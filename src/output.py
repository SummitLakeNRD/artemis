import os
import cv2
import json
import psycopg
from uuid import uuid4
from datetime import datetime


class outputFiles:
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
        time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S-%f")
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
        counter = 0
        for boxes, conf, pred_class in zip(bboxes, confidence, predicted_class):
            filename = os.path.join(self.output, str(file_prefix + '_' + str(counter) + '.json'))
            counter += 1
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

class outputDatabase:        
    def __init__(self, sitename, conf_threshold, 
                 latitude, longitude):
        self.site = sitename
        self.conf = conf_threshold
        self.latitude = latitude
        self.longitude = longitude
        self.color = (0, 255, 255)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def postgresAppend(self, frame, bboxes, confidence, 
                       predicted_class, bbox_output):
        # Conditional formating of image and converting to byte array
        date = str(datetime.now())
        if bbox_output == 'yes' and max(confidence, default = 0) >= self.conf:
            for box, conf, pred_class in zip(bboxes, confidence, predicted_class):
                label = "%s" % (pred_class + ": " + str(round(conf, 2)))
                cv2.putText(frame, label, (round(box[0]), round(box[1]) - 10), 
                            self.font, 1, self.color, 1) 
                cv2.rectangle(frame, (round(box[0]), round(box[1])), 
                              (round(box[2]), round(box[3])),
                              self.color, 2)
                image_bytea = cv2.imencode('.jpg', frame)[1].tobytes()
        elif max(confidence, default = 0) >= self.conf:
                image_bytea = cv2.imencode('.jpg', frame)[1].tobytes()
        else:
            pass
        # Create connection tunnel to postgres database
        with psycopg.connect(
                dbname="<DBNAME>", 
                user="<USER>",
                password="<PASSWORD>",
                host='<HOST>',
                port='<PORT>') as conn:
            # Open a cursor to perform database operations
            with conn.cursor() as cur:
                for boxes, conf, pred_class in zip(bboxes, confidence, predicted_class):
                    # Pass data to fill query placeholders and let Psycopg perform
                    # the correct conversion 
                    cur.execute(
                        """INSERT INTO game_camera
                        (uuid, datetime, site, class, confidence, x1, y1,
                        x2, y2, latitude, longitude, image) VALUES 
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (uuid4(), date, self.site, pred_class, conf,
                         boxes[0], boxes[1], boxes[2], boxes[3],
                         self.latitude, self.longitude, 
                         psycopg.Binary(image_bytea)))
                    # Make the changes to the database persistent
                    conn.commit()


