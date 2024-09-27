import torch

class inference:
    def __init__(self, neural_network, model_confidence, processor_id):
        if torch.cuda.is_available():
            processor_id = processor_id
        else:
            processor_id = 'CPU'
        try: 
            self.model = torch.hub.load('./yolov5', 'custom', path=neural_network, 
                                        source='local', device=processor_id)
        except FileNotFoundError:
            print('YOLOv5 backend code file directory is incorrect, repair directory link')
        self.model.conf = model_confidence
        self.model.iou = 0.3 # 0-1 value for threshold to stop overlapping detections
        self.classes = self.model.names

    def detection(self, frame):
        results = self.model(frame).xyxy[0].tolist()
        boxes = [det[0:4] for det in results]
        confidence = [det[4] for det in results]
        pred_class = [self.classes[int(det[5])] for det in results]
        return boxes, confidence, pred_class
    
