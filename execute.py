import cv2
import signal
import threading
from argparse import ArgumentParser


from src.videoSignal import videoSignal
from src.inference import inference
from src.frameImport import frameImport
from src.output import output


def main():
    # Main arguments included in command line execution of program,
    # useful for running program from a systemd service file.
    
    parser = ArgumentParser(description='Artemis AI program for SLPT wildlife detection')
    parser.add_argument('video_source', type=str, help='Provide RTSP link for IP camera')
    parser.add_argument('model', type=str, help='Provide/path/to/megadetector/model.pt')
    parser.add_argument('conf_threshold', type=float, default=0.5, help='Confidence value for positive detection')
    parser.add_argument('gpu_device', type=str, help='Select GPU number (will be 0-7 for Lambda server)')
    parser.add_argument('site_name', type=str, help='Name of game camera site')
    parser.add_argument('latitude', type=str, help='Latitude of game camera site')
    parser.add_argument('longitude', type=str, help='Longitude of game camera site')
    parser.add_argument('output_directory', type=str, default='./outfile/', help='Output file directory')
    parser.add_argument('--output_with_bounding_boxes', type=str, default='no', help='enter either yes or no to add bounding boxes to video output')
    args = parser.parse_args()


    vs = videoSignal()
    signal.signal(signal.SIGINT, vs.handler)
    fi = frameImport(args.video_source, vs)

    i = inference(args.model, args.conf_threshold, args.gpu_device)
    o = output(args.site_name, args.output_directory, args.conf_threshold, 
               args.latitude, args.longitude)

    t1 = threading.Thread(target=fi.receiveFrame)
    t1.start()

    # Run detection on stream
    while vs.keep_running():
        frame = fi.grabFrame() #Imports frame from video source
        if frame is None:
            break
        #frame = cv2.resize(frame, (640, 480))
        boxes, conf, pred_class = i.detection(frame)
        print(conf)
        file_prefix = o.imageOut(frame, boxes, conf, pred_class, 
                                 args.output_with_bounding_boxes)
        o.jsonOut(file_prefix, boxes, conf, pred_class)


    t1.join()

if __name__ == '__main__':
    main()

