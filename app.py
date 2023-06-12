from flask import Flask, render_template, request, redirect, url_for, session, Response, flash, jsonify
import random
app = Flask(__name__)
import os

@app.route("/", methods=["POST", "GET"])
def index():
    return render_template('home.html')

@app.route('/cameratampering', methods=['POST', 'GET'])
def cameratampering():
    return render_template('cameratampering.html')

@app.route('/rcnn', methods=['POST', 'GET'])
def rcnn():
    return render_template('rcnn.html')

@app.route('/face', methods=['POST', 'GET'])
def face1():
    return render_template('face.html')

@app.route('/video', methods=['POST', 'GET'])
def video():
    print("Working")
    import numpy as np
    import winsound
    from playsound import playsound
    import cv2
    cap = cv2.VideoCapture(1)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    kernel = np.ones((5,5), np.uint8)
    
    window_open = True  # Initialize flag variable
    while(window_open):
        ret, frame = cap.read()
        if(frame is None):
            print("End of frame")
            break
        else:
            a = 0
            bounding_rect = []
            fgmask = fgbg.apply(frame)
            fgmask= cv2.erode(fgmask, kernel, iterations=5) 
            fgmask = cv2.dilate(fgmask, kernel, iterations = 5)
            cv2.imshow('frame',frame)
            contours,_ = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for i in range(0,len(contours)):
                bounding_rect.append(cv2.boundingRect(contours[i]))
            for i in range(0,len(contours)):
                if bounding_rect[i][2] >=40 or bounding_rect[i][3] >=40:
                    a = a+(bounding_rect[i][2])*bounding_rect[i][3]
                if(a >=int(frame.shape[0])*int(frame.shape[1])/3):
                    cv2.putText(frame,"TAMPERING DETECTED",(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
                    winsound.Beep(1000, 1500)
                    print('playing sound using winsound')  
                    break
            cv2.imshow('frame',frame)
            
            key = cv2.waitKey(1) & 0xFF  # Listen for key press event
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if key == ord('q'):  # If "q" key is pressed, set flag variable to False
                window_open = False
                cv2.destroyAllWindows()
                render_template('home.html')
    cap.release()  # Release video capture object
    cv2.destroyAllWindows()  # Destroy all windows
    
    return render_template('home.html')


@app.route('/video1', methods=['POST', 'GET'])
def video1():
    import os
    from werkzeug.utils import secure_filename  
    f = request.files['file']
    f.save(secure_filename(f.filename))
    print('file uploaded successfully')
    print("Working")
    import numpy as np
    import cv2
    cap = cv2.VideoCapture(f.filename)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    ret, frame = cap.read() 
    fgmask = fgbg.apply(frame)
    kernel = np.ones((5,5), np.uint8)
    while(True):
        ret, frame = cap.read()
        if(frame is None):
            print("End of frame")
            break
        else:
            a = 0
            bounding_rect = []
            fgmask = fgbg.apply(frame)
            fgmask= cv2.erode(fgmask, kernel, iterations=5) 
            fgmask = cv2.dilate(fgmask, kernel, iterations = 5)
            cv2.imshow('frame',frame)
            contours,_ = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            for i in range(0,len(contours)):
                bounding_rect.append(cv2.boundingRect(contours[i]))
            for i in range(0,len(contours)):
                if bounding_rect[i][2] >=40 or bounding_rect[i][3] >=40:
                    a = a+(bounding_rect[i][2])*bounding_rect[i][3]
                if(a >=int(frame.shape[0])*int(frame.shape[1])/3):
                    cv2.putText(frame,"TAMPERING DETECTED",(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)   
                cv2.imshow('frame',frame)        
               
        k = cv2.waitKey(30) & 0xff
        render_template('home.html')
        if k == 27:
            break
    return render_template('home.html')

@app.route('/object', methods=['POST', 'GET'])
def object():
    from imutils.video import VideoStream
    from imutils.video import FPS
    import numpy as np
    import argparse
    import imutils
    import time
    import cv2

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor","knife"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

    # initialize the video stream, allow the cammera sensor to warmup,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    vs = VideoStream(src=1).start()
    time.sleep(2.0)
    fps = FPS().start()

    # loop over the frames from the video stream
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
	    # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)

	    # pass the blob through the network and obtain the detections and
	    # predictions
        net.setInput(blob)
        detections = net.forward()

	# loop over the detections
        for i in np.arange(0, detections.shape[2]):
		    # extract the confidence (i.e., probability) associated with
		    # the prediction
            confidence = detections[0, 0, i, 2]

		    # filter out weak detections by ensuring the `confidence` is
		    # greater than the minimum confidence
            if confidence > 0.2:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
				    confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

	    # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            cv2.destroyAllWindows()
            render_template('home.html')
            break

	# update the FPS counter
        fps.update()

# stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    return render_template('home.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    import os
    from werkzeug.utils import secure_filename  
    f = request.files['file']
    f.save(secure_filename(f.filename))
    print('file uploaded successfully')

    print("Working")
   
    import numpy as np
    import argparse
    import time
    import cv2
    import os
        
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True,
    # 	help="path to input image")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
        help="threshold when applyong non-maxima suppression")
    args = vars(ap.parse_args())

    # load the COCO class labels our YOLO model was trained on
    labelsPath = "YOLO\yolo-coco\coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = "YOLO\yolo-coco\yolov3.weights"
    configPath = "YOLO\yolo-coco\yolov3.cfg"

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading RCNN from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load our input image and grab its spatial dimensions
    image = cv2.imread(f.filename)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    lin = net.getLayerNames()
    ln = [lin[i-1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    # [,frame,no of detections,[classid,class score,conf,x,y,h,w]
    end = time.time()

    # show timing information on YOLO
    print("[INFO] RCNN took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
        args["threshold"])

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)

    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    return render_template('home.html')

from utils import *
@app.route('/face-det', methods=['POST', 'GET'])
def face():
    import argparse
    import sys
    import os
    import cv2
    import utils
    import winsound
    #####################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-cfg', type=str, default='./cfg/yolov3-face.cfg',
                        help='path to config file')
    parser.add_argument('--model-weights', type=str,
                        default='./model-weights/yolov3-wider_16000.weights',
                        help='path to weights of model')
    parser.add_argument('--image', type=str, default='C:\\Users\\Aarush\\Desktop\\Files\\Project\\atm.jpg',
                    help='path to image file')
    parser.add_argument('--video', type=str, default='C:\\Users\\Aarush\\Desktop\\Files\\Project\\75.mp4',
                    help='path to video file')
    parser.add_argument('--src', type=int, default=1,
                        help='source of the camera')
    parser.add_argument('--output-dir', type=str, default='outputs/',
                        help='path to the output directory')
    args = parser.parse_args()

    #####################################################################
    # print the arguments
    print('----- info -----')
    print('[i] The config file: ', args.model_cfg)
    print('[i] The weights of model file: ', args.model_weights)
    print('[i] Path to image file: ', args.image)
    print('[i] Path to video file: ', args.video)
    print('###########################################################\n')

    # check outputs directory
    if not os.path.exists("outputs/"):
        print('==> Creating the {} directory...'.format("outputs/"))
        os.makedirs("outputs/")
    else:
        print('==> Skipping create the {} directory...'.format("outputs/"))

    # Give the configuration and weight files for the model and load the network
    # using them.
    net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



    wind_name = 'face detection using YOLOv3'
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

    output_file = ''

    # if args.image:
    #     if not os.path.isfile(args.image):
    #         print("[!] ==> Input image file {} doesn't exist".format(args.image))
    #         sys.exit(1)
    #     cap = cv2.VideoCapture(args.image)
    #     output_file = args.image[:-4].rsplit('/')[-1] + '_yoloface.jpg'
    # elif args.video:
    #     if not os.path.isfile(args.video):
    #         print("[!] ==> Input video file {} doesn't exist".format(args.video))
    #         sys.exit(1)
    #     cap = cv2.VideoCapture(args.video)
    #     output_file = args.video[:-4].rsplit('/')[-1] + '_yoloface.avi'
    # else:
    #     #Get data from the camera
    cap = cv2.VideoCapture(0)

    # Get the video writer initialized to save the output video
    #if not args.image:
    video_writer = cv2.VideoWriter(os.path.join("outputs/", output_file),
                                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                cap.get(cv2.CAP_PROP_FPS), (
                                    round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                    round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:

        has_frame, frame = cap.read()

        # Stop the program if reached end of video
        if not has_frame:
            print('[i] ==> Done processing!!!')
            print('[i] ==> Output file is stored at', os.path.join("outputs/", output_file))
            cv2.waitKey(1000)
            break

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416),[0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        print('[i] ==> # detected faces: {}'.format(len(faces)))
        print('#' * 60)
        if len(faces) > 2 :
            winsound.Beep(1000, 1500)
            print('playing sound using playsound')  
        # initialize the set of information we'll displaying on the frame
        info = [
            ('number of faces detected', '{}'.format(len(faces)))
        ]


        for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            cv2.putText(frame, text, (10, (i * 20) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

        # Save the output video to file
        # if args.image:
        #     cv2.imwrite(os.path.join("outputs/", output_file), frame.astype(np.uint8))
        # else:
        video_writer.write(frame.astype(np.uint8))

        cv2.imshow(wind_name, frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            cv2.destroyAllWindows()
            print('[i] ==> Interrupted by user!')
            render_template('home.html')
            break

    cap.release()
    cv2.destroyAllWindows()

    print('==> All done!')
    print('***********************************************************')
        
    
        
    cv2.waitKey(0)
    return render_template('home.html')
    
if __name__ == '__main__':
    app.run(debug=True)