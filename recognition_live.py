import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import numpy as np
from vision_lib import (find_motor,
                        find_motor_shaft,
                        bound_contours,
                        cropped_padder,
                        crop_minAreaRect,
                        find_motor_orientation)
from wrs_recongition_cnn import WrsRecognitionCNN

class RecognitionLive:
    def __init__(self, window, window_title, video_source='./templates/motor_rotation_feed.webm',
                 find_motor=False,
                 taskboard=False):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        self.find_motor = find_motor
        self.task_board = taskboard

        if self.task_board:
            # load CNN for taskboard
            self.wrsCnn = WrsRecognitionCNN('recognition_CNN/networks/gray_deep-gpu_net.pb')
            self.max_batch_size = 20

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot = tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        if self.task_board:
            self.delay = 100
        else:
            self.delay = 15
        self.count = np.int64(0)
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        roi = frame.copy()
        self.count += 1
        if ret:
            if self.find_motor:
                # find motor
                roi, theta = find_motor_orientation(frame)

            if self.task_board:
                img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

                # Adaptive Guassian Threshold is to detect sharp edges in the Image. For more information Google it.
                gaussian_binary = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3.5)

                kernel = np.ones((3, 3), np.uint8)
                gaussian_binary = cv2.erode((gaussian_binary==0).astype(np.uint8)*255, kernel, iterations=1)
                gaussian_binary = cv2.dilate(gaussian_binary, kernel, iterations=1)
                centers_kernel = np.ones((15, 15), np.uint8)
                closed_img = cv2.morphologyEx(gaussian_binary, cv2.MORPH_CLOSE, centers_kernel)

                roi, rects = bound_contours(roi, closed_img)

                input_batch = np.zeros((len(rects), self.wrsCnn.height, self.wrsCnn.width, self.wrsCnn.channels))
                coords = np.zeros((len(rects), 2))

                j = 0
                for rect in rects:
                    # crop part
                    res = crop_minAreaRect(img_gray, rect)  # cropped image
                    if res is not None:
                        cropped_img, (x, y) = res
                        if cropped_img is not None and cropped_img.shape[0] < 300 and cropped_img.shape[1] < 300:
                            input_batch[j, :, :, :] = cropped_padder(
                                cropped_img.reshape((cropped_img.shape[0], cropped_img.shape[1], 1)))
                            coords[j, :] = [x, y]
                            j += 1

                batch_size = j
                if batch_size != 0:

                    if batch_size > self.max_batch_size:
                        batches= list(range(0, batch_size, self.max_batch_size))
                        classes = []
                        predictions = []
                        for i in range(len(batches)-1):
                            sub_classes, sub_predictions = self.wrsCnn.predict(input_batch[batches[i]:batches[i+1], :, :])
                            classes += sub_classes.tolist()
                            predictions += sub_predictions.tolist()
                        sub_classes, sub_predictions = self.wrsCnn.predict(input_batch[batches[-1]:batch_size, :, :])
                        classes += sub_classes.tolist()
                        predictions += sub_predictions.tolist()
                    else:
                        classes, predictions = self.wrsCnn.predict(input_batch[:j, :, :])

                    for i in range(batch_size):
                        cv2.putText(roi, str(classes[i]), tuple(coords[i, :].astype(np.int32)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    color=(255, 0, 0), thickness=2)
                    print(self.count)

            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(roi))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source='./templates/motor_rotation_feed.webm'):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return ret, None


# Create a window and pass it to the motor rotation app
RecognitionLive(tkinter.Tk(), "Tkinter and OpenCV",
                video_source='C:/Users/ls769/Pictures/Camera Roll/test_taskboard.mp4',
                find_motor=True,
                taskboard=False
                )