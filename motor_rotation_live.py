import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import numpy as np
from vision_motor_lib import (find_motor, find_motor_shaft)

class MotorRotationFinder:

    motor_loc = None
    motor_prev_loc = None
    shaft_loc = None
    shaft_prev_loc = None
    angle = None
    prev_angle = None

    def __init__(self, window, window_title, video_source='./templates/motor_rotation_feed.webm'):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot = tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 5
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

        if ret:
            self.count += 1
            # find motor
            image_rgb = frame.copy()
            roi, motor_loc = find_motor(image_rgb)
            roi = cv2.cvtColor(roi, cv2.COLOR_LAB2RGB)
            if motor_loc is not None:
                x, y, w, h = motor_loc  # unpack
                cropped_motor = image_rgb[y:y + h, x:x + w]

                # find shaft
                self.prev_loc = self.motor_loc
                self.motor_loc = np.array([np.int32(x + w / 2), np.int32(y + h / 2)])
                shaft_loc = find_motor_shaft(cropped_motor, mode='circles')
                if shaft_loc is not None:
                    self.shaft_prev_loc = self.shaft_loc
                    self.shaft_loc = shaft_loc + np.array([x, y])

                    # compute angle from motor and shaft location
                    theta = np.rad2deg(np.arctan2(*(self.shaft_loc[::-1] - self.motor_loc[::-1])))
                    self.angle = theta
                    self.prev_angle = self.angle

                    # draw in roi for display
                    compass_vect = np.int32((self.shaft_loc - self.motor_loc) * 5 + self.motor_loc)
                    cv2.line(roi, tuple(self.motor_loc), tuple(compass_vect), (255, 0, 0), 15)
            if self.angle is not None:
                cv2.putText(roi,
                                str('Motor Orientation: {0:.2f}'.format(self.angle)),
                                (25, 70),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                color=(255, 0, 0),
                                thickness=3)
                cv2.putText(roi,
                            'Frame Number: ' + str(self.count),
                            (25, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            color=(255, 0, 0),
                            thickness=3)
                frame = roi

            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source='./templates/motor_rotation_feed.webm'):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        self.vid.set(3, 1920)
        self.vid.set(4, 1080)
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
MotorRotationFinder(tkinter.Tk(), "Tkinter and OpenCV")