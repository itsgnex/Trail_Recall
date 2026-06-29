import os

import cv2


def discover_cameras(max_index=10):
    cameras = []
    old_log_level = cv2.getLogLevel() if hasattr(cv2, "getLogLevel") else None
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)
    stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        for index in range(max_index + 1):
            capture = cv2.VideoCapture(index)
            ok, frame = capture.read() if capture.isOpened() else (False, None)
            capture.release()
            if not ok:
                continue

            cameras.append(index)
            print(f"Camera {index} is available.")
            cv2.putText(frame, f"Camera ID: {index}", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow(f"Camera ID: {index}", frame)
            cv2.waitKey(700)
            cv2.destroyWindow(f"Camera ID: {index}")
    finally:
        os.dup2(stderr, 2)
        os.close(stderr)
        os.close(devnull)
        if old_log_level is not None:
            cv2.setLogLevel(old_log_level)
    return cameras


def choose_camera(max_index=10):
    cameras = discover_cameras(max_index)
    if not cameras:
        print("No cameras found from index 0 to 10. Check camera connections and macOS camera permission.")
        return None

    while True:
        choice = input(f"Choose camera {cameras}: ").strip()
        try:
            index = int(choice)
        except ValueError:
            print("Please type a camera number.")
            continue
        if index in cameras:
            return index
        print(f"Camera {index} was not available. Choose one of {cameras}.")


class Camera:
    def __init__(self, index):
        self.index = index
        self.capture = None
        self.opened = False

    def __enter__(self):
        self.capture = cv2.VideoCapture(self.index)
        self.opened = self.capture.isOpened()
        if self.opened:
            cv2.namedWindow("Gaze Assistant", cv2.WINDOW_NORMAL)
        return self

    def __exit__(self, *_):
        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()

    def read(self):
        for _ in range(3):
            ok, frame = self.capture.read()
            if ok:
                return frame
        return None

    def show(self, frame):
        cv2.imshow("Gaze Assistant", frame)
        return cv2.waitKey(1) & 0xFF == ord("q")
