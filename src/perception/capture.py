import time
from dataclasses import replace

import cv2
import mediapipe as mp

from src.perception.presence_detector import PresenceDetector, PresenceFrame, SmokingEventTracker
from src.perception.shoulder_tracker import ShoulderSample, ShoulderTracker


class CameraCapture:
    """Frame and face-landmark capture for the local webcam."""

    def __init__(self, camera_index: int = 0, *, detect_presence: bool = False) -> None:
        self.camera_index = camera_index
        self.failed_reads = 0
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam index {camera_index}. "
                "Check camera permissions or choose another camera in settings."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._presence_detector = PresenceDetector() if detect_presence else None
        self._smoking_tracker = SmokingEventTracker() if detect_presence else None
        self._pose = None
        self._shoulder_tracker = ShoulderTracker() if detect_presence else None
        self.last_shoulder_sample: ShoulderSample | None = None
        self.last_pose_landmarks = None
        self._presence_stride = 1
        self._pose_stride = 1
        self._frame_index = 0
        self._cached_presence: PresenceFrame | None = None
        if detect_presence:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def get_frame_and_landmarks(self):
        ret, frame = self.cap.read()
        if not ret:
            self.failed_reads += 1
            return None, None

        self.failed_reads = 0

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        landmarks = None
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

        return frame, landmarks

    def get_frame_landmarks_presence(self) -> tuple[object | None, object | None, PresenceFrame | None]:
        ret, frame = self.cap.read()
        if not ret:
            self.failed_reads += 1
            return None, None, None

        self.failed_reads = 0
        self._frame_index += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        landmarks = None
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

        run_pose = self._pose is not None and self._frame_index % self._pose_stride == 0
        pose_landmarks = self.last_pose_landmarks
        if run_pose:
            pose_results = self._pose.process(rgb_frame)
            if pose_results.pose_landmarks:
                pose_landmarks = pose_results.pose_landmarks.landmark
            else:
                pose_landmarks = None
            self.last_pose_landmarks = pose_landmarks

        shoulder = self.last_shoulder_sample
        if self._shoulder_tracker is not None and run_pose:
            shoulder = self._shoulder_tracker.update(pose_landmarks)
            self.last_shoulder_sample = shoulder

        presence = self._cached_presence
        run_presence = self._presence_detector is not None and self._frame_index % self._presence_stride == 0
        if run_presence:
            presence = self._presence_detector.detect(rgb_frame, landmarks)
            self._cached_presence = presence

        if self._smoking_tracker is not None and presence is not None:
            events = self._smoking_tracker.update(
                presence,
                rgb_frame,
                landmarks,
                time.monotonic(),
                shoulder=shoulder,
            )
            if events:
                presence = replace(presence, events=events)
                self._cached_presence = presence

        return frame, landmarks, presence

    def release(self) -> None:
        if self._presence_detector is not None:
            self._presence_detector.close()
        if self._pose is not None:
            self._pose.close()
        self.face_mesh.close()
        self.cap.release()
        cv2.destroyAllWindows()
