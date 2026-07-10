import cv2
import mediapipe as mp


class CameraCapture:
    """Frame and face-landmark capture for webcam now, wearable camera later."""

    def __init__(self, camera_index: int = 0) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def get_frame_and_landmarks(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        landmarks = None
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

        return frame, landmarks

    def release(self) -> None:
        self.face_mesh.close()
        self.cap.release()
        cv2.destroyAllWindows()
