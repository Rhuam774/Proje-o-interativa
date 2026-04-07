import json
import os
import numpy as np

CONFIG_FILE = "config.json"

class Config:
    def __init__(self):
        self.camera_id = 0
        self.projector_screen_index = 0
        self.homography_matrix = None
        self.points_camera = []
        self.points_projector = []
        self.simulation_mode = False
        self.load()

    def save(self):
        data = {
            "camera_id": self.camera_id,
            "projector_screen_index": self.projector_screen_index,
            "homography_matrix": self.homography_matrix.tolist() if self.homography_matrix is not None else None,
            "points_camera": self.points_camera,
            "points_projector": self.points_projector,
            "simulation_mode": self.simulation_mode
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=4)

    def load(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    data = json.load(f)
                    self.camera_id = data.get("camera_id", 0)
                    
                    # Detecta monitores disponíveis
                    import screeninfo
                    monitors = screeninfo.get_monitors()
                    if len(monitors) > 1:
                        self.projector_screen_index = data.get("projector_screen_index", 1)
                    else:
                        self.projector_screen_index = 0 # Fallback se tiver só 1 tela
                        
                    h_matrix = data.get("homography_matrix")
                    if h_matrix is not None:
                        self.homography_matrix = np.array(h_matrix)
                    self.points_camera = data.get("points_camera", [])
                    self.points_projector = data.get("points_projector", [])
                    self.simulation_mode = data.get("simulation_mode", False)
            except Exception as e:
                print(f"Erro ao carregar config: {e}")

    def list_devices_info(self):
        import screeninfo
        import cv2
        monitors = screeninfo.get_monitors()
        cameras = []
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append(i)
                cap.release()
        return monitors, cameras

    def list_devices(self):
        monitors, cameras = self.list_devices_info()
        print("\n--- LISTA DE MONITORES ---")
        for i, m in enumerate(monitors):
            print(f"[{i}] {m.name} - {m.width}x{m.height}")
        print("\n--- LISTA DE CAMERAS ---")
        for c in cameras:
            print(f"[{c}] Câmera detectada")

    def update_homography(self, pts_cam, pts_proj):
        self.points_camera = pts_cam
        self.points_projector = pts_proj
        # Calcular homografia
        import cv2
        h, status = cv2.findHomography(np.array(pts_cam), np.array(pts_proj))
        self.homography_matrix = h
        self.save()
