import cv2
import numpy as np
import screeninfo
from config import Config

class Calibration:
    def __init__(self, config):
        self.config = config
        self.points_camera = []
        self.points_projector = []
        self.screen = screeninfo.get_monitors()[self.config.projector_screen_index]
        self.w_proj = self.screen.width
        self.h_proj = self.screen.height

    def get_calibration_points(self):
        # Pontos relativos na projeção (10% de margem das bordas)
        margin = 0.1
        pts = [
            (int(self.w_proj * margin), int(self.h_proj * margin)),
            (int(self.w_proj * (1-margin)), int(self.h_proj * margin)),
            (int(self.w_proj * (1-margin)), int(self.h_proj * (1-margin))),
            (int(self.w_proj * margin), int(self.h_proj * (1-margin)))
        ]
        return pts

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points_camera) < 4:
                self.points_camera.append((x, y))
                print(f"Ponto capturado na câmera: {x}, {y}")

    def calibrate(self, cap):
        self.points_camera = []
        self.points_projector = self.get_calibration_points()
        
        # Criar janela do projetor
        proj_win = "Calibracao Projetor"
        cv2.namedWindow(proj_win, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(proj_win, self.screen.x, self.screen.y)
        cv2.setWindowProperty(proj_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Imagem preta para o projetor
        canvas = np.zeros((self.h_proj, self.w_proj, 3), dtype=np.uint8)
        
        # Desenhar as 4 cruzes
        for pt in self.points_projector:
            cv2.line(canvas, (pt[0]-20, pt[1]), (pt[0]+20, pt[1]), (255, 255, 255), 3)
            cv2.line(canvas, (pt[0], pt[1]-20), (pt[0], pt[1]+20), (255, 255, 255), 3)

        cv2.imshow(proj_win, canvas)

        # Janela da câmera
        cam_win = "Clique nas 4 cruzes que voce ve na CAMERA"
        cv2.namedWindow(cam_win)
        cv2.setMouseCallback(cam_win, self.mouse_callback)

        while len(self.points_camera) < 4:
            ret, frame = cap.read()
            if not ret: break
            
            # Mostrar pontos já clicados
            for pt in self.points_camera:
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)
            
            cv2.imshow(cam_win, frame)
            if cv2.waitKey(1) & 0xFF == 27: # Esc para sair
                break

        if len(self.points_camera) == 4:
            self.config.update_homography(self.points_camera, self.points_projector)
            print("Calibração concluída e salva!")
        
        cv2.destroyWindow(proj_win)
        cv2.destroyWindow(cam_win)

if __name__ == "__main__":
    # Teste isolado
    cfg = Config()
    cap = cv2.VideoCapture(cfg.camera_id)
    calib = Calibration(cfg)
    calib.calibrate(cap)
    cap.release()
    cv2.destroyAllWindows()
