import cv2
import numpy as np
import screeninfo

class Renderer:
    def __init__(self, config):
        self.config = config
        self.screen_idx = self.config.projector_screen_index
        monitors = screeninfo.get_monitors()
        if self.screen_idx < len(monitors):
            self.screen = monitors[self.screen_idx]
        else:
            self.screen = monitors[0]
            
        self.w = self.screen.width
        self.h = self.screen.height
        self.win_name = "Projecao (Simulada)" if self.config.simulation_mode else "Projecao Real"
        
        if not self.config.simulation_mode:
            cv2.namedWindow(self.win_name, cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow(self.win_name, self.screen.x, self.screen.y)
            cv2.setWindowProperty(self.win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

    def project(self, detections, camera_frame=None):
        if self.config.simulation_mode and camera_frame is not None:
            # No modo simulação, o "fundo" é a própria câmera
            canvas = cv2.resize(camera_frame, (self.w, self.h))
            # Efeito de escurecer um pouco para o projetor "brilhar"
            canvas = cv2.addWeighted(canvas, 0.4, np.zeros_like(canvas), 0, 0)
        else:
            # Tela preta de fundo (modo real)
            canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
        if self.config.homography_matrix is not None or self.config.simulation_mode:
            H = self.config.homography_matrix
            for d in detections:
                bx, by, bw, bh = d["box"]
                cx, cy = bx + bw//2, by + bh//2
                
                if self.config.simulation_mode and H is None:
                    # Se simulação sem calibração, usa coordenadas diretas (proporcional)
                    h_cam, w_cam = camera_frame.shape[:2]
                    px = int(cx * self.w / w_cam)
                    py = int(cy * self.h / h_cam)
                elif H is not None:
                    pts = np.array([[[cx, cy]]], dtype=np.float32)
                    proj_pts = cv2.perspectiveTransform(pts, H)
                    px, py = int(proj_pts[0][0][0]), int(proj_pts[0][0][1])
                else:
                    continue

                if 0 <= px < self.w and 0 <= py < self.h:
                    color = d.get("color", (255, 255, 255))
                    # Adicionar brilho (efeito de luz)
                    overlay = canvas.copy()
                    cv2.line(overlay, (px-30, py-30), (px+30, py+30), color, 4)
                    cv2.line(overlay, (px-30, py+30), (px+30, py-30), color, 4)
                    cv2.circle(overlay, (px, py), 40, color, 2)
                    label = d.get("label", "")
                    cv2.putText(overlay, label, (px+45, py+10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Blend para parecer luz projetada
                    cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)

        cv2.imshow(self.win_name, canvas)
        if cv2.waitKey(1) == ord('q'):
            return False
        return True

    def draw_wizard(self, pt):
        # Tela preta de fundo
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
        # Desenhar cruz gigante (Cor vermelha para contraste)
        color = (0, 0, 255) # Vermelho (BGR)
        size = 100
        cv2.line(canvas, (pt[0]-size, pt[1]), (pt[0]+size, pt[1]), color, 10)
        cv2.line(canvas, (pt[0], pt[1]-size), (pt[0], pt[1]+size), color, 10)
        
        # Círculo de alvo
        cv2.circle(canvas, pt, size//2, color, 2)
        
        # Texto de instrução
        cv2.putText(canvas, "COLOQUE O MARCADOR AQUI", (pt[0]-150, pt[1]+size+40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow(self.win_name, canvas)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyWindow(self.win_name)
