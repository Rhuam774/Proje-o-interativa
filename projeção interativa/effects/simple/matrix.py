import cv2
import numpy as np
import math
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_fps import AdaptiveFPSController

class MatrixEffect:
    def __init__(self):
        """Inicializa o efeito Data Rain"""
        self.name = "Data Rain"
        self.category = "simple"
        self.description = "Digital Rain escorrendo pelo caminho"
        self.effect_id = "effect_matrix"
        
        # Controlador de FPS adaptativo
        self._fps_controller = AdaptiveFPSController()
    
    def apply(self, frame, trail_canvas, cam_trail_canvas, H, yolo_enabled, trackers, trackers_lock, detection_data):
        """Aplica o efeito Matrix ao frame atual"""
        d = detection_data
        tid = d["id"]
        x, y, w, h = d["box"]
        cx = d.get("true_cx", x + w // 2)
        cy = d.get("true_cy", y + h // 2)
        
        # Obter histórico do rastreador
        with trackers_lock:
            if tid in trackers and "history" in trackers[tid]:
                hist = trackers[tid]["history"]
            else:
                hist = []
        
        if len(hist) >= 2:
            point_count = len(hist)
            
            # Sistema de FPS adaptativo baseado no número de pontos
            if not self._fps_controller.should_process_frame(point_count, trail_canvas):
                return  # Pula frame - mantém canvas
            
            base_thickness = max(3, int(w / 8))
            
            # Coordenadas atuais
            pt1_cam = np.float32([[[hist[-2][0], hist[-2][1]]]])
            pt2_cam = np.float32([[[cx, cy]]])
            pt1_proj = cv2.perspectiveTransform(pt1_cam, H)
            pt2_proj = cv2.perspectiveTransform(pt2_cam, H)
            p1 = (int(pt1_proj[0][0][0]), int(pt1_proj[0][0][1]))
            p2 = (int(pt2_proj[0][0][0]), int(pt2_proj[0][0][1]))
            pc1 = (int(hist[-2][0]), int(hist[-2][1]))
            pc2 = (cx, cy)
            
            if random.random() > 0.3:
                drop_len = random.randint(20, 80)
                rx = random.randint(-int(base_thickness), int(base_thickness))
                start_p = (p2[0] + rx, p2[1])
                end_p = (p2[0] + rx, p2[1] + drop_len)
                cv2.line(trail_canvas, start_p, end_p, (0, 255, 0), random.randint(1, 4))
                if yolo_enabled:
                    start_c = (pc2[0] + rx, pc2[1])
                    end_c = (pc2[0] + rx, pc2[1] + drop_len)
                    cv2.line(cam_trail_canvas, start_c, end_c, (0, 255, 0), random.randint(1, 4))
    
    def get_config_ui(self, parent_frame, save_callback):
        """Este efeito não tem configurações personalizáveis"""
        return None
    
    def cleanup(self):
        """Limpeza de recursos (não necessário para este efeito)"""
        pass
