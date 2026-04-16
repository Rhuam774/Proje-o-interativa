import cv2
import numpy as np
import math
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_fps import AdaptiveFPSController

class PulseEffect:
    def __init__(self):
        """Inicializa o efeito Pulsos de Radar"""
        self.name = "Pulsos de Radar"
        self.category = "simple"
        self.description = "Desloca ecos circulares/elípticos em cascata"
        self.effect_id = "effect_pulse"
        
        # Controlador de FPS adaptativo
        self._fps_controller = AdaptiveFPSController()
    
    def apply(self, frame, trail_canvas, cam_trail_canvas, H, yolo_enabled, trackers, trackers_lock, detection_data):
        """Aplica o efeito Pulse ao frame atual"""
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
            
            # Calcular distância e ângulo
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            
            if dist > 3 or random.random() > 0.8:
                rc = int(base_thickness * random.uniform(1.0, 3.5))
                # Cria bordas grossas internas
                cv2.circle(trail_canvas, p2, int(rc*0.4), d["color"], -1)
                if yolo_enabled:
                    cv2.circle(cam_trail_canvas, pc2, int(rc*0.4), d["color"], -1)
                
                # Bordas vazadas externas como ecos sonoros (cyan/magenta shifting)
                shift_color = (min(255, d["color"][0]+100), d["color"][1], min(255, d["color"][2]+100))
                cv2.circle(trail_canvas, p2, rc, shift_color, int(max(1, 4 - rc/10)))
                if yolo_enabled:
                    cv2.circle(cam_trail_canvas, pc2, rc, shift_color, int(max(1, 4 - rc/10)))
                
                if random.random() > 0.5:
                    cv2.ellipse(trail_canvas, p2, (rc + 10, int(rc*0.3)), math.degrees(angle), 0, 360, (255, 255, 255), 1)
                    if yolo_enabled:
                        cv2.ellipse(cam_trail_canvas, pc2, (rc + 10, int(rc*0.3)), math.degrees(angle), 0, 360, (255, 255, 255), 1)
    
    def get_config_ui(self, parent_frame, save_callback):
        """Este efeito não tem configurações personalizáveis"""
        return None
    
    def cleanup(self):
        """Limpeza de recursos (não necessário para este efeito)"""
        pass
