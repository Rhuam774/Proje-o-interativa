import cv2
import numpy as np
import math
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_fps import AdaptiveFPSController

class NebulaEffect:
    def __init__(self):
        """Inicializa o efeito Nebulosa Estelar"""
        self.name = "Poeira Cósmica"
        self.category = "simple"
        self.description = "Poeira estelar colorida e pontilhada suave"
        self.effect_id = "effect_nebula"
        
        # Controlador de FPS adaptativo
        self._fps_controller = AdaptiveFPSController()
    
    def apply(self, frame, trail_canvas, cam_trail_canvas, H, yolo_enabled, trackers, trackers_lock, detection_data):
        """Aplica o efeito Nebula ao frame atual"""
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
            
            num_stars = random.randint(10, 25)
            thick_neb = max(10, int(base_thickness * 2.5))
            
            for _ in range(num_stars):
                t = random.random()
                px = int(p1[0] * (1-t) + p2[0] * t)
                py = int(p1[1] * (1-t) + p2[1] * t)
                cpx = int(pc1[0] * (1-t) + pc2[0] * t)
                cpy = int(pc1[1] * (1-t) + pc2[1] * t)
                
                # Distribuição gaussiana para parecer concentração no centro
                ox = int(random.gauss(0, thick_neb / 2))
                oy = int(random.gauss(0, thick_neb / 2))
                
                col = (
                    min(255, d["color"][0] + random.randint(0, 100)),
                    min(255, d["color"][1] + random.randint(0, 80)),
                    min(255, d["color"][2] + random.randint(0, 100))
                )
                
                # Algumas estrelas maiores e brilhantes (núcleo branco)
                if random.random() > 0.9:
                    cv2.circle(trail_canvas, (px+ox, py+oy), random.randint(2, 4), (255, 255, 255), -1)
                    if yolo_enabled:
                        cv2.circle(cam_trail_canvas, (cpx+ox, cpy+oy), random.randint(2, 4), (255, 255, 255), -1)
                else:
                    cv2.circle(trail_canvas, (px+ox, py+oy), random.randint(1, 2), col, -1)
                    if yolo_enabled:
                        cv2.circle(cam_trail_canvas, (cpx+ox, cpy+oy), random.randint(1, 2), col, -1)
    
    def get_config_ui(self, parent_frame, save_callback):
        """Este efeito não tem configurações personalizáveis"""
        return None
    
    def cleanup(self):
        """Limpeza de recursos (não necessário para este efeito)"""
        pass
