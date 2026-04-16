import cv2
import numpy as np
import math
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_fps import AdaptiveFPSController

class PlasmaEffect:
    def __init__(self):
        """Inicializa o efeito Raio Plasmático"""
        self.name = "Energia Pura"
        self.category = "simple"
        self.description = "Descargas elétricas caóticas ligando os pontos"
        self.effect_id = "effect_plasma"
        
        # Controlador de FPS adaptativo
        self._fps_controller = AdaptiveFPSController()
    
    def apply(self, frame, trail_canvas, cam_trail_canvas, H, yolo_enabled, trackers, trackers_lock, detection_data):
        """Aplica o efeito Plasma ao frame atual"""
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
            
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            
            if dist > 5:
                for _ in range(3):  # Gera múltiplas ramificações
                    pt_ant_p = p1
                    pt_ant_c = pc1
                    segments = int(dist // 5) + 2
                    
                    plasma_color = (
                        min(255, d["color"][0] + 150),
                        min(255, d["color"][1] + 100),
                        min(255, d["color"][2] + 50)
                    )
                    
                    for s in range(1, segments + 1):
                        t = s / segments
                        jitter = int(base_thickness * 1.5)
                        
                        if s == segments:
                            px, py = p2[0], p2[1]
                            cpx, cpy = pc2[0], pc2[1]
                        else:
                            px = int(p1[0] * (1-t) + p2[0] * t) + random.randint(-jitter, jitter)
                            py = int(p1[1] * (1-t) + p2[1] * t) + random.randint(-jitter, jitter)
                            cpx = int(pc1[0] * (1-t) + pc2[0] * t) + random.randint(-jitter, jitter)
                            cpy = int(pc1[1] * (1-t) + pc2[1] * t) + random.randint(-jitter, jitter)
                        
                        thick_plasma = random.randint(1, max(3, int(base_thickness//2)))
                        cv2.line(trail_canvas, pt_ant_p, (px, py), plasma_color, thick_plasma)
                        if yolo_enabled:
                            cv2.line(cam_trail_canvas, pt_ant_c, (cpx, cpy), plasma_color, thick_plasma)
                        
                        # Brilho central fino (raios brancos)
                        if random.random() > 0.5:
                            cv2.line(trail_canvas, pt_ant_p, (px, py), (255, 255, 255), 1)
                            if yolo_enabled:
                                cv2.line(cam_trail_canvas, pt_ant_c, (cpx, cpy), (255, 255, 255), 1)
                        
                        pt_ant_p = (px, py)
                        pt_ant_c = (cpx, cpy)
    
    def get_config_ui(self, parent_frame, save_callback):
        """Este efeito não tem configurações personalizáveis"""
        return None
    
    def cleanup(self):
        """Limpeza de recursos (não necessário para este efeito)"""
        pass
