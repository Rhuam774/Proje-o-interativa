import cv2
import numpy as np
import math
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_fps import AdaptiveFPSController

class HologramEffect:
    def __init__(self):
        """Inicializa o efeito Varredura Laser"""
        self.name = "Varredura Laser"
        self.category = "simple"
        self.description = "Linhas de grade CRT/Laser horizontais dinâmicas"
        self.effect_id = "effect_hologram"
        
        # Controlador de FPS adaptativo
        self._fps_controller = AdaptiveFPSController()
    
    def apply(self, frame, trail_canvas, cam_trail_canvas, H, yolo_enabled, trackers, trackers_lock, detection_data):
        """Aplica o efeito Hologram ao frame atual"""
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
            
            if dist > 1:
                scan_w = max(50, base_thickness * 3)
                for y_off in range(-int(base_thickness), int(base_thickness), 8):
                    if random.random() > 0.5:
                        color_h = (255, 255, 0) if random.random() > 0.5 else d["color"]
                        pt1_p = (int(p2[0] - scan_w/2), p2[1] + y_off)
                        pt2_p = (int(p2[0] + scan_w/2), p2[1] + y_off)
                        cv2.line(trail_canvas, pt1_p, pt2_p, color_h, 1)
                        if yolo_enabled:
                            pt1_c = (int(pc2[0] - scan_w/2), pc2[1] + y_off)
                            pt2_c = (int(pc2[0] + scan_w/2), pc2[1] + y_off)
                            cv2.line(cam_trail_canvas, pt1_c, pt2_c, color_h, 1)
    
    def get_config_ui(self, parent_frame, save_callback):
        """Este efeito não tem configurações personalizáveis"""
        return None
    
    def cleanup(self):
        """Limpeza de recursos (não necessário para este efeito)"""
        pass
