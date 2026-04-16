import cv2
import numpy as np
import math
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_fps import AdaptiveFPSController

class LiquidEffect:
    def __init__(self):
        """Inicializa o efeito Fluido Orgânico"""
        self.name = "Fluido Orgânico"
        self.category = "simple"
        self.description = "Tinta densa com espalhamento radial orgânico (Metaballs feeling)"
        self.effect_id = "effect_liquid"
        
        # Controlador de FPS adaptativo
        self._fps_controller = AdaptiveFPSController()
    
    def apply(self, frame, trail_canvas, cam_trail_canvas, H, yolo_enabled, trackers, trackers_lock, detection_data):
        """Aplica o efeito Liquid ao frame atual"""
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
            
            num_splatters = random.randint(2, 6)
            for _ in range(num_splatters):
                dist_off = random.uniform(0, base_thickness * 1.5)
                ang_off = random.uniform(0, math.pi * 2)
                lx = int(p2[0] + dist_off * math.cos(ang_off))
                ly = int(p2[1] + dist_off * math.sin(ang_off))
                lcx = int(pc2[0] + dist_off * math.cos(ang_off))
                lcy = int(pc2[1] + dist_off * math.sin(ang_off))
                rad = random.randint(3, int(base_thickness * 0.8) + 5)
                color_l = (min(255, d["color"][0]+30), d["color"][1], max(0, d["color"][2]-30))
                cv2.circle(trail_canvas, (lx, ly), rad, color_l, -1)
                if yolo_enabled:
                    cv2.circle(cam_trail_canvas, (lcx, lcy), rad, color_l, -1)
    
    def get_config_ui(self, parent_frame, save_callback):
        """Este efeito não tem configurações personalizáveis"""
        return None
    
    def cleanup(self):
        """Limpeza de recursos (não necessário para este efeito)"""
        pass
