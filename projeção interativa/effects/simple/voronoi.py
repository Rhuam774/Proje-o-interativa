import cv2
import numpy as np
import math
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_fps import AdaptiveFPSController

class VoronoiEffect:
    def __init__(self):
        """Inicializa o efeito Estilhaços Geométricos"""
        self.name = "Estilhaços Geométricos"
        self.category = "simple"
        self.description = "Polígonos irregulares vivos que estilhaçam"
        self.effect_id = "effect_voronoi"
        
        # Controlador de FPS adaptativo
        self._fps_controller = AdaptiveFPSController()
    
    def apply(self, frame, trail_canvas, cam_trail_canvas, H, yolo_enabled, trackers, trackers_lock, detection_data):
        """Aplica o efeito Voronoi ao frame atual"""
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
            
            if dist > 2:
                for _ in range(2):
                    pts_v_p = []
                    pts_v_c = []
                    cx_p, cy_p = p2[0], p2[1]
                    cx_c, cy_c = pc2[0], pc2[1]
                    for _ in range(3):  # Triangulo de estilhaço
                        r_off = random.randint(10, int(base_thickness * 2) + 20)
                        a_off = random.uniform(0, math.pi * 2)
                        pts_v_p.append([int(cx_p + r_off*math.cos(a_off)), int(cy_p + r_off*math.sin(a_off))])
                        pts_v_c.append([int(cx_c + r_off*math.cos(a_off)), int(cy_c + r_off*math.sin(a_off))])
                    
                    pts_v_p = np.array([pts_v_p], np.int32)
                    pts_v_c = np.array([pts_v_c], np.int32)
                    if random.random() > 0.5:
                        cv2.fillPoly(trail_canvas, pts_v_p, d["color"])
                        if yolo_enabled:
                            cv2.fillPoly(cam_trail_canvas, pts_v_c, d["color"])
                    else:
                        cv2.polylines(trail_canvas, pts_v_p, True, (255, 255, 255), 1)
                        if yolo_enabled:
                            cv2.polylines(cam_trail_canvas, pts_v_c, True, (255, 255, 255), 1)
    
    def get_config_ui(self, parent_frame, save_callback):
        """Este efeito não tem configurações personalizáveis"""
        return None
    
    def cleanup(self):
        """Limpeza de recursos (não necessário para este efeito)"""
        pass
