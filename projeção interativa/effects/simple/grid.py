import cv2
import numpy as np
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_fps import AdaptiveFPSController

class GridEffect:
    def __init__(self):
        """Inicializa o efeito Malha Gravitacional"""
        self.name = "Malha Gravitacional"
        self.category = "simple"
        self.description = "Deforma uma geometria de rede baseada no eixo de movimento"
        self.effect_id = "effect_grid"
        
        # Controlador de FPS adaptativo
        self._fps_controller = AdaptiveFPSController()
    
    def apply(self, frame, trail_canvas, cam_trail_canvas, H, yolo_enabled, trackers, trackers_lock, detection_data):
        """Aplica o efeito Grid ao frame atual"""
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
            
            grid_size = int(max(15, base_thickness * 1.5))
            gx, gy = (p2[0] // grid_size) * grid_size, (p2[1] // grid_size) * grid_size
            cgx, cgy = (pc2[0] // grid_size) * grid_size, (pc2[1] // grid_size) * grid_size
            
            # Matriz de pontos 5x5 ao redor do alvo distorcida pela gravidade da caixa
            for idx_y in range(-2, 3):
                for idx_x in range(-2, 3):
                    # Coords na projeção
                    nx, ny = gx + idx_x * grid_size, gy + idx_y * grid_size
                    dist_g = math.hypot(p2[0] - nx, p2[1] - ny)
                    
                    warp = max(0.0, 1.0 - dist_g / (grid_size * 2))
                    wx = int(nx + (p2[0] - nx) * warp * 0.7)
                    wy = int(ny + (p2[1] - ny) * warp * 0.7)
                    
                    cv2.circle(trail_canvas, (wx, wy), 2, (255, 255, 255), -1)
                    cv2.line(trail_canvas, (wx, wy), (p2[0], p2[1]), d["color"], 1)
                    
                    # Coords na câmera
                    cnx, cny = cgx + idx_x * grid_size, cgy + idx_y * grid_size
                    dist_cg = math.hypot(pc2[0] - cnx, pc2[1] - cny)
                    
                    cwarp = max(0.0, 1.0 - dist_cg / (grid_size * 2))
                    cwx = int(cnx + (pc2[0] - cnx) * cwarp * 0.7)
                    cwy = int(cny + (pc2[1] - cny) * cwarp * 0.7)
                    if yolo_enabled:
                        cv2.circle(cam_trail_canvas, (cwx, cwy), 2, (255, 255, 255), -1)
                        cv2.line(cam_trail_canvas, (cwx, cwy), (pc2[0], pc2[1]), d["color"], 1)
    
    def get_config_ui(self, parent_frame, save_callback):
        """Este efeito não tem configurações personalizáveis"""
        return None
    
    def cleanup(self):
        """Limpeza de recursos (não necessário para este efeito)"""
        pass
