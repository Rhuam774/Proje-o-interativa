import cv2
import numpy as np
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_fps import AdaptiveFPSController

class PlexusEffect:
    def __init__(self):
        """Inicializa o efeito Plexus Neural"""
        self.name = "Plexus Neural"
        self.category = "simple"
        self.description = "Gera polígonos entre pontos distantes mas compatíveis"
        self.effect_id = "effect_plexus"
        
        # Controlador de FPS adaptativo
        self._fps_controller = AdaptiveFPSController()
    
    def apply(self, frame, trail_canvas, cam_trail_canvas, H, yolo_enabled, trackers, trackers_lock, detection_data):
        """
        Aplica o efeito Plexus ao frame atual
        
        Parâmetros:
        - frame: Frame atual da câmera (numpy array)
        - trail_canvas: Canvas do projetor para desenhar (numpy array)
        - cam_trail_canvas: Canvas do monitor para desenhar (numpy array)
        - H: Matriz de homografia para transformação de perspectiva
        - yolo_enabled: Boolean indicando se visão IA está ativa
        - trackers: Dicionário de rastreadores ativos
        - trackers_lock: Lock para acesso seguro ao dicionário de rastreadores
        - detection_data: Dados da detecção atual (dicionário)
        """
        d = detection_data
        tid = d["id"]
        
        # Obter histórico do rastreador
        with trackers_lock:
            if tid in trackers and "history" in trackers[tid]:
                hist = trackers[tid]["history"]
            else:
                hist = []
        
        if len(hist) > 4:
            recent_hist = hist[-40:]  # Até 40 pontos na memória
            point_count = len(recent_hist)
            
            # Sistema de FPS adaptativo baseado no número de pontos
            if not self._fps_controller.should_process_frame(point_count, trail_canvas):
                return  # Pula frame - mantém canvas
            
            pts_cam = np.array([[h[0], h[1]] for h in recent_hist], dtype=np.float32).reshape(-1, 1, 2)
            pts_proj = cv2.perspectiveTransform(pts_cam, H)
            
            pts_p = [(int(p[0][0]), int(p[0][1])) for p in pts_proj]
            pts_c = [(int(p[0][0]), int(p[0][1])) for p in pts_cam]

            thresh = 100  # Conecta-se se a distância for menor que 100px
            for i in range(max(0, len(pts_p)-15), len(pts_p)):
                for j in range(max(0, i-6), i):
                    dist_pts = math.hypot(pts_p[i][0] - pts_p[j][0], pts_p[i][1] - pts_p[j][1])
                    if dist_pts < thresh:
                        t = 1 if dist_pts > thresh * 0.5 else 2
                        cv2.line(trail_canvas, pts_p[i], pts_p[j], d["color"], t)
                        if yolo_enabled:
                            cv2.line(cam_trail_canvas, pts_c[i], pts_c[j], d["color"], t)
                        
                        # Se muito perto, desenha pequenas facetas/triangulos
                        if dist_pts < thresh * 0.3 and i > 2:
                            k = i - 2
                            pts_tri_proj = np.array([pts_p[i], pts_p[j], pts_p[k]], np.int32)
                            cv2.fillPoly(trail_canvas, [pts_tri_proj], (d["color"][0]//2, d["color"][1]//2, d["color"][2]//2))
                            if yolo_enabled:
                                pts_tri_cam = np.array([pts_c[i], pts_c[j], pts_c[k]], np.int32)
                                cv2.fillPoly(cam_trail_canvas, [pts_tri_cam], (d["color"][0]//2, d["color"][1]//2, d["color"][2]//2))
    
    def get_config_ui(self, parent_frame, save_callback):
        """Este efeito não tem configurações personalizáveis"""
        return None
    
    def cleanup(self):
        """Limpeza de recursos (não necessário para este efeito)"""
        pass
