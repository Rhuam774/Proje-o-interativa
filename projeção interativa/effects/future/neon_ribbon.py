import cv2
import numpy as np
import math
import random
import time

class NeonRibbonEffect:
    def __init__(self):
        """Inicializa o efeito Fita Neon Profissional"""
        self.name = "Fita Neon Profissional"
        self.category = "future"
        self.description = "Fita neon suave que segue o movimento com brilho intenso"
        self.effect_id = "effect_neon_ribbon"
        
        # Pontos da fita
        self.points = []
        self.num_points = 60
        self.width = 640
        self.height = 480
        
        # Suavização de posição (smoothing)
        self.smoothed_cx = None
        self.smoothed_cy = None
        self.smoothing_factor = 0.3
        
        # Controlador de FPS adaptativo
        from adaptive_fps import AdaptiveFPSController
        self._fps_controller = AdaptiveFPSController()
    
    def apply(self, frame, trail_canvas, cam_trail_canvas, H, yolo_enabled, trackers, trackers_lock, detection_data, config=None):
        """Aplica o efeito Fita Neon ao frame atual"""
        d = detection_data
        tid = d["id"]
        x, y, w, h = d["box"]
        cx = d.get("true_cx", x + w // 2)
        cy = d.get("true_cy", y + h // 2)
        
        # Sistema de FPS adaptativo
        total_entities = len(self.points)
        if not self._fps_controller.should_process_frame(total_entities, trail_canvas):
            return
        
        # Aplicar suavização à posição (amortecer vibração)
        if self.smoothed_cx is None:
            self.smoothed_cx = cx
            self.smoothed_cy = cy
        else:
            self.smoothed_cx = cx * self.smoothing_factor + self.smoothed_cx * (1 - self.smoothing_factor)
            self.smoothed_cy = cy * self.smoothing_factor + self.smoothed_cy * (1 - self.smoothing_factor)
        
        smooth_cx = int(self.smoothed_cx)
        smooth_cy = int(self.smoothed_cy)
        
        # Inicializar pontos se necessário
        if len(self.points) == 0:
            for i in range(self.num_points):
                self.points.append({'x': smooth_cx, 'y': smooth_cy})
        
        # Atualizar primeiro ponto (segue o mouse)
        self.points[0]['x'] += (smooth_cx - self.points[0]['x']) * 0.4
        self.points[0]['y'] += (smooth_cy - self.points[0]['y']) * 0.4
        
        # Atualizar pontos seguintes (seguem o ponto anterior)
        for i in range(1, len(self.points)):
            self.points[i]['x'] += (self.points[i-1]['x'] - self.points[i]['x']) * 0.4
            self.points[i]['y'] += (self.points[i-1]['y'] - self.points[i]['y']) * 0.4
        
        # Desenhar fita neon com curvas suaves
        if len(self.points) >= 3:
            # Cor neon cíclica
            hue = int((time.time() * 100) % 360)
            color = self.hsl_to_bgr(hue, 1.0, 0.6)
            color_white = (255, 255, 255)
            
            # Desenhar no trail_canvas
            self.draw_neon_ribbon(trail_canvas, self.points, color, color_white)
            
            if yolo_enabled:
                self.draw_neon_ribbon(cam_trail_canvas, self.points, color, color_white)
    
    def draw_neon_ribbon(self, canvas, points, color, color_white):
        """Desenha a fita neon com curvas suaves e brilho (fiel ao HTML)"""
        if len(points) < 3:
            return
        
        # Converter pontos para array numpy
        pts = np.array([[p['x'], p['y']] for p in points], dtype=np.int32)
        
        # Desenhar curva usando quadraticCurveTo como no HTML
        curve_points = []
        curve_points.append(pts[0])
        
        for i in range(1, len(pts) - 1):
            # Ponto médio entre ponto atual e próximo (como no HTML)
            xc = (pts[i][0] + pts[i+1][0]) // 2
            yc = (pts[i][1] + pts[i+1][1]) // 2
            
            # Adicionar ponto de controle e ponto médio
            # quadraticCurveTo(particles[i].x, particles[i].y, xc, yc)
            # Simulamos isso com mais pontos na curva
            for t in np.linspace(0, 1, 5):
                # Interpolação quadrática
                x = (1-t)**2 * pts[i-1][0] + 2*(1-t)*t * pts[i][0] + t**2 * xc
                y = (1-t)**2 * pts[i-1][1] + 2*(1-t)*t * pts[i][1] + t**2 * yc
                curve_points.append([int(x), int(y)])
        
        curve_points.append(pts[-1])
        curve_points = np.array(curve_points, dtype=np.int32)
        
        # Desenhar linha externa com brilho (shadowBlur simulado com múltiplas passagens)
        thickness = 14
        # Simular shadowBlur com linhas mais finas e transparentes
        for blur_offset in range(3):
            alpha = int(100 - blur_offset * 30)
            blur_color = [int(c * alpha / 255) for c in color]
            cv2.polylines(canvas, [curve_points], False, blur_color, thickness + (3-blur_offset)*2)
        
        # Linha principal
        cv2.polylines(canvas, [curve_points], False, color, thickness)
        
        # Desenhar linha interna branca
        thickness_inner = 4
        cv2.polylines(canvas, [curve_points], False, color_white, thickness_inner)
    
    def hsl_to_bgr(self, h, s, l):
        """Converte HSL para BGR"""
        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l - c / 2
        
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return [(r + m) * 255, (g + m) * 255, (b + m) * 255]
    
    def get_config_ui(self, parent_frame, save_callback):
        """Este efeito não tem configurações personalizáveis"""
        return None
    
    def cleanup(self):
        """Limpeza de recursos"""
        self.points = []
