import cv2
import numpy as np
import math
import random

class CollidingBallsEffect:
    def __init__(self):
        """Inicializa o efeito Bolinhas Colidindo"""
        self.name = "Bolinhas Colidindo"
        self.category = "future"
        self.description = "Física de bolinhas com colisões realistas e brilho"
        self.effect_id = "effect_colliding_balls"
        
        # Estado das partículas
        self.particles = []
        self.max_particles = 80
        self.width = 640
        self.height = 480
        
        # Suavização de posição (smoothing)
        self.smoothed_cx = None
        self.smoothed_cy = None
        self.smoothing_factor = 0.3  # Fator de suavização (0.3 = 30% nova posição, 70% antiga)
        
        # Controlador de FPS adaptativo
        from adaptive_fps import AdaptiveFPSController
        self._fps_controller = AdaptiveFPSController()
    
    def apply(self, frame, trail_canvas, cam_trail_canvas, H, yolo_enabled, trackers, trackers_lock, detection_data, config=None):
        """Aplica o efeito Bolinhas Colidindo ao frame atual"""
        d = detection_data
        tid = d["id"]
        x, y, w, h = d["box"]
        cx = d.get("true_cx", x + w // 2)
        cy = d.get("true_cy", y + h // 2)
        
        # Obter histórico para calcular velocidade
        with trackers_lock:
            if tid in trackers and "history" in trackers[tid]:
                hist = trackers[tid]["history"]
            else:
                hist = []
        
        # Aplicar suavização à posição (amortecer vibração)
        if self.smoothed_cx is None:
            self.smoothed_cx = cx
            self.smoothed_cy = cy
        else:
            self.smoothed_cx = cx * self.smoothing_factor + self.smoothed_cx * (1 - self.smoothing_factor)
            self.smoothed_cy = cy * self.smoothing_factor + self.smoothed_cy * (1 - self.smoothing_factor)
        
        # Usar posição suavizada
        smooth_cx = int(self.smoothed_cx)
        smooth_cy = int(self.smoothed_cy)
        
        # Calcular velocidade do movimento
        vx = 0
        vy = 0
        if len(hist) >= 2:
            vx = smooth_cx - hist[-2][0]
            vy = smooth_cy - hist[-2][1]
        
        speed = math.hypot(vx, vy)
        
        # Sistema de FPS adaptativo
        total_entities = len(self.particles)
        if not self._fps_controller.should_process_frame(total_entities, trail_canvas):
            return
        
        # Adicionar novas partículas quando há movimento
        if speed > 2:
            for _ in range(min(3, int(speed))):
                if len(self.particles) < self.max_particles:
                    self.particles.append({
                        'x': smooth_cx,
                        'y': smooth_cy,
                        'vx': vx * 0.15 + (random.random() - 0.5) * 4,
                        'vy': vy * 0.15 + (random.random() - 0.5) * 2,
                        'life': 300,
                        'hue': random.randint(0, 360),
                        'size': random.uniform(4, 9)
                    })
        
        # Física básica - gravidade e movimento
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['vy'] += 0.4  # Gravidade
            p['x'] += p['vx']
            p['y'] += p['vy']
            
            # Colisão com bordas
            if p['x'] < p['size']:
                p['x'] = p['size']
                p['vx'] *= -0.8
            elif p['x'] > self.width - p['size']:
                p['x'] = self.width - p['size']
                p['vx'] *= -0.8
            
            if p['y'] < p['size']:
                p['y'] = p['size']
                p['vy'] *= -0.8
            elif p['y'] > self.height - p['size']:
                p['y'] = self.height - p['size']
                p['vy'] *= -0.7
                p['vx'] *= 0.95  # Atrito no chão
            
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.pop(i)
        
        # Física de colisão entre bolinhas
        for i in range(len(self.particles)):
            for j in range(i + 1, len(self.particles)):
                p1 = self.particles[i]
                p2 = self.particles[j]
                
                dx = p2['x'] - p1['x']
                dy = p2['y'] - p1['y']
                dist = math.hypot(dx, dy)
                min_dist = p1['size'] + p2['size']
                
                if dist < min_dist and dist > 0:
                    overlap = min_dist - dist
                    nx = dx / dist
                    ny = dy / dist
                    
                    # Separa as partículas
                    p1['x'] -= nx * overlap * 0.5
                    p1['y'] -= ny * overlap * 0.5
                    p2['x'] += nx * overlap * 0.5
                    p2['y'] += ny * overlap * 0.5
                    
                    # Transferência de energia (quique)
                    kx = p1['vx'] - p2['vx']
                    ky = p1['vy'] - p2['vy']
                    p = 2 * (nx * kx + ny * ky) / 2
                    
                    if p > 0:
                        p1['vx'] -= p * nx * 0.8
                        p1['vy'] -= p * ny * 0.8
                        p2['vx'] += p * nx * 0.8
                        p2['vy'] += p * ny * 0.8
        
        # Renderização com brilho
        for p in self.particles:
            alpha = min(1.0, p['life'] / 30)
            
            # Converter HSL para BGR
            color = self.hsl_to_bgr(p['hue'], 0.9, 0.5)
            color = [int(c * alpha) for c in color]
            
            # Desenhar bola principal
            cv2.circle(trail_canvas, (int(p['x']), int(p['y'])), int(p['size']), tuple(color), -1)
            
            # Desenhar brilho (highlight)
            highlight_color = [int(255 * alpha), int(255 * alpha), int(255 * alpha)]
            highlight_x = int(p['x'] - p['size'] * 0.3)
            highlight_y = int(p['y'] - p['size'] * 0.3)
            highlight_size = int(p['size'] * 0.25)
            cv2.circle(trail_canvas, (highlight_x, highlight_y), highlight_size, tuple(highlight_color), -1)
            
            if yolo_enabled:
                cv2.circle(cam_trail_canvas, (int(p['x']), int(p['y'])), int(p['size']), tuple(color), -1)
                cv2.circle(cam_trail_canvas, (highlight_x, highlight_y), highlight_size, tuple(highlight_color), -1)
    
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
        self.particles = []
