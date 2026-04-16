import cv2
import numpy as np
import math
import random

class NeuralNetworkEffect:
    def __init__(self):
        """Inicializa o efeito Rede Neural"""
        self.name = "Rede Neural"
        self.category = "future"
        self.description = "Partículas conectadas formando rede neural interativa"
        self.effect_id = "effect_neural_network"
        
        # Partículas da rede
        self.particles = []
        self.num_particles = 300
        self.width = 640
        self.height = 480
        self.connection_distance = 70
        self.mouse_influence_distance = 100
        
        # Suavização de posição (smoothing)
        self.smoothed_cx = None
        self.smoothed_cy = None
        self.smoothing_factor = 0.3
        
        # Controlador de FPS adaptativo
        from adaptive_fps import AdaptiveFPSController
        self._fps_controller = AdaptiveFPSController()
    
    def apply(self, frame, trail_canvas, cam_trail_canvas, H, yolo_enabled, trackers, trackers_lock, detection_data, config=None):
        """Aplica o efeito Rede Neural ao frame atual"""
        d = detection_data
        tid = d["id"]
        x, y, w, h = d["box"]
        cx = d.get("true_cx", x + w // 2)
        cy = d.get("true_cy", y + h // 2)
        
        # Sistema de FPS adaptativo
        total_entities = len(self.particles)
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
        
        # Inicializar partículas se necessário
        if len(self.particles) == 0:
            for i in range(self.num_particles):
                self.particles.append({
                    'x': random.uniform(0, self.width),
                    'y': random.uniform(0, self.height),
                    'vx': random.uniform(-0.75, 0.75),
                    'vy': random.uniform(-0.75, 0.75)
                })
        
        # Atualizar partículas
        for p in self.particles:
            # Calcular distância até o mouse
            dist = math.hypot(p['x'] - smooth_cx, p['y'] - smooth_cy)
            
            # Força de repulsão do mouse
            if dist < self.mouse_influence_distance:
                force = (self.mouse_influence_distance - dist) / self.mouse_influence_distance
                p['vx'] += ((p['x'] - smooth_cx) / dist) * force * 0.5
                p['vy'] += ((p['y'] - smooth_cy) / dist) * force * 0.5
            
            # Atrito
            p['vx'] *= 0.95
            p['vy'] *= 0.95
            
            # Movimento aleatório
            p['x'] += p['vx'] + (random.random() - 0.5) * 0.5
            p['y'] += p['vy'] + (random.random() - 0.5) * 0.5
            
            # Wrap-around nas bordas
            if p['x'] < 0:
                p['x'] = self.width
            elif p['x'] > self.width:
                p['x'] = 0
            if p['y'] < 0:
                p['y'] = self.height
            elif p['y'] > self.height:
                p['y'] = 0
        
        # Desenhar partículas
        for p in self.particles:
            cv2.circle(trail_canvas, (int(p['x']), int(p['y'])), 1, (255, 255, 255), -1)
            if yolo_enabled:
                cv2.circle(cam_trail_canvas, (int(p['x']), int(p['y'])), 1, (255, 255, 255), -1)
        
        # Desenhar conexões entre partículas próximas
        for i in range(len(self.particles)):
            for j in range(i + 1, len(self.particles)):
                p1 = self.particles[i]
                p2 = self.particles[j]
                
                dist = math.hypot(p1['x'] - p2['x'], p1['y'] - p2['y'])
                
                if dist < self.connection_distance:
                    # Calcular transparência baseada na distância
                    alpha = int((0.8 - dist / self.connection_distance) * 255)
                    color = (100, 200, 255)
                    
                    cv2.line(trail_canvas, 
                            (int(p1['x']), int(p1['y'])), 
                            (int(p2['x']), int(p2['y'])), 
                            color, 1)
                    
                    if yolo_enabled:
                        cv2.line(cam_trail_canvas, 
                                (int(p1['x']), int(p1['y'])), 
                                (int(p2['x']), int(p2['y'])), 
                                color, 1)
    
    def get_config_ui(self, parent_frame, save_callback):
        """Este efeito não tem configurações personalizáveis"""
        return None
    
    def cleanup(self):
        """Limpeza de recursos"""
        self.particles = []
