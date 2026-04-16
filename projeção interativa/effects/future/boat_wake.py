import cv2
import numpy as np
import math
import random

class BoatWakeEffect:
    def __init__(self):
        """Inicializa o efeito Rastro de Barco"""
        self.name = "Rastro de Barco"
        self.category = "future"
        self.description = "Rastro vetorial realista de barco com espuma e esteira"
        self.effect_id = "effect_boat_wake"
        
        # Estado do rastro
        self.ripples = []
        self.particles = []
        self.max_ripples = 80
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
        """Aplica o efeito Rastro de Barco ao frame atual"""
        d = detection_data
        tid = d["id"]
        x, y, w, h = d["box"]
        cx = d.get("true_cx", x + w // 2)
        cy = d.get("true_cy", y + h // 2)
        
        # Calcular velocidade e ângulo
        vx = 0
        vy = 0
        angle = 0
        
        with trackers_lock:
            if tid in trackers and "history" in trackers[tid]:
                hist = trackers[tid]["history"]
                if len(hist) >= 2:
                    vx = cx - hist[-2][0]
                    vy = cy - hist[-2][1]
                    angle = math.atan2(vy, vx)
        
        speed = math.hypot(vx, vy)
        
        # Sistema de FPS adaptativo
        total_entities = len(self.ripples) + len(self.particles)
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
        
        # Fundo azul escuro do mar
        trail_canvas[:] = (5, 16, 32)
        if yolo_enabled:
            cam_trail_canvas[:] = (5, 16, 32)
        
        # Adicionar novos ripples quando há movimento
        if speed > 1:
            self.ripples.insert(0, {
                'x': smooth_cx,
                'y': smooth_cy,
                'angle': angle,
                'spread': 2,
                'life': 1.0
            })
            if len(self.ripples) > self.max_ripples:
                self.ripples.pop()
        
        # Atualizar ripples (expandir e desaparecer)
        for r in self.ripples:
            r['spread'] += 0.8
            r['life'] -= 0.012
        
        # Remover ripples mortos
        self.ripples = [r for r in self.ripples if r['life'] > 0]
        
        # Desenhar rastro em formato V (lados do barco) - fiel ao HTML
        for i in range(len(self.ripples) - 1):
            n1 = self.ripples[i]
            n2 = self.ripples[i + 1]
            
            # Lado esquerdo do barco
            l1x = n1['x'] + math.cos(n1['angle'] - math.pi / 2.2) * n1['spread']
            l1y = n1['y'] + math.sin(n1['angle'] - math.pi / 2.2) * n1['spread']
            l2x = n2['x'] + math.cos(n2['angle'] - math.pi / 2.2) * n2['spread']
            l2y = n2['y'] + math.sin(n2['angle'] - math.pi / 2.2) * n2['spread']
            
            alpha = int(n1['life'] * 204)  # n1.life * 0.8 * 255
            color = (int(180 * alpha / 255), int(230 * alpha / 255), int(255 * alpha / 255))
            thickness = int(1.5 + (1 - n1['life']) * 2)
            
            # Simular lineCap round com círculos nas pontas
            cv2.line(trail_canvas, (int(l1x), int(l1y)), (int(l2x), int(l2y)), color, thickness)
            cv2.circle(trail_canvas, (int(l1x), int(l1y)), thickness // 2, color, -1)
            cv2.circle(trail_canvas, (int(l2x), int(l2y)), thickness // 2, color, -1)
            
            if yolo_enabled:
                cv2.line(cam_trail_canvas, (int(l1x), int(l1y)), (int(l2x), int(l2y)), color, thickness)
                cv2.circle(cam_trail_canvas, (int(l1x), int(l1y)), thickness // 2, color, -1)
                cv2.circle(cam_trail_canvas, (int(l2x), int(l2y)), thickness // 2, color, -1)
            
            # Lado direito do barco
            r1x = n1['x'] + math.cos(n1['angle'] + math.pi / 2.2) * n1['spread']
            r1y = n1['y'] + math.sin(n1['angle'] + math.pi / 2.2) * n1['spread']
            r2x = n2['x'] + math.cos(n2['angle'] + math.pi / 2.2) * n2['spread']
            r2y = n2['y'] + math.sin(n2['angle'] + math.pi / 2.2) * n2['spread']
            
            cv2.line(trail_canvas, (int(r1x), int(r1y)), (int(r2x), int(r2y)), color, thickness)
            cv2.circle(trail_canvas, (int(r1x), int(r1y)), thickness // 2, color, -1)
            cv2.circle(trail_canvas, (int(r2x), int(r2y)), thickness // 2, color, -1)
            
            if yolo_enabled:
                cv2.line(cam_trail_canvas, (int(r1x), int(r1y)), (int(r2x), int(r2y)), color, thickness)
                cv2.circle(cam_trail_canvas, (int(r1x), int(r1y)), thickness // 2, color, -1)
                cv2.circle(cam_trail_canvas, (int(r2x), int(r2y)), thickness // 2, color, -1)
        
        # Adicionar partículas de espuma na esteira
        if speed > 2 and random.random() > 0.4:
            self.particles.append({
                'x': smooth_cx + (random.random() - 0.5) * 12,
                'y': smooth_cy + (random.random() - 0.5) * 12,
                'vx': -vx * 0.05,
                'vy': -vy * 0.05,
                'life': 1.0,
                'size': random.uniform(0.5, 2.5)
            })
        
        # Atualizar e desenhar partículas de espuma com modo 'lighter'
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 0.02
            
            if p['life'] <= 0:
                self.particles.pop(i)
            else:
                alpha = int(p['life'] * 255)
                # Simular modo 'lighter' com add
                color = (alpha, alpha, alpha)
                cv2.circle(trail_canvas, (int(p['x']), int(p['y'])), int(p['size']), color, -1)
                if yolo_enabled:
                    cv2.circle(cam_trail_canvas, (int(p['x']), int(p['y'])), int(p['size']), color, -1)
        
        # Desenhar barco minimalista no cursor
        self.draw_boat(trail_canvas, smooth_cx, smooth_cy, angle)
        if yolo_enabled:
            self.draw_boat(cam_trail_canvas, smooth_cx, smooth_cy, angle)
    
    def draw_boat(self, canvas, cx, cy, angle):
        """Desenha o barco minimalista"""
        # Criar pontos do barco
        boat_points = np.array([
            [10, 0],      # Proa
            [-6, 4],      # Popa direita
            [-4, 0],      # Recuo do motor
            [-6, -4]      # Popa esquerda
        ], dtype=np.float32)
        
        # Rotacionar pontos
        rotation_matrix = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)]
        ])
        
        rotated_points = np.dot(boat_points, rotation_matrix.T)
        rotated_points[:, 0] += cx
        rotated_points[:, 1] += cy
        
        # Desenhar barco
        cv2.fillPoly(canvas, [rotated_points.astype(np.int32)], (255, 255, 255))
    
    def get_config_ui(self, parent_frame, save_callback):
        """Este efeito não tem configurações personalizáveis"""
        return None
    
    def cleanup(self):
        """Limpeza de recursos"""
        self.ripples = []
        self.particles = []
