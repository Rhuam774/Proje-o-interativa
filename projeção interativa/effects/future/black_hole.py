import cv2
import numpy as np
import math
import random

class BlackHoleEffect:
    def __init__(self):
        """Inicializa o efeito Buraco Negro"""
        self.name = "Buraco Negro"
        self.category = "future"
        self.description = "Partículas orbitais sugadas para um buraco negro com brilho roxo"
        self.effect_id = "effect_black_hole"
        
        # Partículas orbitais
        self.particles = []
        self.num_particles = 600
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
        """Aplica o efeito Buraco Negro ao frame atual"""
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
            max_radius = max(self.width, self.height) / 2
            for i in range(self.num_particles):
                self.particles.append({
                    'angle': random.uniform(0, math.pi * 2),
                    'radius': random.uniform(0, max_radius),
                    'speed': random.uniform(0.005, 0.025),
                    'size': random.uniform(0.5, 2.0)
                })
        
        # Fundo com fade para criar rastro
        overlay = trail_canvas.copy()
        cv2.addWeighted(trail_canvas, 0.85, overlay, 0.15, 0, trail_canvas)
        if yolo_enabled:
            overlay_cam = cam_trail_canvas.copy()
            cv2.addWeighted(cam_trail_canvas, 0.85, overlay_cam, 0.15, 0, cam_trail_canvas)
        
        # Atualizar e desenhar partículas
        for p in self.particles:
            # Aumentar velocidade conforme se aproximam do buraco
            speed_mult = 1 + (1000 / (p['radius'] + 10))
            p['angle'] += p['speed'] * speed_mult * 0.05
            p['radius'] -= speed_mult * 0.5
            
            # Resetar partícula se for sugada
            if p['radius'] < 2:
                p['radius'] = max(self.width, self.height)
                p['angle'] = random.uniform(0, math.pi * 2)
            
            # Calcular posição
            px = smooth_cx + math.cos(p['angle']) * p['radius']
            py = smooth_cy + math.sin(p['angle']) * p['radius']
            
            # Calcular cor baseada na distância (roxo para azul)
            dist_factor = min(1.0, 100 / p['radius'])
            r = int(150 + p['radius'] * 0.2)
            g = 100
            b = 255
            alpha = int(min(1.0, dist_factor) * 255)
            
            color = (r, g, b)
            
            # Desenhar partícula
            cv2.circle(trail_canvas, (int(px), int(py)), int(p['size']), color, -1)
            if yolo_enabled:
                cv2.circle(cam_trail_canvas, (int(px), int(py)), int(p['size']), color, -1)
        
        # Desenhar buraco negro com brilho roxo
        # Criar máscara de brilho
        glow_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.circle(glow_mask, (int(cx), int(cy)), 30, 255, -1)
        
        # Aplicar blur para criar brilho
        glow_blur = cv2.GaussianBlur(glow_mask, (61, 61), 0)
        glow_blur = cv2.merge([glow_blur, glow_blur, glow_blur])
        
        # Colorizar com roxo
        purple_tint = np.ones_like(trail_canvas) * (160, 32, 240)
        glow_final = cv2.multiply(glow_blur / 255, purple_tint / 255) * 255
        glow_final = glow_final.astype(np.uint8)
        
        # Adicionar brilho ao canvas
        cv2.addWeighted(trail_canvas, 1.0, glow_final, 0.3, 0, trail_canvas)
        if yolo_enabled:
            cv2.addWeighted(cam_trail_canvas, 1.0, glow_final, 0.3, 0, cam_trail_canvas)
        
        # Desenhar centro preto do buraco com shadowBlur roxo (como no HTML)
        # Simular shadowBlur com múltiplos círculos
        for blur_r in range(30, 0, -3):
            alpha = int(255 * (1 - blur_r / 30))
            glow_color = [int(160 * alpha / 255), int(32 * alpha / 255), int(240 * alpha / 255)]
            cv2.circle(trail_canvas, (smooth_cx, smooth_cy), 8 + blur_r, glow_color, -1)
        
        if yolo_enabled:
            for blur_r in range(30, 0, -3):
                alpha = int(255 * (1 - blur_r / 30))
                glow_color = [int(160 * alpha / 255), int(32 * alpha / 255), int(240 * alpha / 255)]
                cv2.circle(cam_trail_canvas, (smooth_cx, smooth_cy), 8 + blur_r, glow_color, -1)
        
        # Centro preto
        cv2.circle(trail_canvas, (smooth_cx, smooth_cy), 8, (0, 0, 0), -1)
        if yolo_enabled:
            cv2.circle(cam_trail_canvas, (smooth_cx, smooth_cy), 8, (0, 0, 0), -1)
    
    def get_config_ui(self, parent_frame, save_callback):
        """Este efeito não tem configurações personalizáveis"""
        return None
    
    def cleanup(self):
        """Limpeza de recursos"""
        self.particles = []
