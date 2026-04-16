import cv2
import numpy as np
import math
import random

class LanternConeEffect:
    def __init__(self):
        """Inicializa o efeito Lamparina e Cone"""
        self.name = "Lamparina e Cone"
        self.category = "future"
        self.description = "Lamparina com cone de luz interativo e poeira flutuante"
        self.effect_id = "effect_lantern_cone"
        
        # Estado do jogo
        self.charge = 100
        self.batteries = 0
        self.pickups = []
        self.dust = []
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
        """Aplica o efeito Lamparina e Cone ao frame atual"""
        d = detection_data
        tid = d["id"]
        x, y, w, h = d["box"]
        cx = d.get("true_cx", x + w // 2)
        cy = d.get("true_cy", y + h // 2)
        
        # Aplicar suavização à posição (amortecer vibração)
        if self.smoothed_cx is None:
            self.smoothed_cx = cx
            self.smoothed_cy = cy
        else:
            self.smoothed_cx = cx * self.smoothing_factor + self.smoothed_cx * (1 - self.smoothing_factor)
            self.smoothed_cy = cy * self.smoothing_factor + self.smoothed_cy * (1 - self.smoothing_factor)
        
        smooth_cx = int(self.smoothed_cx)
        smooth_cy = int(self.smoothed_cy)
        
        # Calcular ângulo do movimento
        angle = 0
        with trackers_lock:
            if tid in trackers and "history" in trackers[tid]:
                hist = trackers[tid]["history"]
                if len(hist) >= 2:
                    angle = math.atan2(smooth_cy - hist[-2][1], smooth_cx - hist[-2][0])
        
        # Sistema de FPS adaptativo
        total_entities = len(self.dust) + len(self.pickups)
        if not self._fps_controller.should_process_frame(total_entities, trail_canvas):
            return
        
        # Inicializar poeira se necessário
        if len(self.dust) == 0:
            for i in range(300):
                self.dust.append({
                    'x': random.uniform(0, self.width),
                    'y': random.uniform(0, self.height),
                    'vx': random.uniform(-0.15, 0.15),
                    'vy': random.uniform(-0.15, 0.15),
                    'size': random.uniform(0.3, 1.5)
                })
        
        # Atualizar carga
        self.charge -= 0.1
        if self.charge <= 0:
            if self.batteries > 0:
                self.batteries -= 1
                self.charge = 100
            else:
                self.charge = 0
        
        # Spawn baterias
        if len(self.pickups) < 5:
            self.pickups.append({
                'x': random.uniform(0, self.width),
                'y': random.uniform(0, self.height)
            })
        
        # Coletar baterias
        for i in range(len(self.pickups) - 1, -1, -1):
            dist = math.hypot(smooth_cx - self.pickups[i]['x'], smooth_cy - self.pickups[i]['y'])
            if dist < 25 and self.batteries < 3:
                self.batteries += 1
                self.pickups.pop(i)
        
        # Atualizar poeira
        for d in self.dust:
            d['x'] += d['vx']
            d['y'] += d['vy']
            if d['x'] < 0:
                d['x'] = self.width
            elif d['x'] > self.width:
                d['x'] = 0
            if d['y'] < 0:
                d['y'] = self.height
            elif d['y'] > self.height:
                d['y'] = 0
        
        # Desenhar fundo escuro
        trail_canvas[:] = (2, 2, 2)
        if yolo_enabled:
            cam_trail_canvas[:] = (2, 2, 2)
        
        # Parâmetros da luz (igual ao HTML)
        lamp_radius = 100 if self.charge > 0 else 40
        cone_length = 300 if self.charge > 0 else 100
        cone_spread = math.pi / 2.5
        
        # Criar máscara de luz com gradientes radiais (como no HTML)
        light_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Luz circular com gradiente radial (lamparina)
        for r in range(lamp_radius, 0, -2):
            alpha = int(255 * (1 - r / lamp_radius))
            cv2.circle(light_mask, (smooth_cx, smooth_cy), r, alpha, -1)
        
        # Cone de luz com gradiente radial
        cone_points = []
        cone_points.append((int(smooth_cx), int(smooth_cy)))
        
        # Calcular pontos do cone
        num_cone_points = 50
        for i in range(num_cone_points + 1):
            theta = angle - cone_spread + (2 * cone_spread * i / num_cone_points)
            px = int(smooth_cx + math.cos(theta) * cone_length)
            py = int(smooth_cy + math.sin(theta) * cone_length)
            cone_points.append((px, py))
        
        cone_points = np.array(cone_points, dtype=np.int32)
        
        # Desenhar cone com gradiente radial (simulado com múltiplos fills)
        temp_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for r in range(cone_length, 0, -10):
            alpha = int(230 * (1 - r / cone_length))
            # Criar cone truncado
            cone_points_trunc = []
            cone_points_trunc.append((smooth_cx, smooth_cy))
            for i in range(1, len(cone_points)):
                # Interpolar ponto baseado no raio
                pt = cone_points[i]
                dist = math.hypot(pt[0] - smooth_cx, pt[1] - smooth_cy)
                if dist <= r:
                    cone_points_trunc.append(pt)
            
            if len(cone_points_trunc) > 2:
                cone_points_trunc = np.array(cone_points_trunc, dtype=np.int32)
                cv2.fillPoly(temp_mask, [cone_points_trunc], alpha)
        
        # Combinar luz circular e cone
        cv2.add(light_mask, temp_mask, light_mask)
        
        # Aplicar máscara de luz
        light_mask_3ch = cv2.merge([light_mask, light_mask, light_mask])
        
        # Desenhar grade no canvas
        grid_color = (18, 18, 18)
        for gx in range(0, self.width, 15):
            cv2.line(trail_canvas, (gx, 0), (gx, self.height), grid_color, 1)
        for gy in range(0, self.height, 15):
            cv2.line(trail_canvas, (0, gy), (self.width, gy), grid_color, 1)
        
        if yolo_enabled:
            for gx in range(0, self.width, 15):
                cv2.line(cam_trail_canvas, (gx, 0), (gx, self.height), grid_color, 1)
            for gy in range(0, self.height, 15):
                cv2.line(cam_trail_canvas, (0, gy), (self.width, gy), grid_color, 1)
        
        # Desenhar baterias (verde)
        for p in self.pickups:
            cv2.rectangle(trail_canvas, (int(p['x']) - 2, int(p['y']) - 4), 
                         (int(p['x']) + 2, int(p['y']) + 4), (46, 204, 113), -1)
            cv2.rectangle(trail_canvas, (int(p['x']) - 1, int(p['y']) - 5), 
                         (int(p['x']) + 1, int(p['y']) - 5), (39, 174, 96), -1)
            
            if yolo_enabled:
                cv2.rectangle(cam_trail_canvas, (int(p['x']) - 2, int(p['y']) - 4), 
                             (int(p['x']) + 2, int(p['y']) + 4), (46, 204, 113), -1)
                cv2.rectangle(cam_trail_canvas, (int(p['x']) - 1, int(p['y']) - 5), 
                             (int(p['x']) + 1, int(p['y']) - 5), (39, 174, 96), -1)
        
        # Desenhar poeira (branco)
        for d in self.dust:
            cv2.circle(trail_canvas, (int(d['x']), int(d['y'])), int(d['size']), (255, 255, 255), -1)
            if yolo_enabled:
                cv2.circle(cam_trail_canvas, (int(d['x']), int(d['y'])), int(d['size']), (255, 255, 255), -1)
        
        # Aplicar máscara para revelar apenas áreas iluminadas
        result = cv2.bitwise_and(trail_canvas, light_mask_3ch)
        trail_canvas[:] = result
        
        if yolo_enabled:
            result_cam = cv2.bitwise_and(cam_trail_canvas, light_mask_3ch)
            cam_trail_canvas[:] = result_cam
        
        # Colorizar as luzes com modo 'lighter' (como no HTML)
        # Coloriza Lamparina
        lamp_color = (255, 200, 50) if self.charge > 0 else (255, 50, 0)
        lamp_overlay = np.zeros_like(trail_canvas)
        for r in range(lamp_radius, 0, -3):
            alpha = int(51 * (1 - r / lamp_radius))  # 0.2 * 255 = 51
            color_with_alpha = [int(c * alpha / 255) for c in lamp_color]
            cv2.circle(lamp_overlay, (smooth_cx, smooth_cy), r, color_with_alpha, -1)
        cv2.add(trail_canvas, lamp_overlay)
        
        if yolo_enabled:
            lamp_overlay_cam = np.zeros_like(cam_trail_canvas)
            for r in range(lamp_radius, 0, -3):
                alpha = int(51 * (1 - r / lamp_radius))
                color_with_alpha = [int(c * alpha / 255) for c in lamp_color]
                cv2.circle(lamp_overlay_cam, (smooth_cx, smooth_cy), r, color_with_alpha, -1)
            cv2.add(cam_trail_canvas, lamp_overlay_cam)
        
        # Coloriza Cone
        cone_color = (255, 220, 100) if self.charge > 0 else (255, 100, 0)
        cone_overlay = np.zeros_like(trail_canvas)
        for r in range(cone_length, 0, -10):
            alpha = int(38 * (1 - r / cone_length)) if self.charge > 0 else int(25 * (1 - r / cone_length))  # 0.15*255=38, 0.1*255=25
            # Desenhar cone truncado com gradiente
            cone_points_trunc = []
            cone_points_trunc.append((smooth_cx, smooth_cy))
            for i in range(1, len(cone_points)):
                pt = cone_points[i]
                dist = math.hypot(pt[0] - smooth_cx, pt[1] - smooth_cy)
                if dist <= r:
                    cone_points_trunc.append(pt)
            
            if len(cone_points_trunc) > 2:
                cone_points_trunc = np.array(cone_points_trunc, dtype=np.int32)
                color_with_alpha = [int(c * alpha / 255) for c in cone_color]
                cv2.fillPoly(cone_overlay, [cone_points_trunc], color_with_alpha)
        
        cv2.add(trail_canvas, cone_overlay)
        
        if yolo_enabled:
            cone_overlay_cam = np.zeros_like(cam_trail_canvas)
            for r in range(cone_length, 0, -10):
                alpha = int(38 * (1 - r / cone_length)) if self.charge > 0 else int(25 * (1 - r / cone_length))
                cone_points_trunc = []
                cone_points_trunc.append((smooth_cx, smooth_cy))
                for i in range(1, len(cone_points)):
                    pt = cone_points[i]
                    dist = math.hypot(pt[0] - smooth_cx, pt[1] - smooth_cy)
                    if dist <= r:
                        cone_points_trunc.append(pt)
                
                if len(cone_points_trunc) > 2:
                    cone_points_trunc = np.array(cone_points_trunc, dtype=np.int32)
                    color_with_alpha = [int(c * alpha / 255) for c in cone_color]
                    cv2.fillPoly(cone_overlay_cam, [cone_points_trunc], color_with_alpha)
            
            cv2.add(cam_trail_canvas, cone_overlay_cam)
        
        # Desenhar cursor
        cv2.circle(trail_canvas, (smooth_cx, smooth_cy), 2, (119, 119, 119), -1)
        if yolo_enabled:
            cv2.circle(cam_trail_canvas, (smooth_cx, smooth_cy), 2, (119, 119, 119), -1)
        
        # Interface de status
        status_color = (255, 51, 51) if self.charge <= 0 else (170, 170, 170)
        status_text = f"{max(0, int(self.charge))}% | Bat: {self.batteries}/3"
        cv2.putText(trail_canvas, status_text, (smooth_cx + 8, smooth_cy + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, status_color, 1)
        if yolo_enabled:
            cv2.putText(cam_trail_canvas, status_text, (smooth_cx + 8, smooth_cy + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, status_color, 1)
    
    def get_config_ui(self, parent_frame, save_callback):
        """Este efeito não tem configurações personalizáveis"""
        return None
    
    def cleanup(self):
        """Limpeza de recursos"""
        self.pickups = []
        self.dust = []
        self.charge = 100
        self.batteries = 0
