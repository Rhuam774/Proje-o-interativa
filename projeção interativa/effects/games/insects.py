import cv2
import numpy as np
import math
import random
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_fps import AdaptiveFPSController

class InsectsEffect:
    def __init__(self):
        """Inicializa o jogo Ataque de Insetos"""
        self.name = "Sobrevivência"
        self.category = "games"
        self.description = "Ataque de Insetos - Sobreviva às aranhas"
        self.effect_id = "effect_insects"
        
        # Estado do jogo
        self._insects = []
        self._game_over_state = {}
        self._game_reset_at = 0
        self._losing_obj_pos = (0, 0, 0, 0)
        
        # Controlador de FPS adaptativo
        self._fps_controller = AdaptiveFPSController()
    
    def apply(self, frame, trail_canvas, cam_trail_canvas, H, yolo_enabled, trackers, trackers_lock, detection_data, all_detections, projector_frame, monitor_frame):
        """
        Aplica o jogo Insects ao frame atual
        
        Parâmetros adicionais específicos para jogos:
        - all_detections: Lista de todas as detecções atuais
        - projector_frame: Frame do projetor para desenhar o jogo
        - monitor_frame: Frame do monitor para desenhar o jogo
        """
        now = time.time()
        
        # Gerenciador global de reinicialização (Game Over compartilhado)
        if self._game_reset_at > 0:
            delta = now - self._game_reset_at
            if delta < 2.5:  # 2.5 segundos de luto/tela vermelha
                px, py, cx, cy = self._losing_obj_pos
                cv2.putText(projector_frame, "VOCE PERDEU!", (int(px - 110), int(py - 30)), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 4)
                cv2.putText(projector_frame, "REINICIANDO...", (int(px - 80), int(py + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if yolo_enabled:
                    cv2.putText(monitor_frame, "GAME OVER", (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return  # Retorna para não processar mais
            else:
                self._game_reset_at = 0  # Reinicia o jogo!
                self._insects = []  # Limpa sobreviventes do round anterior
        
        alive_dets = all_detections
        
        # Sistema de FPS adaptativo baseado no número de insetos + detecções
        total_entities = len(self._insects) + len(alive_dets)
        if not self._fps_controller.should_process_frame(total_entities, projector_frame):
            return  # Pula frame - mantém canvas
        
        # Spawn aleatório das bordas da área de projeção
        if alive_dets and len(self._insects) < min(15, len(alive_dets) * 5):
            if random.random() < 0.1:
                edge = random.choice(["top", "bottom", "left", "right"])
                if edge == "top":
                    ix, iy = random.randint(0, 640), -10
                elif edge == "bottom":
                    ix, iy = random.randint(0, 640), 490
                elif edge == "left":
                    ix, iy = -10, random.randint(0, 480)
                else:
                    ix, iy = 650, random.randint(0, 480)
                self._insects.append({"x": float(ix), "y": float(iy), "s": random.uniform(2.5, 6.0)})
        
        dead_insects = []
        for i, ins in enumerate(self._insects):
            if not alive_dets:
                break
            
            # Procura o objeto (presa) mais próximo!
            best_d = None
            best_dist = 999999
            best_px, best_py = None, None
            
            for d in alive_dets:
                cx = d.get("true_cx", d["box"][0] + d["box"][2] // 2)
                cy = d.get("true_cy", d["box"][1] + d["box"][3] // 2)
                
                px, py = cx, cy
                if H is not None:  # Distância é baseada no mundo do Projetor
                    c_cam = np.float32([[[cx, cy]]])
                    c_proj = cv2.perspectiveTransform(c_cam, H)[0][0]
                    px, py = c_proj[0], c_proj[1]
                
                dist = math.hypot(ins["x"] - px, ins["y"] - py)
                if dist < best_dist:
                    best_dist = dist
                    best_d = d
                    best_px, best_py = px, py
            
            if best_d is not None:
                angle = math.atan2(best_py - ins["y"], best_px - ins["x"])
                wobble = math.sin(now * 30.0 + i) * 0.4  # Faz eles andarem rebolando
                
                ins["x"] += math.cos(angle + wobble) * ins["s"]
                ins["y"] += math.sin(angle + wobble) * ins["s"]
                
                ix, iy = int(ins["x"]), int(ins["y"])
                
                # Projetor desenha corpo e patas
                cv2.circle(projector_frame, (ix, iy), 5, (255, 255, 255), -1)
                cv2.circle(projector_frame, (ix, iy), 7, (0, 0, 255), 1)
                cv2.line(projector_frame, (ix, iy), (ix+int(math.cos(angle+1.5+wobble)*8), iy+int(math.sin(angle+1.5+wobble)*8)), (255, 255, 255), 1)
                cv2.line(projector_frame, (ix, iy), (ix+int(math.cos(angle-1.5-wobble)*8), iy+int(math.sin(angle-1.5-wobble)*8)), (255, 255, 255), 1)
                
                # Monitor só desenha se Visão IA estiver ligada
                if yolo_enabled:
                    cv2.circle(monitor_frame, (ix, iy), 3, (255, 255, 255), -1)
                    cv2.circle(monitor_frame, (ix, iy), 5, (0, 0, 255), 1)
                
                # Colisão! Se a aranha tocar o corpo
                avg_radius = (best_d["box"][2] + best_d["box"][3]) / 4.0
                if H is not None:
                    avg_radius *= 640 / max(1, 640)
                
                if best_dist < max(15, avg_radius * 0.8):
                    self._game_reset_at = now
                    self._losing_obj_pos = (best_px, best_py, cx, cy)  # Salva local da morte
                    dead_insects.append(i)
                    break  # Para o loop de insetos pois o jogo acabou
        
        # Deleta os insetos suicidas que mataram o jogador
        for i in sorted(dead_insects, reverse=True):
            self._insects.pop(i)
    
    def get_config_ui(self, parent_frame, save_callback):
        """Este jogo não tem configurações personalizáveis"""
        return None
    
    def cleanup(self):
        """Limpa recursos do jogo"""
        self._insects = []
        self._game_reset_at = 0
