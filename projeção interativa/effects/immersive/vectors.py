import cv2
import numpy as np
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_fps import AdaptiveFPSController

class VectorsEffect:
    def __init__(self):
        """Inicializa o efeito Vetores Imersivos 1"""
        self.name = "Vetores Imersivos 1"
        self.category = "immersive"
        self.description = "Detecta bordas do vídeo em tempo real usando Canny"
        self.effect_id = "effect_vectors"
        
        # Buffer temporal para filtro de bordas
        self._edge_buffer = None
        self._edge_buffer_size = 3  # Reduzido de 5 para 3 para menos processamento
        
        # Controlador de FPS adaptativo
        self._fps_controller = AdaptiveFPSController()
    
    def apply(self, frame, trail_canvas, cam_trail_canvas, H, yolo_enabled, trackers, trackers_lock, detection_data, config):
        """
        Aplica o efeito Vectors ao frame atual
        
        Parâmetros adicionais:
        - config: Dicionário de configurações do efeito (canny_threshold1, canny_threshold2, etc)
        """
        # Inicializar buffer temporal se necessário ou se o tamanho mudou
        frame_height, frame_width = frame.shape[:2]
        if self._edge_buffer is None or self._edge_buffer.shape != (frame_height, frame_width):
            self._edge_buffer = np.zeros((frame_height, frame_width), dtype=np.uint8)
        
        # LIMPAR canvas completamente a cada frame (será ajustado no modo adaptativo)
        # Não limpa aqui - será controlado pelo sistema FPS adaptativo
        
        # Carregar configurações
        canny1 = config["canny_threshold1"]
        canny2 = config["canny_threshold2"]
        density = config["vector_density"]
        consistency_threshold = config.get("consistency_threshold", 30)
        edge_thickness = config.get("edge_thickness", 1)
        contour_mode = config.get("contour_mode", "center")
        particle_size = config.get("particle_size", 1)
        ignore_green_blue = config.get("ignore_green_blue", False)
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Aplicar blur leve para reduzir ruído
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Detectar bordas com Canny
        edges = cv2.Canny(blurred, canny1, canny2)
        
        # Atualizar buffer temporal (incrementar bordas atuais)
        self._edge_buffer = np.clip(self._edge_buffer + (edges > 0).astype(np.uint8), 0, self._edge_buffer_size)
        
        # Decrementar buffer apenas para pixels que não são bordas atuais (bordas antigas diminuem)
        mask_decrement = (self._edge_buffer > 0) & (edges == 0)
        self._edge_buffer[mask_decrement] -= 1
        
        # Filtrar bordas por consistência (só mantém as que aparecem em X% dos frames)
        min_consistency = int(self._edge_buffer_size * consistency_threshold / 100)
        stable_edges = (self._edge_buffer >= min_consistency).astype(np.uint8)
        
        # Calcular gradiente para direção das bordas
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Encontrar coordenadas das bordas estáveis
        stable_edge_coords = np.where(stable_edges > 0)
        edge_count = len(stable_edge_coords[0])
        
        # Limite máximo de bordas para evitar travamento
        MAX_EDGES = 5000
        if edge_count > MAX_EDGES:
            # Amostrar aleatoriamente para não travar
            indices = np.random.choice(edge_count, MAX_EDGES, replace=False)
            stable_edge_coords = (stable_edge_coords[0][indices], stable_edge_coords[1][indices])
            edge_count = MAX_EDGES
        
        # Sistema de FPS adaptativo usando controlador genérico
        if not self._fps_controller.should_process_frame(edge_count, trail_canvas):
            return  # Pula frame - mantém canvas
        
        # Limpa canvas para processar novo frame
        trail_canvas[:] = 0
        cam_trail_canvas[:] = 0
        
        # Sem limite de edges - processa tudo que detectar (estilo stop-motion)
        step = max(1, density)
        for i in range(0, len(stable_edge_coords[0]), step):
            y, x = stable_edge_coords[0][i], stable_edge_coords[1][i]
            
            # Verificar se deve projetar apenas em bordas verdes e azuis
            if ignore_green_blue:
                b, g, r = frame[y, x]
                # Processar APENAS se for predominantemente verde ou azul
                is_green = g > 100 and g > r * 1.5 and g > b * 1.5
                is_blue = b > 100 and b > r * 1.5 and b > g * 1.5
                if not (is_green or is_blue):
                    continue  # Pular se não for verde nem azul
            
            # Calcular direção do gradiente
            gx = sobel_x[y, x]
            gy = sobel_y[y, x]
            angle = math.atan2(gy, gx)
            
            # Coordenadas na câmera
            pt_cam = (int(x), int(y))
            
            # Transformar para coordenadas do projetor
            if H is not None:
                p_cam = np.float32([[[x, y]]])
                p_proj = cv2.perspectiveTransform(p_cam, H)
                pt_proj = (int(p_proj[0][0][0]), int(p_proj[0][0][1]))
            else:
                pt_proj = pt_cam
            
            # Cor baseada na direção do gradiente (HSV)
            hue = int((angle + math.pi) / (2 * math.pi) * 180)
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            
            # Calcular offset para modo lateral
            offset = edge_thickness if contour_mode == "lateral" else 0
            if offset > 0:
                # Calcular direção perpendicular ao gradiente
                perp_angle = angle + math.pi / 2
                offset_x = int(offset * math.cos(perp_angle))
                offset_y = int(offset * math.sin(perp_angle))
            
            # Desenhar no canvas do projetor
            if contour_mode == "center":
                # Modo central: linha única no meio
                cv2.circle(trail_canvas, pt_proj, particle_size, tuple(map(int, color_bgr)), -1)
            else:
                # Modo lateral: duas linhas de cada lado
                pt1_proj = (pt_proj[0] + offset_x, pt_proj[1] + offset_y)
                pt2_proj = (pt_proj[0] - offset_x, pt_proj[1] - offset_y)
                cv2.circle(trail_canvas, pt1_proj, particle_size, tuple(map(int, color_bgr)), -1)
                cv2.circle(trail_canvas, pt2_proj, particle_size, tuple(map(int, color_bgr)), -1)
            
            # Desenhar no monitor (se Visão IA estiver ativa)
            if yolo_enabled:
                if contour_mode == "center":
                    cv2.circle(cam_trail_canvas, pt_cam, particle_size, tuple(map(int, color_bgr)), -1)
                else:
                    pt1_cam = (pt_cam[0] + offset_x, pt_cam[1] + offset_y)
                    pt2_cam = (pt_cam[0] - offset_x, pt_cam[1] - offset_y)
                    cv2.circle(cam_trail_canvas, pt1_cam, particle_size, tuple(map(int, color_bgr)), -1)
                    cv2.circle(cam_trail_canvas, pt2_cam, particle_size, tuple(map(int, color_bgr)), -1)
        
        # Adicionar indicador de atividade
        edge_count = len(stable_edge_coords[0])
        cv2.putText(trail_canvas, f"EDGES: {edge_count}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Debug: mostrar se está desenhando pontos
        points_drawn = 0
        step = max(1, density)
        for i in range(0, len(stable_edge_coords[0]), step):
            points_drawn += 1
        cv2.putText(trail_canvas, f"POINTS: {points_drawn}", 
                   (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def get_config_ui(self, parent_frame, save_callback, config):
        """
        Retorna widgets de configuração do efeito
        
        Parâmetros:
        - parent_frame: Frame tkinter onde adicionar os widgets
        - save_callback: Função para chamar quando configuração mudar
        - config: Dicionário de configurações atual
        
        Retorna: Lista de widgets tkinter
        """
        import tkinter as tk
        from tkinter import ttk
        
        widgets = []
        
        # Slider: Canny Threshold 1
        tk.Label(parent_frame, text="Sensibilidade de Borda (Canny Threshold 1):", 
                 fg="#bac2de", bg="#1e1e2e", font=("Arial", 9)).pack(anchor=tk.W, pady=(5, 0))
        
        canny1_var = tk.IntVar(value=config["canny_threshold1"])
        
        def on_canny1_change(val):
            config["canny_threshold1"] = int(val)
            save_callback()
        
        canny1_slider = tk.Scale(parent_frame, from_=5, to=100, orient=tk.HORIZONTAL,
                                variable=canny1_var, command=on_canny1_change,
                                bg="#1e1e2e", fg="#cdd6f4", troughcolor="#313244",
                                highlightthickness=0, font=("Arial", 9))
        canny1_slider.pack(fill=tk.X, pady=5)
        widgets.append(canny1_slider)
        
        tk.Label(parent_frame, text="Menor = mais sensível | Maior = menos sensível",
                 fg="#6c7086", bg="#1e1e2e", font=("Arial", 8)).pack(anchor=tk.W)
        
        # Slider: Canny Threshold 2
        tk.Label(parent_frame, text="Sensibilidade de Borda (Canny Threshold 2):", 
                 fg="#bac2de", bg="#1e1e2e", font=("Arial", 9)).pack(anchor=tk.W, pady=(15, 0))
        
        canny2_var = tk.IntVar(value=config["canny_threshold2"])
        
        def on_canny2_change(val):
            config["canny_threshold2"] = int(val)
            save_callback()
        
        canny2_slider = tk.Scale(parent_frame, from_=20, to=200, orient=tk.HORIZONTAL,
                                variable=canny2_var, command=on_canny2_change,
                                bg="#1e1e2e", fg="#cdd6f4", troughcolor="#313244",
                                highlightthickness=0, font=("Arial", 9))
        canny2_slider.pack(fill=tk.X, pady=5)
        widgets.append(canny2_slider)
        
        tk.Label(parent_frame, text="Menor = mais bordas | Maior = menos bordas",
                 fg="#6c7086", bg="#1e1e2e", font=("Arial", 8)).pack(anchor=tk.W)
        
        # Slider: Densidade
        tk.Label(parent_frame, text="Densidade de Vetores (Amostragem):", 
                 fg="#bac2de", bg="#1e1e2e", font=("Arial", 9)).pack(anchor=tk.W, pady=(15, 0))
        
        density_var = tk.IntVar(value=config["vector_density"])
        
        def on_density_change(val):
            config["vector_density"] = int(val)
            save_callback()
        
        density_slider = tk.Scale(parent_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                 variable=density_var, command=on_density_change,
                                 bg="#1e1e2e", fg="#cdd6f4", troughcolor="#313244",
                                 highlightthickness=0, font=("Arial", 9))
        density_slider.pack(fill=tk.X, pady=5)
        widgets.append(density_slider)
        
        tk.Label(parent_frame, text="Menor = mais detalhado | Maior = mais suave",
                 fg="#6c7086", bg="#1e1e2e", font=("Arial", 8)).pack(anchor=tk.W)
        
        # Slider: Consistência
        tk.Label(parent_frame, text="Consistência (Filtro Anti-Piscamento):", 
                 fg="#bac2de", bg="#1e1e2e", font=("Arial", 9)).pack(anchor=tk.W, pady=(15, 0))
        
        consistency_var = tk.IntVar(value=config.get("consistency_threshold", 30))
        
        def on_consistency_change(val):
            config["consistency_threshold"] = int(val)
            save_callback()
        
        consistency_slider = tk.Scale(parent_frame, from_=20, to=100, orient=tk.HORIZONTAL,
                                       variable=consistency_var, command=on_consistency_change,
                                       bg="#1e1e2e", fg="#cdd6f4", troughcolor="#313244",
                                       highlightthickness=0, font=("Arial", 9))
        consistency_slider.pack(fill=tk.X, pady=5)
        widgets.append(consistency_slider)
        
        tk.Label(parent_frame, text="Menor = mais sensível | Maior = mais estável (filtra piscamento)",
                 fg="#6c7086", bg="#1e1e2e", font=("Arial", 8)).pack(anchor=tk.W)
        
        # Slider: Espessura dos Contornos
        tk.Label(parent_frame, text="Espessura dos Contornos:", 
                 fg="#bac2de", bg="#1e1e2e", font=("Arial", 9)).pack(anchor=tk.W, pady=(15, 0))
        
        thickness_var = tk.IntVar(value=config.get("edge_thickness", 1))
        
        def on_thickness_change(val):
            config["edge_thickness"] = int(val)
            save_callback()
        
        thickness_slider = tk.Scale(parent_frame, from_=1, to=5, orient=tk.HORIZONTAL,
                                    variable=thickness_var, command=on_thickness_change,
                                    bg="#1e1e2e", fg="#cdd6f4", troughcolor="#313244",
                                    highlightthickness=0, font=("Arial", 9))
        thickness_slider.pack(fill=tk.X, pady=5)
        widgets.append(thickness_slider)
        
        tk.Label(parent_frame, text="1 = fino | 5 = grosso",
                 fg="#6c7086", bg="#1e1e2e", font=("Arial", 8)).pack(anchor=tk.W)
        
        # Slider: Tamanho das Partículas
        tk.Label(parent_frame, text="Tamanho das Partículas:", 
                 fg="#bac2de", bg="#1e1e2e", font=("Arial", 9)).pack(anchor=tk.W, pady=(15, 0))
        
        particle_var = tk.IntVar(value=config.get("particle_size", 1))
        
        def on_particle_change(val):
            config["particle_size"] = int(val)
            save_callback()
        
        particle_slider = tk.Scale(parent_frame, from_=1, to=3, orient=tk.HORIZONTAL,
                                   variable=particle_var, command=on_particle_change,
                                   bg="#1e1e2e", fg="#cdd6f4", troughcolor="#313244",
                                   highlightthickness=0, font=("Arial", 9))
        particle_slider.pack(fill=tk.X, pady=5)
        widgets.append(particle_slider)
        
        tk.Label(parent_frame, text="1 = pequeno | 3 = grande",
                 fg="#6c7086", bg="#1e1e2e", font=("Arial", 8)).pack(anchor=tk.W)
        
        # Radiobutton: Modo de Contorno
        tk.Label(parent_frame, text="Modo de Contorno:", 
                 fg="#bac2de", bg="#1e1e2e", font=("Arial", 9)).pack(anchor=tk.W, pady=(15, 0))
        
        contour_mode_var = tk.StringVar(value=config.get("contour_mode", "center"))
        
        def on_contour_mode_change(val):
            config["contour_mode"] = val
            save_callback()
        
        contour_frame = tk.Frame(parent_frame, bg="#1e1e2e")
        contour_frame.pack(fill=tk.X, pady=5)
        
        tk.Radiobutton(contour_frame, text="Central (linha no meio)", variable=contour_mode_var, 
                      value="center", command=lambda: on_contour_mode_change("center"),
                      bg="#1e1e2e", fg="#cdd6f4", selectcolor="#313244",
                      activebackground="#1e1e2e", activeforeground="#cdd6f4",
                      font=("Arial", 9)).pack(anchor=tk.W)
        tk.Radiobutton(contour_frame, text="Lateral (duas linhas dos lados)", variable=contour_mode_var, 
                      value="lateral", command=lambda: on_contour_mode_change("lateral"),
                      bg="#1e1e2e", fg="#cdd6f4", selectcolor="#313244",
                      activebackground="#1e1e2e", activeforeground="#cdd6f4",
                      font=("Arial", 9)).pack(anchor=tk.W)
        
        return widgets
    
    def cleanup(self):
        """Limpa recursos do efeito"""
        self._edge_buffer = None
