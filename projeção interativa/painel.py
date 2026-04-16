import cv2
import numpy as np
import time
import math
import threading
import random
import sys
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import json
import os
from effects.effect_loader import EffectLoader
class Config:
    # ─── MUDANÇA DE CÂMERA AQUI ───
    # 0 = Câmera do notebook
    # 1, 2 = Câmera Virtual USB (ex: DroidCam, Iriun Webcam via cabo)
    # "http://192.168.1.105:8080/video" = IP Webcam (IP Fixo Sugerido)
    # Configurações padrão: 240x180, equilíbrio qualidade/performance
    # Alternativas disponíveis (descomente para usar):
    # CAMERA_ID = "http://192.168.0.227:8080/video?resolution=640x480&quality=25&fps=8"  # 640x480 (alta qualidade)
    # CAMERA_ID = "http://192.168.0.227:8080/video?resolution=320x240&quality=25&fps=8"  # 320x240 (média qualidade)
    # CAMERA_ID = "http://192.168.0.227:8080/video?resolution=240x180&quality=25&fps=8"  # 240x180 (compacto)
    CAMERA_ID = 0  # Câmera do notebook (fallback se IP não conectar)
    # CAMERA_ID = "http://192.168.0.227:8080/video?resolution=640x480&quality=25&fps=8"  # IP Webcam 

    # Configurações correspondentes (ajuste junto com CAMERA_ID acima):
    # WIDTH = 640; HEIGHT = 480; FPS = 8  # Para 640x480
    # WIDTH = 320; HEIGHT = 240; FPS = 8  # Para 320x240
    # WIDTH = 240; HEIGHT = 180; FPS = 8  # Para 240x180
    WIDTH = 640
    HEIGHT = 480
    FPS = 8 # Configurações para 640x480 (alta qualidade)

    # Cores (BGR)
    CYAN    = (220, 255, 0)
    GREEN   = (0, 255, 120)
    RED     = (60, 60, 255)
    YELLOW  = (0, 220, 255)
    WHITE   = (255, 255, 255)
    ORANGE  = (0, 165, 255)
    MAGENTA = (255, 100, 255)
    GRAY    = (100, 100, 100)
    DARK    = (20, 20, 20)
    PANEL   = (35, 35, 45)
    ACCENT  = (255, 150, 0)


# (Classe Button removida - UI agora é Tkinter nativo)


# ═══════════════════════════════════════════════════════════
# APLICAÇÃO PRINCIPAL
# ═══════════════════════════════════════════════════════════
class App:
    def __init__(self):
        # ─── PING RÁPIDO (Evita Timeout Travado do OpenCV) ───
        if isinstance(Config.CAMERA_ID, str) and Config.CAMERA_ID.startswith("http"):
            import socket
            import urllib.parse
            try:
                parsed = urllib.parse.urlparse(Config.CAMERA_ID)
                host = parsed.hostname
                port = parsed.port or 80
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.2)  # Meio segundo no máximo, se não carregar já era
                if s.connect_ex((host, port)) != 0:
                    print(f"\n[Acesso Rápido] Celular ({host}) não conectado. Puxando Notebook...")
                    Config.CAMERA_ID = 0
                s.close()
            except:
                Config.CAMERA_ID = 0

        # Câmera Dinâmica (Celular IP ou USB)
        if isinstance(Config.CAMERA_ID, str):
            # Para câmeras IP, adicionar timeout e retry
            import urllib.parse
            parsed = urllib.parse.urlparse(Config.CAMERA_ID)
            if parsed.hostname:
                import socket
                try:
                    # Testar conexão primeiro
                    socket.create_connection((parsed.hostname, parsed.port or 8080), timeout=3)
                except:
                    print(f"[Aviso] Celular ({parsed.hostname}) não conectado. Puxando Notebook...")
                    Config.CAMERA_ID = 0
            
            self.cap = cv2.VideoCapture(Config.CAMERA_ID) # Via IP (Wi-Fi)
        else:
            self.cap = cv2.VideoCapture(Config.CAMERA_ID, cv2.CAP_DSHOW) # Via USB/Embutida
            
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.FPS)
        
        # Para câmeras IP, limpar o buffer inicial para evitar delay
        if isinstance(Config.CAMERA_ID, str):
            for _ in range(10):  # Limpar 10 frames iniciais
                self.cap.read()
        
        # Ajuste FFMPEG só funciona em cameras USB/Locais, evitamos para IP para não corromper
        if not isinstance(Config.CAMERA_ID, str):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Estado
        self.running = True
        self.yolo_enabled = False
        self.yolo_model = None
        self.simulation_mode = False
        self.selecting_roi = False
        self.roi_start = None
        self.roi_end = None

        # Rastreadores MOSSE
        self.trackers = {}
        self._next_id = 0
        self._trackers_lock = threading.Lock()
        self._pending_auto_trackers = []

        # YOLO — roda em thread separada, atua apenas como "Descobridor"
        self._yolo_frame = None  # Frame compartilhado com a thread YOLO
        self._yolo_lock = threading.Lock()
        self._yolo_thread = None
        self._yolo_dets = []  # Detecções YOLO compartilhadas

        # Calibração
        self.calibrating = False
        self.calib_pts_cam = []
        self.calib_pts_proj = [(0, 0), (Config.WIDTH, 0), (Config.WIDTH, Config.HEIGHT), (0, Config.HEIGHT)]
        self.calib_step = 0
        self._H = None  # Matriz de homografia
        try:
            import os
            if os.path.exists('calibration.npy'):
                self._H = np.load('calibration.npy')
                print("[Sistema] Matriz de calibração anterior carregada com sucesso.")
        except Exception as e:
            print(f"[Sistema] Erro ao carregar calibração anterior: {e}")

        # FPS counter
        self._fps = 0
        self._frame_times = []
        
        # Suavização de posição dos rastreadores (para reduzir tremor no contorno)
        self._smoothed_positions = {}
        # Histórico de qualidade do rastreamento (para CSRT)
        self._tracker_quality = {}

        # Efeitos de Projeção
        self.active_effect = None  # Efeito imersivo desativado por padrão (ativa pelo botão)
        self._trail_canvas = None  # Canvas persistente para rastros
        self._trail_fade_rate = 3  # Desvanecimento suave do rastro cósmico

        # Sistema de carregamento de efeitos
        self.effect_loader = EffectLoader()
        self.effect_loader.load_all_effects()
        # Forçar recarregamento para garantir alterações mais recentes
        self.effect_loader.reload_effects()
        # Recarregar módulos específicos para garantir alterações
        import importlib
        import sys
        for module_name in list(sys.modules.keys()):
            if 'future' in module_name or 'colliding_balls' in module_name or 'neon_ribbon' in module_name or 'neural_network' in module_name or 'lantern_cone' in module_name or 'black_hole' in module_name or 'boat_wake' in module_name:
                try:
                    importlib.reload(sys.modules[module_name])
                except:
                    pass
        self.effect_loader.reload_effects()
        print(f"[Sistema] Efeitos carregados dinamicamente: {len(self.effect_loader.get_all_effects())}")

        # Configurações do efeito Vetores Imersivos
        self._vector_config = self._load_vector_config()
        self._vector_config_panel = None  # Referência para o painel de configuração

        # Buffer temporal para filtro de bordas (elimina piscamento)
        self._edge_buffer = None  # Armazena contagem de aparições de cada pixel
        self._edge_buffer_size = 5  # Número de frames para análise de consistência

        # Memória e Seleção
        self.selected_tracker = None
        self.memory_profiles = []
        self._memory_lock = threading.Lock()
        self._memory_thread = threading.Thread(target=self._memory_loop, daemon=True)
        self._memory_thread.start()

        # UI - Janela de Controle (Tkinter)
        self.root = tk.Tk()
        self.root.title("CONTRÔLE - Projeção Interativa")
        self.root.geometry("250x650")
        self.root.attributes("-topmost", True)
        self.root.configure(bg='#1e1e2e')
        
        # PROTOCOLO DE FECHAMENTO: Garante que o X da janela mate o processo
        self.root.protocol("WM_DELETE_WINDOW", self._handle_exit)


        # Estilo Escuro Premium
        lbl_title = tk.Label(self.root, text="CORE DASHBOARD", font=("Impact", 18), fg="#00d2ff", bg="#1e1e2e")
        lbl_title.pack(pady=20)

        def make_btn(text, cmd, color="#313244"):
            btn = tk.Button(self.root, text=text, command=cmd, font=("Arial", 10, "bold"),
                            bg=color, fg="white", activebackground="#45475a", 
                            relief=tk.FLAT, pady=10, cursor="hand2")
            btn.pack(fill=tk.X, padx=20, pady=5)
            return btn

        self.tk_btn_yolo = make_btn("🤖 VISÃO IA: OFF", lambda: self._handle_toggle_yolo(), color="#313244")
        self.tk_btn_sim = make_btn("SIMULAÇÃO: OFF", lambda: self._handle_toggle_sim())
        make_btn("CALIBRAR SISTEMA", lambda: self._start_calibration(), "#f38ba8")
        make_btn("LIMPAR RASTREADORES", lambda: self._handle_reset(), "#fab387")
        self.tk_btn_fx = make_btn("EFEITOS: SIMPLES (NENHUM)", lambda: self._launch_effects_panel(), "#89b4fa")
        make_btn("MEMÓRIA / CONFIGS", lambda: self._handle_config_panel(), "#cba6f7")
        make_btn("BUSCAR CELULAR (IP CAM)", lambda: self._launch_camera_panel(), "#f9e2af")
        make_btn("SAIR DO SISTEMA", lambda: self._handle_exit(), "#a6e3a1")

        self.root.protocol("WM_DELETE_WINDOW", self._handle_exit)

        # Monitores e Janelas
        import screeninfo
        screens = screeninfo.get_monitors()
        self.screens = screens
        print(f"[Sistema] Monitores detectados: {len(screens)}")
        
        # Janelas OpenCV
        cv2.namedWindow("PAINEL DE CONTROLE (PC)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("PAINEL DE CONTROLE (PC)", 800, 600)
        
        cv2.namedWindow("SAIDA PROJETOR", cv2.WND_PROP_FULLSCREEN)
        # Se houver segundo monitor (projetor), mover a janela para lá
        if len(screens) > 1:
            m = screens[1] # Segundo monitor
            cv2.moveWindow("SAIDA PROJETOR", m.x, m.y)
            cv2.setWindowProperty("SAIDA PROJETOR", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print(f"[Sistema] Projetor detectado em {m.x},{m.y}. Movendo janela...")
        else:
            cv2.resizeWindow("SAIDA PROJETOR", 640, 480)
            print("[Aviso] Apenas um monitor detectado. Use Alt+Tab para alternar!")

        self._click_ready = False 
        cv2.setMouseCallback("PAINEL DE CONTROLE (PC)", self._mouse_callback)

        self._cam_fail_count = 0
        if not self.cap.isOpened():
             print(f"[Aviso] Falha na câmera {Config.CAMERA_ID}. Buscando Câmeras USB/Notebook locais...")
             fallback_success = False
             
             for cid in [0, 1, 2]:
                 for backend in [cv2.CAP_ANY, cv2.CAP_DSHOW]:
                     self.cap = cv2.VideoCapture(cid, backend)
                     if self.cap.isOpened():
                         Config.CAMERA_ID = cid
                         fallback_success = True
                         break
                 if fallback_success: break
             
             if fallback_success:
                 self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                 self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.WIDTH)
                 self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.HEIGHT)
                 self.cap.set(cv2.CAP_PROP_FPS, Config.FPS)
                 print(f"[Aviso] Câmera local (ID {Config.CAMERA_ID}) ativada com sucesso.")
             else:
                 messagebox.showwarning("Câmera não encontrada", "Nenhuma câmera do notebook detectada ou conectada.\n\nVerifique permissões de Webcam.")

    def _load_vector_config(self):
        """Carrega configurações salvas do efeito Vetores Imersivos"""
        default_config = {
            "canny_threshold1": 20,
            "canny_threshold2": 60,
            "vector_density": 2,
            "consistency_threshold": 30,  # Porcentagem de consistência (30% = aparece em 30% dos frames)
            "edge_thickness": 1,  # Espessura dos contornos (1-5 pixels)
            "contour_mode": "center",  # "center" = linha no meio, "lateral" = duas linhas de cada lado
            "particle_size": 1,  # Tamanho das partículas (1-3 pixels)
            "ignore_green_blue": False  # Projetar apenas em bordas verdes e azuis (para garantir que as cores projetadas sejam as mesmas)
        }
        try:
            if os.path.exists('vector_config.json'):
                with open('vector_config.json', 'r') as f:
                    config = json.load(f)
                    # Mesclar com defaults para garantir que todas as chaves existam
                    default_config.update(config)
                    print("[Sistema] Configurações de vetores carregadas.")
                    return default_config
        except Exception as e:
            print(f"[Sistema] Erro ao carregar configurações de vetores: {e}")
        return default_config

    def _save_vector_config(self):
        """Salva configurações do efeito Vetores Imersivos"""
        try:
            with open('vector_config.json', 'w') as f:
                json.dump(self._vector_config, f, indent=4)
            print("[Sistema] Configurações de vetores salvas.")
        except Exception as e:
            print(f"[Sistema] Erro ao salvar configurações de vetores: {e}")

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mx, my = x, y
            # 1. Se estiver calibrando, o clique registra o ponto
            if self.calibrating:
                self._handle_calibration_record(mx, my)
                return

            # 2. Seleção de Rastreador Existente
            with self._trackers_lock:
                for tid, data in self.trackers.items():
                    tx, ty, tw, th = data["last_box"]
                    if tx <= mx <= tx + tw and ty <= my <= ty + th:
                        self.selected_tracker = tid
                        return

            # 3. Criar Novo Rastreador (Drag and Drop)
            self.selected_tracker = None
            self.selecting_roi = True
            self.roi_start = (x, y)
            self.roi_end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.selecting_roi:
            self.roi_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self._click_ready = True
            if self.selecting_roi:
                self.selecting_roi = False
                self.roi_end = (x, y)
                self._finish_roi_selection()

    def _handle_toggle_yolo(self):
        self.yolo_enabled = not self.yolo_enabled
        if self.yolo_enabled:
            if self._yolo_thread is None or not self._yolo_thread.is_alive():
                self._yolo_thread = threading.Thread(target=self._yolo_loop, daemon=True)
                self._yolo_thread.start()
        else:
            self._yolo_dets = []
            with self._trackers_lock:
                tids_to_remove = [tid for tid, d in self.trackers.items() if d.get("type") == "auto"]
                for tid in tids_to_remove:
                    del self.trackers[tid]
                self._pending_auto_trackers.clear()
        self.tk_btn_yolo.config(text=f"🤖 VISÃO IA: {'ON (ATIVA)' if self.yolo_enabled else 'OFF'}")

    def _handle_toggle_sim(self):
        self.simulation_mode = not self.simulation_mode
        self.tk_btn_sim.config(text=f"SIMULAÇÃO: {'ON' if self.simulation_mode else 'OFF'}")

    def _handle_reset(self):
        with self._trackers_lock:
            self.trackers = {}
            self._pending_auto_trackers = []
        self._trail_canvas = None  # Limpar rastros junto
        print("[Sistema] Rastreadores limpos.")

    def _launch_effects_panel(self):
        def run_tk():
            root = tk.Toplevel(self.root)
            root.title("EFEITOS DE PROJEÇÃO")
            root.geometry("500x700")
            root.attributes("-topmost", True)
            root.configure(bg='#1e1e2e', padx=20, pady=20)

            tk.Label(root, text="✨ PAINEL DE EFEITOS", font=("Impact", 16),
                     fg="#89b4fa", bg="#1e1e2e").pack(pady=(0, 15))

            # Status atual
            current_effect = self.effect_loader.get_effect(self.active_effect)
            current_display = current_effect.name if current_effect else "Rastro Simples Seco"
            status_var = tk.StringVar(value=current_display)
            status_lbl = tk.Label(root, textvariable=status_var, font=("Consolas", 11),
                                  fg="#a6e3a1", bg="#1e1e2e")
            status_lbl.pack(pady=(0, 10))

            def set_effect(name, display):
                self.active_effect = name
                if name is None:
                    self._trail_canvas = None
                    self._cam_trail_canvas = None
                    status_var.set(display)
                    self.tk_btn_fx.config(text="EFEITOS: SIMPLES")
                else:
                    status_var.set(display)
                    self.tk_btn_fx.config(text=f"EFEITOS: {display.upper()}")
                    # Se for o efeito de vetores, abrir painel de configuração
                    if name == "effect_vectors":
                        self._launch_vector_config_panel()
                print(f"[Efeitos] Ativado: {display}")

            # ─── CRIAR ABAS (TABS) ───
            notebook = ttk.Notebook(root)
            notebook.pack(fill=tk.BOTH, expand=True, pady=10)

            # Estilo das abas
            style = ttk.Style()
            style.theme_use('clam')
            style.configure("TNotebook", background="#1e1e2e", borderwidth=0)
            style.configure("TNotebook.Tab", background="#313244", foreground="#cdd6f4", 
                           padding=[10, 5], font=("Arial", 10, "bold"))
            style.map("TNotebook.Tab", background=[("selected", "#89b4fa")], 
                    foreground=[("selected", "#1e1e2e")])

            btn_style = {"font": ("Arial", 10, "bold"),
                         "relief": tk.FLAT, "pady": 10, "cursor": "hand2"}

            # Cores para botões
            button_colors = ["#cba6f7", "#89b4fa", "#f38ba8", "#89dceb", "#f9e2af", "#a6e3a1", "#94e2d5", "#f2cdcd", "#b4befe"]
            color_index = 0

            # ─── ABA 1: EFEITOS SIMPLES ───
            tab_simple = tk.Frame(notebook, bg='#1e1e2e', padx=10, pady=10)
            notebook.add(tab_simple, text="🎨 Efeitos Simples")

            simple_effects = self.effect_loader.get_effects_by_category("simple")
            if simple_effects:
                for effect_id, effect in simple_effects.items():
                    color = button_colors[color_index % len(button_colors)]
                    color_index += 1
                    tk.Button(tab_simple, text=f"🎨  {effect.name}", bg=color, fg="white",
                              command=lambda eid=effect_id, ename=effect.name: set_effect(eid, ename),
                              **btn_style).pack(fill=tk.X, pady=3)
            else:
                tk.Label(tab_simple, text="Nenhum efeito simples encontrado", fg="#6c7086", bg="#1e1e2e").pack(pady=20)

            # ─── ABA 2: JOGOS ───
            tab_games = tk.Frame(notebook, bg='#1e1e2e', padx=10, pady=10)
            notebook.add(tab_games, text="🎮 Jogos")

            game_effects = self.effect_loader.get_effects_by_category("games")
            if game_effects:
                for effect_id, effect in game_effects.items():
                    color = button_colors[color_index % len(button_colors)]
                    color_index += 1
                    tk.Button(tab_games, text=f"🎮  {effect.name}", bg=color, fg="white",
                              command=lambda eid=effect_id, ename=effect.name: set_effect(eid, ename),
                              **btn_style).pack(fill=tk.X, pady=3)
            else:
                tk.Label(tab_games, text="Nenhum jogo encontrado", fg="#6c7086", bg="#1e1e2e").pack(pady=20)

            # ─── ABA 3: VISUALIZAÇÕES IMERSIVAS ───
            tab_immersive = tk.Frame(notebook, bg='#1e1e2e', padx=10, pady=10)
            notebook.add(tab_immersive, text="🌟 Visualizações Imersivas")

            immersive_effects = self.effect_loader.get_effects_by_category("immersive")
            if immersive_effects:
                for effect_id, effect in immersive_effects.items():
                    color = button_colors[color_index % len(button_colors)]
                    color_index += 1
                    tk.Button(tab_immersive, text=f"🌟  {effect.name}", bg=color, fg="white",
                              command=lambda eid=effect_id, ename=effect.name: set_effect(eid, ename),
                              **btn_style).pack(fill=tk.X, pady=3)
                    
                    # Adicionar botão de configuração para efeito vectors
                    if effect_id == "effect_vectors":
                        tk.Button(tab_immersive, text="⚙️  Configurar Vetores", bg="#45475a", fg="white",
                                  command=lambda: self._launch_vector_config_panel(),
                                  **btn_style).pack(fill=tk.X, pady=3)
            else:
                tk.Label(tab_immersive, text="Nenhuma visualização imersiva encontrada", fg="#6c7086", bg="#1e1e2e").pack(pady=20)

            # ─── ABA 4: FUTUROS EFEITOS ───
            tab_future = tk.Frame(notebook, bg='#1e1e2e', padx=10, pady=10)
            notebook.add(tab_future, text="🚀 Futuros Efeitos")

            future_effects = self.effect_loader.get_effects_by_category("future")
            if future_effects:
                for effect_id, effect in future_effects.items():
                    color = button_colors[color_index % len(button_colors)]
                    color_index += 1
                    tk.Button(tab_future, text=f"🚀  {effect.name}", bg=color, fg="white",
                              command=lambda eid=effect_id, ename=effect.name: set_effect(eid, ename),
                              **btn_style).pack(fill=tk.X, pady=3)
            else:
                tk.Label(tab_future, text="Nenhum efeito futuro encontrado", fg="#6c7086", bg="#1e1e2e").pack(pady=20)

            # ─── BOTÃO DESATIVAR ───
            tk.Button(root, text="🚫  DESATIVAR EFEITOS", bg="#585b70", fg="white",
                      command=lambda: set_effect(None, "Rastro Simples Seco"),
                      font=("Arial", 11, "bold"), relief=tk.FLAT, pady=10, cursor="hand2").pack(fill=tk.X, pady=10)

            # ─── CONTROLES DO RASTRO ───
            ctrl_frame = tk.LabelFrame(root, text="Controles do Rastro", font=("Arial", 10, "bold"),
                                       fg="#cdd6f4", bg="#1e1e2e", padx=15, pady=15)
            ctrl_frame.pack(fill=tk.X, pady=10)

            # Slider de desvanecimento
            tk.Label(ctrl_frame, text="Velocidade de Desvanecimento:", fg="#bac2de",
                     bg="#1e1e2e", font=("Arial", 9)).pack(anchor=tk.W)
            
            fade_var = tk.IntVar(value=self._trail_fade_rate)

            def on_fade_change(val):
                self._trail_fade_rate = int(val)

            fade_slider = tk.Scale(ctrl_frame, from_=0, to=15, orient=tk.HORIZONTAL,
                                   variable=fade_var, command=on_fade_change,
                                   bg="#1e1e2e", fg="#cdd6f4", troughcolor="#313244",
                                   highlightthickness=0, font=("Arial", 9))
            fade_slider.pack(fill=tk.X, pady=5)

            tk.Label(ctrl_frame, text="0 = Permanente  |  15 = Desvanece rápido",
                     fg="#6c7086", bg="#1e1e2e", font=("Arial", 8)).pack(anchor=tk.W)

            # Botão limpar rastro
            def clear_trail():
                self._trail_canvas = None
                self._cam_trail_canvas = None
                print("[Efeitos] Canvas de rastros limpo.")

            tk.Button(ctrl_frame, text="🗑️  LIMPAR RASTROS DA TELA", bg="#f38ba8",
                      fg="white", font=("Arial", 10, "bold"), relief=tk.FLAT,
                      command=clear_trail, cursor="hand2").pack(fill=tk.X, pady=(15, 5))

        run_tk()

    def _launch_vector_config_panel(self):
        """Abre painel de configuração específico para o efeito Vetores Imersivos"""
        def run_tk():
            root = tk.Toplevel(self.root)
            root.title("⚙️ CONFIGURAÇÃO - VETORES IMERSIVOS 1")
            root.geometry("450x650")
            root.attributes("-topmost", True)
            root.configure(bg='#1e1e2e', padx=25, pady=25)

            tk.Label(root, text="🔮 VETORES IMERSIVOS 1", font=("Impact", 16),
                     fg="#b4befe", bg="#1e1e2e").pack(pady=(0, 20))

            # Frame principal de configurações
            config_frame = tk.LabelFrame(root, text="Parâmetros de Detecção", 
                                        font=("Arial", 11, "bold"),
                                        fg="#cdd6f4", bg="#1e1e2e", padx=15, pady=15)
            config_frame.pack(fill=tk.X, pady=10)

            # Slider: Canny Threshold 1
            tk.Label(config_frame, text="Sensibilidade de Borda (Canny Threshold 1):", 
                     fg="#bac2de", bg="#1e1e2e", font=("Arial", 9)).pack(anchor=tk.W, pady=(5, 0))
            
            canny1_var = tk.IntVar(value=self._vector_config["canny_threshold1"])
            
            def on_canny1_change(val):
                self._vector_config["canny_threshold1"] = int(val)
                self._save_vector_config()
            
            canny1_slider = tk.Scale(config_frame, from_=5, to=100, orient=tk.HORIZONTAL,
                                    variable=canny1_var, command=on_canny1_change,
                                    bg="#1e1e2e", fg="#cdd6f4", troughcolor="#313244",
                                    highlightthickness=0, font=("Arial", 9))
            canny1_slider.pack(fill=tk.X, pady=5)
            tk.Label(config_frame, text="Menor = mais sensível | Maior = menos sensível",
                     fg="#6c7086", bg="#1e1e2e", font=("Arial", 8)).pack(anchor=tk.W)

            # Slider: Canny Threshold 2
            tk.Label(config_frame, text="Sensibilidade de Borda (Canny Threshold 2):", 
                     fg="#bac2de", bg="#1e1e2e", font=("Arial", 9)).pack(anchor=tk.W, pady=(15, 0))
            
            canny2_var = tk.IntVar(value=self._vector_config["canny_threshold2"])
            
            def on_canny2_change(val):
                self._vector_config["canny_threshold2"] = int(val)
                self._save_vector_config()
            
            canny2_slider = tk.Scale(config_frame, from_=20, to=200, orient=tk.HORIZONTAL,
                                    variable=canny2_var, command=on_canny2_change,
                                    bg="#1e1e2e", fg="#cdd6f4", troughcolor="#313244",
                                    highlightthickness=0, font=("Arial", 9))
            canny2_slider.pack(fill=tk.X, pady=5)
            tk.Label(config_frame, text="Menor = mais bordas | Maior = menos bordas",
                     fg="#6c7086", bg="#1e1e2e", font=("Arial", 8)).pack(anchor=tk.W)

            # Slider: Densidade de Vetores
            tk.Label(config_frame, text="Densidade de Vetores (Amostragem):", 
                     fg="#bac2de", bg="#1e1e2e", font=("Arial", 9)).pack(anchor=tk.W, pady=(15, 0))
            
            density_var = tk.IntVar(value=self._vector_config["vector_density"])
            
            def on_density_change(val):
                self._vector_config["vector_density"] = int(val)
                self._save_vector_config()
            
            density_slider = tk.Scale(config_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                     variable=density_var, command=on_density_change,
                                     bg="#1e1e2e", fg="#cdd6f4", troughcolor="#313244",
                                     highlightthickness=0, font=("Arial", 9))
            density_slider.pack(fill=tk.X, pady=5)
            tk.Label(config_frame, text="Menor = mais detalhado | Maior = mais suave",
                     fg="#6c7086", bg="#1e1e2e", font=("Arial", 8)).pack(anchor=tk.W)

            # Slider: Consistência (Filtro Anti-Piscamento)
            tk.Label(config_frame, text="Consistência (Filtro Anti-Piscamento):", 
                     fg="#bac2de", bg="#1e1e2e", font=("Arial", 9)).pack(anchor=tk.W, pady=(15, 0))
            
            consistency_var = tk.IntVar(value=self._vector_config.get("consistency_threshold", 30))
            
            def on_consistency_change(val):
                self._vector_config["consistency_threshold"] = int(val)
                self._save_vector_config()
            
            consistency_slider = tk.Scale(config_frame, from_=20, to=100, orient=tk.HORIZONTAL,
                                           variable=consistency_var, command=on_consistency_change,
                                           bg="#1e1e2e", fg="#cdd6f4", troughcolor="#313244",
                                           highlightthickness=0, font=("Arial", 9))
            consistency_slider.pack(fill=tk.X, pady=5)
            tk.Label(config_frame, text="Menor = mais sensível | Maior = mais estável (filtra piscamento)",
                     fg="#6c7086", bg="#1e1e2e", font=("Arial", 8)).pack(anchor=tk.W)

            # Slider: Espessura dos Contornos
            tk.Label(config_frame, text="Espessura dos Contornos:", 
                     fg="#bac2de", bg="#1e1e2e", font=("Arial", 9)).pack(anchor=tk.W, pady=(15, 0))
            
            thickness_var = tk.IntVar(value=self._vector_config.get("edge_thickness", 1))
            
            def on_thickness_change(val):
                self._vector_config["edge_thickness"] = int(val)
                self._save_vector_config()
            
            thickness_slider = tk.Scale(config_frame, from_=1, to=5, orient=tk.HORIZONTAL,
                                        variable=thickness_var, command=on_thickness_change,
                                        bg="#1e1e2e", fg="#cdd6f4", troughcolor="#313244",
                                        highlightthickness=0, font=("Arial", 9))
            thickness_slider.pack(fill=tk.X, pady=5)
            tk.Label(config_frame, text="1 = fino | 5 = grosso",
                     fg="#6c7086", bg="#1e1e2e", font=("Arial", 8)).pack(anchor=tk.W)

            # Slider: Tamanho das Partículas
            tk.Label(config_frame, text="Tamanho das Partículas:", 
                     fg="#bac2de", bg="#1e1e2e", font=("Arial", 9)).pack(anchor=tk.W, pady=(15, 0))
            
            particle_var = tk.IntVar(value=self._vector_config.get("particle_size", 1))
            
            def on_particle_change(val):
                self._vector_config["particle_size"] = int(val)
                self._save_vector_config()
            
            particle_slider = tk.Scale(config_frame, from_=1, to=3, orient=tk.HORIZONTAL,
                                      variable=particle_var, command=on_particle_change,
                                      bg="#1e1e2e", fg="#cdd6f4", troughcolor="#313244",
                                      highlightthickness=0, font=("Arial", 9))
            particle_slider.pack(fill=tk.X, pady=5)
            tk.Label(config_frame, text="1 = pequeno | 3 = grande",
                     fg="#6c7086", bg="#1e1e2e", font=("Arial", 8)).pack(anchor=tk.W)

            # Checkbox: Modo de Contorno
            tk.Label(config_frame, text="Modo de Contorno:", 
                     fg="#bac2de", bg="#1e1e2e", font=("Arial", 9)).pack(anchor=tk.W, pady=(15, 0))
            
            contour_mode_var = tk.StringVar(value=self._vector_config.get("contour_mode", "center"))
            
            def on_contour_mode_change(val):
                self._vector_config["contour_mode"] = val
                self._save_vector_config()
            
            contour_frame = tk.Frame(config_frame, bg="#1e1e2e")
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

            # Checkbox: Projetar Apenas em Cores Verde e Azul
            tk.Label(config_frame, text="Filtro de Cores:", 
                     fg="#bac2de", bg="#1e1e2e", font=("Arial", 9)).pack(anchor=tk.W, pady=(15, 0))
            
            ignore_color_var = tk.BooleanVar(value=self._vector_config.get("ignore_green_blue", False))
            
            def on_ignore_color_change():
                self._vector_config["ignore_green_blue"] = ignore_color_var.get()
                self._save_vector_config()
            
            tk.Checkbutton(config_frame, text="Projetar apenas em bordas verdes e azuis",
                         variable=ignore_color_var, command=on_ignore_color_change,
                         bg="#1e1e2e", fg="#cdd6f4", selectcolor="#313244",
                         activebackground="#1e1e2e", activeforeground="#cdd6f4",
                         font=("Arial", 9)).pack(anchor=tk.W, pady=5)
            tk.Label(config_frame, text="Garante que as cores projetadas sejam as mesmas ignoradas",
                     fg="#6c7086", bg="#1e1e2e", font=("Arial", 8)).pack(anchor=tk.W)

            # Botão de reset
            def reset_config():
                self._vector_config = {
                    "canny_threshold1": 20,
                    "canny_threshold2": 60,
                    "vector_density": 2,
                    "consistency_threshold": 30,
                    "edge_thickness": 1,
                    "contour_mode": "center",
                    "particle_size": 1,
                    "ignore_green_blue": False
                }
                self._save_vector_config()
                # Atualizar sliders
                canny1_var.set(self._vector_config["canny_threshold1"])
                canny2_var.set(self._vector_config["canny_threshold2"])
                density_var.set(self._vector_config["vector_density"])
                consistency_var.set(self._vector_config["consistency_threshold"])
                thickness_var.set(self._vector_config["edge_thickness"])
                particle_var.set(self._vector_config["particle_size"])
                contour_mode_var.set(self._vector_config["contour_mode"])
                ignore_color_var.set(self._vector_config["ignore_green_blue"])
                messagebox.showinfo("Reset", "Configurações restauradas para os valores padrão.", parent=root)

            tk.Button(root, text="🔄 RESTAURAR PADRÕES", bg="#f38ba8", fg="white",
                      font=("Arial", 10, "bold"), relief=tk.FLAT,
                      command=reset_config, cursor="hand2").pack(fill=tk.X, pady=(15, 5))

            # Informações
            info_frame = tk.LabelFrame(root, text="ℹ️ Informações", 
                                      font=("Arial", 10, "bold"),
                                      fg="#cdd6f4", bg="#1e1e2e", padx=15, pady=10)
            info_frame.pack(fill=tk.X, pady=10)
            
            info_text = "Este efeito detecta bordas do vídeo em\ntempo real usando detecção de bordas Canny.\n\nFiltro Anti-Piscamento: Elimina traços\nque aparecem e desaparecem rapidamente.\nAumente a consistência para mais estabilidade.\n\nAs configurações são salvas automaticamente\nno arquivo vector_config.json"
            tk.Label(info_frame, text=info_text, fg="#bac2de", bg="#1e1e2e",
                    font=("Arial", 9), justify=tk.LEFT).pack(anchor=tk.W)

        run_tk()

    def _handle_config_panel(self):
        roi, default_name, box, p_color = None, "", None, None
        if self.selected_tracker is not None:
            tid = self.selected_tracker
            with self._trackers_lock:
                if tid in self.trackers:
                    tdata = self.trackers[tid]
                    box = tdata["last_box"]
                    tx, ty, tw, th = box
                    roi = self._clean_frame[ty:ty+th, tx:tx+tw].copy()
                    default_name = tdata["label"]
                    p_color = tdata["color"]
        self._launch_config_panel(roi, default_name, box, p_color, self.selected_tracker)

    def _handle_exit(self):
        print("\n[Encerrando] Finalizando todos os processos...")
        self.running = False
        try:
            self.cap.release()
            cv2.destroyAllWindows()
            self.root.quit()
        except: pass
        sys.exit(0) # Força a morte do processo e de todas as threads (IA, Memória, etc)


    def _finish_roi_selection(self):
        if self.roi_start is None or self.roi_end is None:
            return
        x1, y1 = self.roi_start
        x2, y2 = self.roi_end
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)

        if w > 15 and h > 15 and hasattr(self, '_clean_frame'):
            # --- ALINHAMENTO MAGNÉTICO (AUTO-CROPPING) ---
            # Se o usuário desenhou uma caixa gigante cheia de fundo, a IA magnética localiza
            # o objeto lá dentro por meio da distinção de luz e sombra antes de fixar o rastreador
            roi = self._clean_frame[y:y+h, x:x+w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            if cv2.countNonZero(thresh) > (w * h) * 0.6: 
                _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                best_c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(best_c) > 20: 
                    bx, by, bw, bh = cv2.boundingRect(best_c)
                    # Aperta a caixa solta perfeitamente contra as bordas do objeto real
                    x, y = x + bx, y + by
                    w, h = bw, bh

            # CSRT é muito inteligente e suporta a nova caixa ultra precisa
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(self._clean_frame, (x, y, w, h))
            
            color = (
                int(np.random.randint(80, 255)),
                int(np.random.randint(80, 255)),
                int(np.random.randint(80, 255))
            )
            with self._trackers_lock:
                self.trackers[self._next_id] = {
                    "tracker": tracker,
                    "label": f"Obj-{self._next_id}",
                    "color": color,
                    "last_box": (x, y, w, h),
                    "type": "manual"
                }
                self._next_id += 1
            print(f"[Tracker] Novo rastreador adicionado (ID: {self._next_id - 1})")

    def _start_calibration(self):
        w, h = 640, 480 
        mg = 0.0 # BORDAS EXATAS como solicitado
        self.calib_pts_proj = [
            (0, 0),             # 1: Top-Left
            (640, 0),           # 2: Top-Right
            (640, 480),         # 3: Bottom-Right
            (0, 480),           # 4: Bottom-Left
        ]
        self.calib_pts_cam = []
        self.calib_step = 0
        self.calibrating = True
        print("[Calibracao] Iniciada! MARQUE AS BORDAS NO MONITOR.")

    def _update_trackers(self, frame):
        dets = []
        to_remove = []
        with self._trackers_lock:
            for tid, data in list(self.trackers.items()):
                success, box = data["tracker"].update(frame)
                if success:
                    x, y, w, h = [int(v) for v in box]
                    
                    # Verificar qualidade do rastreamento (filtro de estabilidade)
                    if tid not in self._tracker_quality:
                        self._tracker_quality[tid] = []
                    self._tracker_quality[tid].append((x, y, w, h))
                    
                    # Manter apenas últimos 10 frames
                    if len(self._tracker_quality[tid]) > 10:
                        self._tracker_quality[tid].pop(0)
                    
                    # Calcular variação da posição (oscilação)
                    if len(self._tracker_quality[tid]) >= 3:
                        recent = self._tracker_quality[tid][-3:]
                        # Calcular desvio padrão da posição
                        xs = [p[0] for p in recent]
                        ys = [p[1] for p in recent]
                        ws = [p[2] for p in recent]
                        hs = [p[3] for p in recent]
                        
                        std_x = np.std(xs) if len(xs) > 1 else 0
                        std_y = np.std(ys) if len(ys) > 1 else 0
                        std_w = np.std(ws) if len(ws) > 1 else 0
                        std_h = np.std(hs) if len(hs) > 1 else 0
                        
                        # Se oscilação for muito alta, usar média dos últimos frames
                        if std_x > 15 or std_y > 15 or std_w > 10 or std_h > 10:
                            x = int(np.mean(xs))
                            y = int(np.mean(ys))
                            w = int(np.mean(ws))
                            h = int(np.mean(hs))
                    
                    # Verificar se o tamanho mudou drasticamente (indica instabilidade)
                    prev_box = data["last_box"]
                    size_change = abs(w - prev_box[2]) + abs(h - prev_box[3])
                    if size_change > 50:  # Mudança muito drástica
                        # Usar tamanho suavizado
                        w = int((w + prev_box[2]) / 2)
                        h = int((h + prev_box[3]) / 2)
                    
                    data["last_box"] = (x, y, w, h)
                    dets.append({"id": tid, "box": (x, y, w, h), "label": data["label"],
                                 "color": data["color"], "type": data.get("type", "manual")})
                else:
                    if self.selected_tracker == tid:
                        self.selected_tracker = None
                    to_remove.append(tid)
                    # Limpar histórico de qualidade
                    if tid in self._tracker_quality:
                        del self._tracker_quality[tid]
            for tid in to_remove:
                del self.trackers[tid]
        return dets

    def _launch_config_panel(self, roi, default_name, box, p_color, target_tid):
        try:
            from PIL import Image, ImageTk
            HAS_PIL = True
        except ImportError:
            HAS_PIL = False

        def run_tk():
            root = tk.Toplevel(self.root)
            root.title("Painel de Configuracoes")
            root.geometry("650x450")
            root.attributes("-topmost", True)
            
            try:
                from tkinter import ttk
                style = ttk.Style()
                if 'clam' in style.theme_names():
                    style.theme_use('clam')
            except:
                pass
                
            left_frame = tk.Frame(root, width=300, padx=20, pady=20)
            left_frame.pack(side=tk.LEFT, fill=tk.Y)
            
            right_frame = tk.Frame(root, width=350, padx=20, pady=20)
            right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

            # LADO ESQUERDO: ITEM ATUAL
            tk.Label(left_frame, text="Rastreador Selecionado", font=("Arial", 12, "bold")).pack(pady=(0, 10))
            
            if roi is not None and roi.size > 0:
                if HAS_PIL:
                    # Mostrar Imagem Recortada
                    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    im = Image.fromarray(rgb_roi)
                    im.thumbnail((150, 150))
                    photo = ImageTk.PhotoImage(image=im)
                    img_lbl = tk.Label(left_frame, image=photo)
                    img_lbl.image = photo 
                    img_lbl.pack(pady=5)
                else:
                    tk.Label(left_frame, text="[ PIL nao instalado para ver Recorte ]", fg="gray").pack(pady=20)

                tk.Label(left_frame, text="Nome do Objeto:").pack(anchor=tk.W)
                name_entry = tk.Entry(left_frame, width=30)
                name_entry.insert(0, default_name)
                name_entry.pack(pady=5)

                def save_memory():
                    name = name_entry.get().strip()
                    if name:
                        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                        bgModel = np.zeros((1, 65), np.float64)
                        fgModel = np.zeros((1, 65), np.float64)
                        rect = (1, 1, max(1, roi.shape[1]-2), max(1, roi.shape[0]-2))
                        try:
                            cv2.grabCut(roi, mask, rect, bgModel, fgModel, 3, cv2.GC_INIT_WITH_RECT)
                            mask_obj = np.where((mask==2)|(mask==0), 0, 255).astype('uint8')
                            if cv2.countNonZero(mask_obj) < 15: raise Exception()
                        except:
                            mask_obj = np.zeros(roi.shape[:2], dtype=np.uint8)
                            r = max(2, int(min(roi.shape[1], roi.shape[0]) * 0.15))
                            cv2.circle(mask_obj, (roi.shape[1]//2, roi.shape[0]//2), r, 255, -1)
                            
                        hist = cv2.calcHist([hsv], [0, 1], mask_obj, [16, 16], [0, 180, 0, 256])
                        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                        
                        # INTELIGÊNCIA: Extração de Features Visuais (ORB)
                        orb = cv2.ORB_create(nfeatures=300)
                        kp, des = orb.detectAndCompute(roi, None)
                        
                        with self._memory_lock:
                            self.memory_profiles.append({
                                "name": name,
                                "hist": hist,
                                "des": des,
                                "w": box[2],
                                "h": box[3],
                                "color": Config.MAGENTA
                            })
                        
                        with self._trackers_lock:
                            if target_tid in self.trackers:
                                self.trackers[target_tid]["label"] = name
                                self.trackers[target_tid]["color"] = Config.MAGENTA
                                self.trackers[target_tid]["type"] = "custom"
                        
                        refresh_list()
                        messagebox.showinfo("Sucesso", f"Item '{name}' salvo no Banco de Memorias!", parent=root)

                tk.Button(left_frame, text="Salvar Identidade Inteligente (IA/Textura)", bg="#4CAF50", fg="white", 
                          command=save_memory, relief=tk.FLAT, font=("Arial", 9, "bold")).pack(pady=10, fill=tk.X)
                
                def delete_tracker():
                    with self._trackers_lock:
                        if target_tid in self.trackers:
                            del self.trackers[target_tid]
                            if self.selected_tracker == target_tid:
                                self.selected_tracker = None
                    root.destroy()
                    
                tk.Button(left_frame, text="Apagar da Tela (Excluir Rastreador)", bg="#ff9800", fg="white",
                          command=delete_tracker, relief=tk.FLAT).pack(pady=5, fill=tk.X)
            else:
                tk.Label(left_frame, text="Nenhum Objeto Selecionado.\n\nClique no quadrado de\num objeto rastreado\nna camera primeiro.", fg="gray").pack(pady=50)

            # LADO DIREITO: BANCO DE DADOS
            tk.Label(right_frame, text="Banco de Memorias (Imortais)", font=("Arial", 12, "bold")).pack(pady=(0, 10))
            
            listbox = tk.Listbox(right_frame, height=15, selectbackground="#03A9F4", selectforeground="white", font=("Courier", 10))
            listbox.pack(fill=tk.BOTH, expand=True)
            
            def refresh_list():
                listbox.delete(0, tk.END)
                with self._memory_lock:
                    for idx, p in enumerate(self.memory_profiles):
                        listbox.insert(tk.END, f" {idx+1} | {p['name']}")
                        
            refresh_list()
            
            def delete_memory():
                sel = listbox.curselection()
                if sel:
                    idx = sel[0]
                    with self._memory_lock:
                        nome_apagado = self.memory_profiles[idx]['name']
                        del self.memory_profiles[idx]
                    refresh_list()
                else:
                    messagebox.showwarning("Aviso", "Selecione um item na lista para apagar.", parent=root)

            tk.Button(right_frame, text="Apagar da Memoria Global", bg="#F44336", fg="white",
                      command=delete_memory, relief=tk.FLAT, font=("Arial", 9, "bold")).pack(pady=10, fill=tk.X)

        run_tk()

    def _launch_camera_panel(self):
        import socket
        def get_local_ip():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                s.close()
                return ip
            except: return "192.168.1.1"

        def scan_local_cameras(listbox):
            listbox.delete(0, tk.END)
            listbox.insert(tk.END, "Buscando câmeras USB...")
            def work():
                found = []
                for i in range(5):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        found.append(str(i))
                        cap.release()
                listbox.delete(0, tk.END)
                if found:
                    for c in found: listbox.insert(tk.END, f"Câmera USB/Integrada ID: {c}")
                else: listbox.insert(tk.END, "Nenhuma câmera USB encontrada.")
            threading.Thread(target=work, daemon=True).start()

        def scan_network_radar(listbox, status_label):
            status_label.config(text="Radar Ativo: Escaneando REDE (0.x e 1.x)...", fg="blue")
            found = []
            def check_ip(ip):
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(0.3) # Timeout maior para redes lentas
                    if s.connect_ex((ip, 8080)) == 0: found.append(f"http://{ip}:8080/video")
                    s.close()
                except: pass
            
            # Gerar lista total de IPs para scan
            ips = [f"192.168.0.{i}" for i in range(1, 255)] + [f"192.168.1.{i}" for i in range(1, 255)]
            
            # Rodar em chunks para não saturar o sistema
            for i in range(0, len(ips), 50):
                threads = [threading.Thread(target=check_ip, args=(ip,), daemon=True) for ip in ips[i:i+50]]
                for t in threads: t.start()
                for t in threads: t.join(timeout=0.2)
            
            listbox.delete(0, tk.END)
            if found:
                for url in found: listbox.insert(tk.END, url)
                status_label.config(text=f"Radar: Sucesso! {len(found)} dispositivo(s).", fg="green")
            else: status_label.config(text="Radar falhou. O celular está no Wi-Fi correto?", fg="red")

        def run_tk():
            root = tk.Toplevel(self.root)
            root.title("ASSISTENTE DE CÂMERA")
            root.geometry("600x550")
            root.attributes("-topmost", True)
            root.configure(padx=20, pady=20)

            tk.Label(root, text="🔍 CONFIGURAR FONTE DE VÍDEO", font=("Arial", 14, "bold")).pack(pady=10)

            # --- SEÇÃO 1: CÂMERAS USB ---
            f_usb = tk.LabelFrame(root, text="Câmeras do Computador (USB / Integrada)", padx=10, pady=10)
            f_usb.pack(fill=tk.X, pady=5)
            
            l_usb = tk.Listbox(f_usb, height=3)
            l_usb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            def select_usb():
                sel = l_usb.get(l_usb.curselection())
                if "ID:" in sel: apply_camera(sel.split("ID: ")[1])

            btn_f_usb = tk.Button(f_usb, text="Procurar", command=lambda: scan_local_cameras(l_usb))
            btn_f_usb.pack(padx=5)
            tk.Button(f_usb, text="Usar Esta", bg="#4CAF50", fg="white", command=select_usb).pack(padx=5, pady=2)

            # --- SEÇÃO 2: IP WEBCAM (CELULAR) ---
            f_ip = tk.LabelFrame(root, text="Celular (IP Webcam via Wi-Fi)", padx=10, pady=10)
            f_ip.pack(fill=tk.BOTH, expand=True, pady=5)
            
            tk.Label(f_ip, text="Digite o endereço que aparece no seu celular:").pack(anchor=tk.W)
            e_url = tk.Entry(f_ip, font=("Consolas", 11), width=45)
            e_url.insert(0, str(Config.CAMERA_ID) if "http" in str(Config.CAMERA_ID) else "http://192.168.1.XX:8080/video")
            e_url.pack(pady=5, fill=tk.X)
            
            def apply_camera(url):
                url = str(url).strip()
                if url.isdigit(): url = int(url)
                else:
                    if not url.startswith("http"): url = "http://" + url
                    if ":8080" in url and not url.endswith("/video"): url += "/video"
                
                print(f"[Sistema] Conectando: {url}")
                new_cap = cv2.VideoCapture(url)
                if new_cap.isOpened():
                    old_cap = self.cap
                    self.cap = new_cap
                    old_cap.release()
                    Config.CAMERA_ID = url
                    print(f"[Sucesso] Conectado: {url}")
                    # Chamada do messagebox movida para o contexto da janela
                    def show_win_msg(): 
                        messagebox.showinfo("Sucesso", f"Câmera conectada:\n{url}", parent=root)
                        root.destroy()
                    root.after(0, show_win_msg)
                else:
                    def show_err():
                         messagebox.showerror("Erro", f"Falha ao abrir:\n{url}", parent=root)
                    root.after(0, show_err)

            tk.Button(f_ip, text="CONECTAR CELULAR", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                      command=lambda: apply_camera(e_url.get())).pack(pady=5, fill=tk.X)

            # Radar de rede
            status_radar = tk.Label(f_ip, text="Ou tente o Radar Automático:", fg="gray")
            status_radar.pack()
            l_radar = tk.Listbox(f_ip, height=3)
            l_radar.pack(fill=tk.BOTH, expand=True)
            
            def select_radar():
                if l_radar.curselection(): apply_camera(l_radar.get(l_radar.curselection()))

            tk.Button(f_ip, text="📡 PROCURAR CELULAR NA REDE (SCAN)", command=lambda: scan_network_radar(l_radar, status_radar)).pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Button(f_ip, text="Usar Radar", command=select_radar).pack(side=tk.LEFT, padx=5)

        run_tk()

    def _memory_loop(self):
        """Scanner passivo e Verificador de Fantasmas"""
        while self.running:
            time.sleep(0.15)
            if not hasattr(self, '_clean_frame') or self._clean_frame is None:
                continue
                
            with self._memory_lock:
                profiles = list(self.memory_profiles)
            
            if not profiles:
                continue

            frame = self._clean_frame.copy()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            with self._trackers_lock:
                current_boxes = []
                for d in self.trackers.values():
                    current_boxes.append(d["last_box"])

            for p in profiles:
                # Radar de Histograma de Cores
                backproj = cv2.calcBackProject([hsv], [0, 1], p["hist"], [0, 180, 0, 256], 1)
                
                # Threshold abaixado para perdoar 'motion blur' de objetos muito rápidos
                _, thresh = cv2.threshold(backproj, 40, 255, cv2.THRESH_BINARY)
                
                # Filtrar sujeira
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

                # 1. Verificador de Fantasma (O rastreador ficou preso na parede?)
                already_tracked_tid = None
                tx, ty, tw, th = 0, 0, 0, 0
                with self._trackers_lock:
                    for tid, d in self.trackers.items():
                        if d.get("type") == "custom" and d["label"] == p["name"]:
                            already_tracked_tid = tid
                            tx, ty, tw, th = d["last_box"]
                            break
                            
                if already_tracked_tid is not None:
                    # Analisar o quadrado real onde o objeto "falso" ou verdadeiro está
                    tx_c = max(0, min(tx, thresh.shape[1]-1))
                    ty_c = max(0, min(ty, thresh.shape[0]-1))
                    # Area validada
                    roi_thresh = thresh[ty_c:ty_c+th, tx_c:tx_c+tw]
                    if roi_thresh.size > 10:
                        density = cv2.countNonZero(roi_thresh) / (tw * th)
                        # Se não há nem 3% da cor alvo viva ali dentro, ele descolou com movimento brusco!
                        if density < 0.03:
                            with self._trackers_lock:
                                if already_tracked_tid in self.trackers:
                                    del self.trackers[already_tracked_tid]
                            print(f"[Memoria] Fantasma na Parede Excluido! '{p['name']}' descolado.")
                    
                    # Continua próximo profile porque ele ou foi deletado agora e vai nascer na proxima rodada,
                    # ou ele é verdadeiro e nao precisa ser escaneado de novo
                    continue 

                # 2. Scanner de Redeploy (Achar onde ele está na cena)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Achar a maior mancha válida de cor
                valid_cnts = []
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    area = w * h
                    p_area = p["w"] * p["h"]
                    
                    # INTELIGÊNCIA: Ignora completamente a forma (w e h estritos)
                    # Concentra apenas se o volume da cor corresponde vagamente (de 15% a 8x do tamanho original)
                    if area > p_area * 0.15 and area < p_area * 8:
                        density = cv2.countNonZero(thresh[y:y+h, x:x+w]) / area
                        if density > 0.08: # Baixado para suportar forte motion blur (movimentos rápidos)
                            valid_cnts.append((x, y, w, h, area))
                
                if valid_cnts:
                    best_box = None
                    has_features = "des" in p and p["des"] is not None and len(p["des"]) > 5
                    
                    if has_features:
                        # Extrair assinatura da textura para validar 
                        orb = cv2.ORB_create(nfeatures=200)
                        try:
                            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        except:
                            bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
                            
                        best_score = -1
                        for cand in valid_cnts:
                            cx, cy, cw, ch, carea = cand
                            roi_cand = self._clean_frame[cy:cy+ch, cx:cx+cw]
                            kp_cand, des_cand = orb.detectAndCompute(roi_cand, None)
                            
                            if des_cand is not None and len(des_cand) > 0:
                                matches = bf.match(p["des"], des_cand)
                                good = [m for m in matches if m.distance < 60]
                                if len(good) > best_score:
                                    best_score = len(good)
                                    best_box = cand
                                    
                        # Se não encontrar correlação forte de textura, usa a maior área colorida como Fallback
                        if best_score < 3:
                            valid_cnts.sort(key=lambda item: item[4], reverse=True)
                            best_box = valid_cnts[0]
                    else:
                        # Objeto de cor sólida e plana (sem ID de textura ORB salvo)
                        valid_cnts.sort(key=lambda item: item[4], reverse=True)
                        best_box = valid_cnts[0]
                        
                    x, y, w, h, area = best_box
                    
                    cx, cy = x + w//2, y + h//2
                    is_dup = False
                    
                    # Verifica se ele já está rastreado cruzadamente para evitar sombras dupes
                    for tb in current_boxes:
                        tcx, tcy = tb[0] + tb[2]//2, tb[1] + tb[3]//2
                        if abs(cx - tcx) < 80 and abs(cy - tcy) < 80:
                            is_dup = True
                            break
                    
                    if not is_dup:
                        with self._trackers_lock:
                            # Reconstruir usando CSRT sempre em vez de MOSSE. MOSSE perde facilmente na deformação.
                            tracker = cv2.legacy.TrackerCSRT_create()
                            tracker.init(self._clean_frame, (x, y, w, h))
                            self.trackers[self._next_id] = {
                                "tracker": tracker,
                                "label": p["name"],
                                "color": p["color"],
                                "last_box": (x, y, w, h),
                                "type": "custom",
                                "miss_count": 0
                            }
                            self._next_id += 1
                            current_boxes.append((x,y,w,h)) 
                        print(f"[Memoria AI] Item IDENTIFICADO (Textura/Volume): '{p['name']}' !")

    def _yolo_loop(self):
        """Thread de IA. Roda YOLO continuamente no frame mais recente.
        Nunca bloqueia o loop principal porque PyTorch libera o GIL."""
        # Carregar modelo na thread (evita travar a janela)
        if self.yolo_model is None:
            print("[IA] Carregando YOLOv8n...")
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n.pt')
            print("[IA] Pronto! Deteccao ativa.")

        while self.running and self.yolo_enabled:
            with self._yolo_lock:
                frame = self._yolo_frame
            if frame is None:
                time.sleep(0.05)
                continue

            try:
                # Aumentar confiança e adicionar NMS mais agressivo
                results = self.yolo_model(frame, verbose=False, imgsz=320, conf=0.5, iou=0.4)
                
                # Caixas puras do YOLO
                yolo_boxes = []
                # Classes relevantes para rastreamento (reduzir falsos positivos)
                relevant_classes = {'person', 'hand', 'cell phone', 'bottle', 'cup', 'laptop', 'book'}
                
                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = self.yolo_model.names[cls]
                        
                        # Filtrar apenas classes relevantes
                        if label not in relevant_classes:
                            continue
                            
                        b = box.xywh[0].cpu().numpy()
                        # Aumentar tamanho mínimo para reduzir falsos positivos pequenos
                        if int(b[2]) >= 30 and int(b[3]) >= 30:
                            yolo_boxes.append({
                                "x": int(b[0] - b[2]/2), "y": int(b[1] - b[3]/2),
                                "w": int(b[2]), "h": int(b[3]),
                                "label": label, "conf": conf,
                                "matched": False
                            })
                            
                with self._trackers_lock:
                    # 1. Resetar status de match na IA para esta rodada
                    for tdata in self.trackers.values():
                        if tdata.get("type") == "auto":
                            tdata["matched_this_frame"] = False

                    # 2. Checar YOLO boxes contra rastreadores atuais
                    for yb in yolo_boxes:
                        ycx, ycy = yb["x"] + yb["w"]//2, yb["y"] + yb["h"]//2
                        for tid, tdata in self.trackers.items():
                            tx, ty, tw, th = tdata["last_box"]
                            tcx, tcy = tx + tw//2, ty + th//2
                            
                            # Match agressivo: Se estiver perto ou dentro da tolerância do tamanho
                            if abs(tcx - ycx) < max(tw//2 + 20, 80) and abs(tcy - ycy) < max(th//2 + 20, 80):
                                yb["matched"] = True
                                if tdata.get("type") == "auto":
                                    tdata["matched_this_frame"] = True

                    # 3. Remover rastreadores auto que viraram fantasmas
                    trackers_to_delete = []
                    for tid, tdata in self.trackers.items():
                        if tdata.get("type") == "auto":
                            if tdata.get("matched_this_frame"):
                                tdata["miss_count"] = 0
                            else:
                                tdata["miss_count"] = tdata.get("miss_count", 0) + 1
                                if tdata["miss_count"] >= 2:
                                    trackers_to_delete.append(tid)

                    for tid in trackers_to_delete:
                        del self.trackers[tid]
                        print(f"[IA] Rastreador Limpo (Saiu de Cena) - ID {tid}")

                    # 4. Criar rastreadores novos com NMS nativo interno
                    valid_spawns = []
                    for yb in yolo_boxes:
                        if not yb["matched"] and yb["w"] >= 30 and yb["h"] >= 30: 
                            ycx, ycy = yb["x"] + yb["w"]//2, yb["y"] + yb["h"]//2
                            is_dup = False
                            for vs in valid_spawns:
                                vcx, vcy = vs["x"] + vs["w"]//2, vs["y"] + vs["h"]//2
                                if abs(ycx - vcx) < 100 and abs(ycy - vcy) < 100:
                                    is_dup = True
                                    break
                            
                            if not is_dup:
                                valid_spawns.append(yb)
                                self._pending_auto_trackers.append({
                                    "box": (yb["x"], yb["y"], yb["w"], yb["h"]),
                                    "label": f"{yb['label']} {yb['conf']:.0%}"
                                })

            except Exception as e:
                print(f"[YOLO] Erro: {e}")

            time.sleep(0.05)  # ~4-5 FPS de IA, sem travar nada

    def _draw_detections(self, frame, dets):
        for d in dets:
            x, y, w, h = d["box"]
            c = d["color"]
            tid = d.get("id")
            
            # Aplicar suavização à posição do rastreador (amortecer tremor)
            if tid is not None:
                if tid not in self._smoothed_positions:
                    self._smoothed_positions[tid] = {'x': x, 'y': y, 'w': w, 'h': h}
                else:
                    # Suavização exponencial mais agressiva (10% nova posição, 90% antiga)
                    smoothing_factor = 0.1
                    self._smoothed_positions[tid]['x'] = x * smoothing_factor + self._smoothed_positions[tid]['x'] * (1 - smoothing_factor)
                    self._smoothed_positions[tid]['y'] = y * smoothing_factor + self._smoothed_positions[tid]['y'] * (1 - smoothing_factor)
                    self._smoothed_positions[tid]['w'] = w * smoothing_factor + self._smoothed_positions[tid]['w'] * (1 - smoothing_factor)
                    self._smoothed_positions[tid]['h'] = h * smoothing_factor + self._smoothed_positions[tid]['h'] * (1 - smoothing_factor)
                
                # Usar posição suavizada para desenhar
                x = int(self._smoothed_positions[tid]['x'])
                y = int(self._smoothed_positions[tid]['y'])
                w = int(self._smoothed_positions[tid]['w'])
                h = int(self._smoothed_positions[tid]['h'])
            
            drawn_shape = False
            
            # --- DYNAMIC CONVEX HULL TRACKING ---
            # Identificação de Forma Pura por Separação Adaptativa (mesmo em listras/padrões heterogêneos)
            pad = max(10, int(min(w, h) * 0.15))
            y1, y2 = max(0, y-pad), min(frame.shape[0], y+h+pad)
            x1, x2 = max(0, x-pad), min(frame.shape[1], x+w+pad)
            roi_w, roi_h = x2 - x1, y2 - y1
            
            if roi_w > 5 and roi_h > 5:
                roi = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # Otsu recorta o objeto central da "mesa" automaticamente (ignorando listras e cor)
                _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                if cv2.countNonZero(thresh) > (roi_w * roi_h) * 0.6:
                    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Fechar áreas finas (como cabos de canetas e falhas listradas/pontilhadas)
                k_size = max(3, int(min(roi_w, roi_h) * 0.2))
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
                
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    best_cnt = max(contours, key=cv2.contourArea)
                    
                    if cv2.contourArea(best_cnt) > 20: 
                        # Extrair o Casulo Emborrachado (Convex Hull) para englobar todas as listras/partes quebradas!
                        hull = cv2.convexHull(best_cnt)
                        
                        eps = 0.015 * cv2.arcLength(hull, True)
                        clean_hull = cv2.approxPolyDP(hull, eps, True)
                        
                        # Retornar ao plano global
                        clean_hull += np.array([[x1, y1]])
                        
                        if tid is not None and tid == self.selected_tracker:
                            cv2.drawContours(frame, [clean_hull], -1, Config.WHITE, 4)
                            cv2.drawContours(frame, [clean_hull], -1, Config.DARK, 1)
                        else:
                            cv2.drawContours(frame, [clean_hull], -1, c, 3)
                        
                        # Fixa o nome e crosshair precisamente
                        rect = cv2.minAreaRect(clean_hull)
                        cv2.polylines(frame, [np.int32(cv2.boxPoints(rect))], True, c, 1)
                        
                        pts_y = [pt[0][1] for pt in clean_hull]
                        top_y = min(pts_y) if pts_y else y
                        cx = int(rect[0][0])
                        cy = int(rect[0][1])
                        
                        cv2.putText(frame, d["label"], (x, max(20, int(top_y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)
                        cv2.drawMarker(frame, (cx, cy), c, cv2.MARKER_CROSS, 8, 1)
                        
                        d["true_cx"] = cx
                        d["true_cy"] = cy
                        drawn_shape = True

            if not drawn_shape:
                if tid is not None and tid == self.selected_tracker:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), Config.WHITE, 1)
                    cv2.rectangle(frame, (x-1, y-1), (x+w+1, y+h+1), Config.DARK, 1)

                cv2.rectangle(frame, (x, y), (x + w, y + h), c, 1)
                cv2.putText(frame, d["label"], (x, y - 3), cv2.FONT_HERSHEY_PLAIN, 0.6, c, 1)
                cx, cy = x + w // 2, y + h // 2
                cv2.drawMarker(frame, (cx, cy), c, cv2.MARKER_CROSS, 5, 1)
                d["true_cx"] = cx
                d["true_cy"] = cy

    def _draw_hud(self, frame):
        # FPS
        fps_text = f"FPS: {self._fps:.0f}"
        cv2.putText(frame, fps_text, (Config.WIDTH - 45, 10),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, Config.GREEN, 1)

        # Contagem de rastreadores
        count = len(self.trackers)
        if count > 0:
            cv2.putText(frame, f"Rastreando: {count}", (4, Config.HEIGHT - 6),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, Config.CYAN, 1)

        # Instrução de uso
        if count == 0 and not self.calibrating:
            msg = "Clique e arraste para rastrear um objeto"
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_PLAIN, 0.8, 1)
            cv2.putText(frame, msg, ((Config.WIDTH - tw) // 2, Config.HEIGHT - 6),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, (150, 150, 150), 1)

    def _draw_roi_selection(self, frame):
        if self.selecting_roi and self.roi_start and self.roi_end:
            cv2.rectangle(frame, self.roi_start, self.roi_end, Config.CYAN, 1)

    def _draw_calibration(self, monitor_frame, projector_frame):
        if not self.calibrating:
            return
        step = self.calib_step
        
        # ─── NO PROJETOR (Sempre mostra os 4 cantos para referência) ───
        # Borda e fundo preto
        cv2.rectangle(projector_frame, (0,0), (640, 480), (0,0,120), 2)
        for i, pt in enumerate(self.calib_pts_proj):
            px, py = pt
            s = 40
            color = (0, 0, 255) # Vermelho padrão
            if i == step: color = (0, 255, 255) # Amarelo para o foco atual
            
            cv2.line(projector_frame, (px-s, py), (px+s, py), color, 4)
            cv2.line(projector_frame, (px, py-s), (px, py+s), color, 4)
            cv2.putText(projector_frame, str(i+1), (px+10, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ─── NO MONITOR (Instruções) ───
        labels = ["SUPERIOR-ESQUERDO", "SUPERIOR-DIREITO", "INFERIOR-DIREITO", "INFERIOR-ESQUERDO"]
        if step < 4:
            msg = f"CALIBRANDO: Clique no CANTO {step+1} ({labels[step]}) da projecao"
            cv2.putText(monitor_frame, msg, (20, Config.HEIGHT - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Desenhar cruzes já clicadas para referência no monitor
            for pt in self.calib_pts_cam:
                cv2.circle(monitor_frame, pt, 8, (0, 255, 0), -1)
                cv2.circle(monitor_frame, pt, 10, (0, 0, 255), 2)

            if len(self.calib_pts_cam) >= 2:
                # Desenhar contorno sendo formado no monitor
                pts = np.array(self.calib_pts_cam, np.int32)
                cv2.polylines(monitor_frame, [pts], False, (0, 0, 255), 2)
                for pt in self.calib_pts_cam:
                    cv2.circle(monitor_frame, pt, 5, (0, 255, 0), -1)

    def _handle_calibration_record(self, x=None, y=None):
        if not self.calibrating:
            return

        cx, cy = None, None
        if x is not None and y is not None:
            cx, cy = x, y
        else:
            # Pegar o centro da primeira detecção (fallback automático se apertar Espaço)
            if hasattr(self, '_clean_frame') and self._clean_frame is not None:
                all_dets = self._update_trackers(self._clean_frame) + self._yolo_dets
                if all_dets:
                    box = all_dets[0]["box"]
                    cx, cy = box[0] + box[2] // 2, box[1] + box[3] // 2
        
        if cx is not None:
            self.calib_pts_cam.append((cx, cy))
            self.calib_step += 1
            print(f"[Calibracao] Ponto {self.calib_step} registrado em ({cx}, {cy})")
            if self.calib_step >= 4:
                self.calibrating = False
                print("[Calibracao] CONCLUIDA! Calculando matriz...")
                # Calcular homografia
                src = np.float32(self.calib_pts_cam)
                dst = np.float32(self.calib_pts_proj)
                H, _ = cv2.findHomography(src, dst)
                self._H = H
                try:
                    np.save('calibration.npy', self._H)
                    print("[Calibracao] Matriz salva para uso futuro (calibration.npy).")
                except Exception as e:
                    print(f"[Calibracao] Erro ao salvar matriz: {e}")
        else:
            print("[Calibracao] Erro: Clique em cima da cruz ou use um marcador.")

    def _calc_fps(self):
        now = time.perf_counter()
        self._frame_times.append(now)
        # Manter últimos 30 timestamps
        if len(self._frame_times) > 30:
            self._frame_times = self._frame_times[-30:]
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self._fps = (len(self._frame_times) - 1) / elapsed

    def run(self):
        print("="*50)
        print("  PROJECAO INTERATIVA")
        print("="*50)
        print("  Mouse: Clique e arraste para rastrear")
        print("  ESPACO: Registrar ponto de calibracao")
        print("  Q/ESC: Sair")
        print("="*50)

        while self.running:
            try:
                self.root.update()
            except: pass

            # MODO SIMPLES: apenas ler frame (sem overhead de buffer clearing)
            ret, frame = self.cap.read()
            if not ret:
                self._cam_fail_count = getattr(self, '_cam_fail_count', 0) + 1
                if self._cam_fail_count > 30 and str(Config.CAMERA_ID) != "0":
                    print("[ERRO CRÍTICO] Conexão com o Celular perdida! (Sem bateria ou Sem Rede)")
                    print("[RECOVERY] Ativando Câmera do Notebook automaticamente...")
                    self.cap.release()
                    
                    fallback_success = False
                    for cid in [0, 1, 2]:
                        for backend in [cv2.CAP_ANY, cv2.CAP_DSHOW]:
                            self.cap = cv2.VideoCapture(cid, backend)
                            if self.cap.isOpened():
                                Config.CAMERA_ID = cid
                                fallback_success = True
                                break
                        if fallback_success: break
                        
                    if fallback_success:
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.WIDTH)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.HEIGHT)
                        self.cap.set(cv2.CAP_PROP_FPS, Config.FPS)
                        print(f"[RECOVERY] Sucesso na Câmera ID {Config.CAMERA_ID}.")
                    self._cam_fail_count = 0
                time.sleep(0.01)
                continue
            else:
                self._cam_fail_count = 0

            frame = cv2.flip(frame, 1)  # ESPELHAR A CAMERA! <--
            self._clean_frame = frame  

            # ─── Converter YOLO Detections -> MOSSE Trackers ───
            with self._trackers_lock:
                while self._pending_auto_trackers:
                    p = self._pending_auto_trackers.pop(0)
                    tracker = cv2.legacy.TrackerCSRT_create()
                    tracker.init(self._clean_frame, p["box"])
                    self.trackers[self._next_id] = {
                        "tracker": tracker,
                        "label": p["label"],
                        "color": (255, 255, 0),  # Ciano forte BGR para a UI
                        "last_box": p["box"],
                        "type": "auto",
                        "miss_count": 0
                    }
                    self._next_id += 1

            # ─── Processamento Principal (CSRT) ───
            all_dets = self._update_trackers(frame)

            # Compartilhar frame independente com a thread YOLO
            with self._yolo_lock:
                self._yolo_frame = frame.copy()

            # ─── Desenhar Frames ───
            monitor_frame = frame.copy()
            projector_frame = np.zeros((480, 640, 3), dtype=np.uint8) # Sempre Preto por padrão
            
            self._draw_detections(monitor_frame, all_dets)
            self._draw_roi_selection(monitor_frame)
            self._draw_calibration(monitor_frame, projector_frame)
            self._draw_hud(monitor_frame)

            # ─── EFEITOS MODULARIZADOS ───
            if self.active_effect:
                # Inicializar canvas se necessário
                if self._trail_canvas is None:
                    self._trail_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                if getattr(self, '_cam_trail_canvas', None) is None:
                    self._cam_trail_canvas = np.zeros((480, 640, 3), dtype=np.uint8)

            # ─── EFEITO VECTORS (processa frame inteiro, fora do loop de detecções) ───
            if self.active_effect == "effect_vectors":
                effect = self.effect_loader.get_effect(self.active_effect)
                if effect:
                    effect.apply(
                        frame=self._clean_frame,
                        trail_canvas=self._trail_canvas,
                        cam_trail_canvas=self._cam_trail_canvas,
                        H=self._H,
                        yolo_enabled=self.yolo_enabled,
                        trackers=self.trackers,
                        trackers_lock=self._trackers_lock,
                        detection_data={},
                        config=self._vector_config
                    )

            # ─── Projetar Interatividade (EFEITOS) ───
            for d in all_dets:
                 tid = d["id"]
                 x, y, w, h = d["box"]
                 cx = d.get("true_cx", x + w // 2)
                 cy = d.get("true_cy", y + h // 2)
                 
                 # Registrar histórico para rastro (sem limite quando efeito ativo)
                 is_effect_active = self.active_effect in ["effect_plexus", "effect_grid", "effect_pulse", "effect_liquid", "effect_voronoi", "effect_matrix", "effect_hologram", "effect_fire", "effect_nebula", "effect_plasma", "effect_insects", "effect_vectors", "effect_colliding_balls", "effect_neon_ribbon", "effect_neural_network", "effect_lantern_cone", "effect_black_hole", "effect_boat_wake"]
                 max_hist = 500 if is_effect_active else 40
                 with self._trackers_lock:
                     if tid in self.trackers:
                         if "history" not in self.trackers[tid]: self.trackers[tid]["history"] = []
                         self.trackers[tid]["history"].append((cx, cy, w, h))  # Guarda dimensões
                         if len(self.trackers[tid]["history"]) > max_hist:
                             self.trackers[tid]["history"].pop(0)

                 if self._H is not None:
                      with self._trackers_lock:
                          if tid not in self.trackers:
                              continue
                          hist = list(self.trackers[tid].get("history", []))
                       
                      if is_effect_active and len(hist) >= 2:
                          # ─── EFEITOS FUTURE (Modularizados) ───
                          if self.active_effect in ["effect_colliding_balls", "effect_neon_ribbon", "effect_neural_network", "effect_lantern_cone", "effect_black_hole", "effect_boat_wake"]:
                              effect = self.effect_loader.get_effect(self.active_effect)
                              if effect:
                                  config = self._vector_config if self.active_effect == "effect_vectors" else {}
                                  effect.apply(
                                      frame=self._clean_frame,
                                      trail_canvas=self._trail_canvas,
                                      cam_trail_canvas=self._cam_trail_canvas,
                                      H=self._H,
                                      yolo_enabled=self.yolo_enabled,
                                      trackers=self.trackers,
                                      trackers_lock=self._trackers_lock,
                                      detection_data=d,
                                      config=config
                                  )
                          # ─── EFEITOS ESPECIAIS ───
                          # Inicializar canvas persistente se necessário
                          if self._trail_canvas is None:
                              self._trail_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                          if getattr(self, '_cam_trail_canvas', None) is None:
                              self._cam_trail_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                          
                          # Pegar os dois últimos pontos para desenhar o segmento novo
                          prev = hist[-2]
                          curr = hist[-1]
                          
                          p_cam = np.float32([[[prev[0], prev[1]]]])
                          c_cam = np.float32([[[curr[0], curr[1]]]])
                          p_proj = cv2.perspectiveTransform(p_cam, self._H)
                          c_proj = cv2.perspectiveTransform(c_cam, self._H)
                          
                          p1 = (int(p_proj[0][0][0]), int(p_proj[0][0][1]))
                          p2 = (int(c_proj[0][0][0]), int(c_proj[0][0][1]))
                          
                          # Coordenadas do monitor de controle PC
                          pc1 = (int(prev[0]), int(prev[1]))
                          pc2 = (int(curr[0]), int(curr[1]))
                          
                          # Calcular espessura baseada na dimensão perpendicular ao movimento
                          dx = curr[0] - prev[0]
                          dy = curr[1] - prev[1]
                          dist = max(1, (dx*dx + dy*dy) ** 0.5)
                          
                          obj_w, obj_h = curr[2], curr[3]
                          
                          if dist > 2:  # Movimento significativo
                              angle = math.atan2(abs(dy), abs(dx))
                              perp_size = abs(obj_w * math.sin(angle)) + abs(obj_h * math.cos(angle))
                              scale_x, scale_y = 640 / Config.WIDTH, 480 / Config.HEIGHT
                              base_thickness = max(3, int(perp_size * min(scale_x, scale_y) * 0.8))
                          else:
                              base_thickness = max(3, int(min(obj_w, obj_h) * 0.6))
                          
                          # Desenhar os Efeitos Específicos no Projetor e no PC
                          # ─── Efeitos de Projeção Mapeada Interativa PRO ───

                          if self.active_effect == "effect_plexus":
                              # 🕸️ PLEXUS (Conexão Neural) - Gera polígonos entre pontos distantes mas compatíveis
                              recent_hist = hist[-40:] # Até 40 pontos na memória
                              if len(recent_hist) > 4:
                                  pts_cam = np.array([[h[0], h[1]] for h in recent_hist], dtype=np.float32).reshape(-1, 1, 2)
                                  pts_proj = cv2.perspectiveTransform(pts_cam, self._H)
                                  
                                  pts_p = [(int(p[0][0]), int(p[0][1])) for p in pts_proj]
                                  pts_c = [(int(p[0][0]), int(p[0][1])) for p in pts_cam]

                                  thresh = 100 # Conecta-se se a distância for menor que 100px
                                  for i in range(max(0, len(pts_p)-15), len(pts_p)):
                                      for j in range(max(0, i-6), i):
                                          dist_pts = math.hypot(pts_p[i][0] - pts_p[j][0], pts_p[i][1] - pts_p[j][1])
                                          if dist_pts < thresh:
                                              t = 1 if dist_pts > thresh * 0.5 else 2
                                              cv2.line(self._trail_canvas, pts_p[i], pts_p[j], d["color"], t)
                                              cv2.line(self._cam_trail_canvas, pts_c[i], pts_c[j], d["color"], t)
                                              
                                              # Se muito perto, desenha pequenas facetas/triangulos
                                              if dist_pts < thresh * 0.3 and i > 2:
                                                  k = i - 2
                                                  pts_tri_proj = np.array([pts_p[i], pts_p[j], pts_p[k]], np.int32)
                                                  cv2.fillPoly(self._trail_canvas, [pts_tri_proj], (d["color"][0]//2, d["color"][1]//2, d["color"][2]//2))
                                                  pts_tri_cam = np.array([pts_c[i], pts_c[j], pts_c[k]], np.int32)
                                                  cv2.fillPoly(self._cam_trail_canvas, [pts_tri_cam], (d["color"][0]//2, d["color"][1]//2, d["color"][2]//2))

                          elif self.active_effect == "effect_grid":
                              # 🌐 MALHA GRAVITACIONAL - Deforma uma geometria de rede baseada no eixo de movimento
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
                                      
                                      cv2.circle(self._trail_canvas, (wx, wy), 2, (255, 255, 255), -1)
                                      cv2.line(self._trail_canvas, (wx, wy), (p2[0], p2[1]), d["color"], 1)
                                      
                                      # Coords na câmera
                                      cnx, cny = cgx + idx_x * grid_size, cgy + idx_y * grid_size
                                      dist_cg = math.hypot(pc2[0] - cnx, pc2[1] - cny)
                                      
                                      cwarp = max(0.0, 1.0 - dist_cg / (grid_size * 2))
                                      cwx = int(cnx + (pc2[0] - cnx) * cwarp * 0.7)
                                      cwy = int(cny + (pc2[1] - cny) * cwarp * 0.7)
                                      cv2.circle(self._cam_trail_canvas, (cwx, cwy), 2, (255, 255, 255), -1)
                                      cv2.line(self._cam_trail_canvas, (cwx, cwy), (pc2[0], pc2[1]), d["color"], 1)

                          elif self.active_effect == "effect_pulse":
                              # 📡 ONDAS CINÉTICAS E RADARES - Desloca ecos circulares/elípticos em cascata
                              if dist > 3 or random.random() > 0.8:
                                  rc = int(base_thickness * random.uniform(1.0, 3.5))
                                  # Cria bordas grossas internas
                                  cv2.circle(self._trail_canvas, p2, int(rc*0.4), d["color"], -1)
                                  cv2.circle(self._cam_trail_canvas, pc2, int(rc*0.4), d["color"], -1)
                                  
                                  # Bordas vazadas externas como ecos sonoros (cyan/magenta shifting)
                                  shift_color = (min(255, d["color"][0]+100), d["color"][1], min(255, d["color"][2]+100))
                                  cv2.circle(self._trail_canvas, p2, rc, shift_color, int(max(1, 4 - rc/10)))
                                  cv2.circle(self._cam_trail_canvas, pc2, rc, shift_color, int(max(1, 4 - rc/10)))
                                  
                                  if random.random() > 0.5:
                                      cv2.ellipse(self._trail_canvas, p2, (rc + 10, int(rc*0.3)), math.degrees(angle), 0, 360, (255, 255, 255), 1)
                                      cv2.ellipse(self._cam_trail_canvas, pc2, (rc + 10, int(rc*0.3)), math.degrees(angle), 0, 360, (255, 255, 255), 1)

                          elif self.active_effect == "effect_liquid":
                              # 💧 RASTRO LÍQUIDO - Tinta densa com espalhamento radial orgânico (Metaballs feeling)
                              num_splatters = random.randint(2, 6)
                              for _ in range(num_splatters):
                                  dist_off = random.uniform(0, base_thickness * 1.5)
                                  ang_off = random.uniform(0, math.pi * 2)
                                  lx = int(p2[0] + dist_off * math.cos(ang_off))
                                  ly = int(p2[1] + dist_off * math.sin(ang_off))
                                  lcx = int(pc2[0] + dist_off * math.cos(ang_off))
                                  lcy = int(pc2[1] + dist_off * math.sin(ang_off))
                                  rad = random.randint(3, int(base_thickness * 0.8) + 5)
                                  color_l = (min(255, d["color"][0]+30), d["color"][1], max(0, d["color"][2]-30))
                                  cv2.circle(self._trail_canvas, (lx, ly), rad, color_l, -1)
                                  cv2.circle(self._cam_trail_canvas, (lcx, lcy), rad, color_l, -1)

                          elif self.active_effect == "effect_voronoi":
                              # 💎 FRACTAL DE VIDRO - Polígonos irregulares vivos que estilhaçam
                              if dist > 2:
                                  for _ in range(2):
                                      pts_v_p = []
                                      pts_v_c = []
                                      cx_p, cy_p = p2[0], p2[1]
                                      cx_c, cy_c = pc2[0], pc2[1]
                                      for _ in range(3): # Triangulo de estilhaço
                                          r_off = random.randint(10, int(base_thickness * 2) + 20)
                                          a_off = random.uniform(0, math.pi * 2)
                                          pts_v_p.append([int(cx_p + r_off*math.cos(a_off)), int(cy_p + r_off*math.sin(a_off))])
                                          pts_v_c.append([int(cx_c + r_off*math.cos(a_off)), int(cy_c + r_off*math.sin(a_off))])
                                      
                                      pts_v_p = np.array([pts_v_p], np.int32)
                                      pts_v_c = np.array([pts_v_c], np.int32)
                                      if random.random() > 0.5:
                                          cv2.fillPoly(self._trail_canvas, pts_v_p, d["color"])
                                          cv2.fillPoly(self._cam_trail_canvas, pts_v_c, d["color"])
                                      else:
                                          cv2.polylines(self._trail_canvas, pts_v_p, True, (255, 255, 255), 1)
                                          cv2.polylines(self._cam_trail_canvas, pts_v_c, True, (255, 255, 255), 1)

                          elif self.active_effect == "effect_matrix":
                              # 📟 CHUVA MATRIX - Digital Rain escorrendo pelo caminho
                              if random.random() > 0.3:
                                  drop_len = random.randint(20, 80)
                                  rx = random.randint(-int(base_thickness), int(base_thickness))
                                  start_p = (p2[0] + rx, p2[1])
                                  end_p = (p2[0] + rx, p2[1] + drop_len)
                                  cv2.line(self._trail_canvas, start_p, end_p, (0, 255, 0), random.randint(1, 4))
                                  start_c = (pc2[0] + rx, pc2[1])
                                  end_c = (pc2[0] + rx, pc2[1] + drop_len)
                                  cv2.line(self._cam_trail_canvas, start_c, end_c, (0, 255, 0), random.randint(1, 4))

                          elif self.active_effect == "effect_hologram":
                              # 🖨️ SCANNER HOLOGRÁFICO - Linhas de grade CRT/Laser horizontais dinâmicas
                              if dist > 1:
                                  scan_w = max(50, base_thickness * 3)
                                  for y_off in range(-int(base_thickness), int(base_thickness), 8):
                                      if random.random() > 0.5:
                                          color_h = (255, 255, 0) if random.random() > 0.5 else d["color"]
                                          pt1_p = (int(p2[0] - scan_w/2), p2[1] + y_off)
                                          pt2_p = (int(p2[0] + scan_w/2), p2[1] + y_off)
                                          cv2.line(self._trail_canvas, pt1_p, pt2_p, color_h, 1)
                                          pt1_c = (int(pc2[0] - scan_w/2), pc2[1] + y_off)
                                          pt2_c = (int(pc2[0] + scan_w/2), pc2[1] + y_off)
                                          cv2.line(self._cam_trail_canvas, pt1_c, pt2_c, color_h, 1)

                          elif self.active_effect == "effect_fire":
                              # 🔥 FOGO CÓSMICO - Chamas que seguem o movimento com centro brilhante e faíscas
                              intensity = min(15, int(dist))
                              thick_fire = max(4, int(base_thickness * 1.8))
                              
                              # Núcleo superaquecido (branco/amarelo)
                              core_color = (255, 255, 255)
                              cv2.line(self._trail_canvas, p1, p2, core_color, max(2, thick_fire // 4))
                              cv2.line(self._cam_trail_canvas, pc1, pc2, core_color, max(2, thick_fire // 4))
                              
                              # Chamas e faíscas radiais
                              for _ in range(intensity + 5):
                                  t = random.random()
                                  px = int(p1[0] * (1-t) + p2[0] * t)
                                  py = int(p1[1] * (1-t) + p2[1] * t)
                                  cpx = int(pc1[0] * (1-t) + pc2[0] * t)
                                  cpy = int(pc1[1] * (1-t) + pc2[1] * t)
                                  
                                  spread = int(thick_fire * 1.5)
                                  ox = random.randint(-spread, spread)
                                  oy = random.randint(-spread, spread)
                                  
                                  # Mistura cor do objeto + fogo
                                  # (em BGR, fogo é Blue baixo, Green médio, Red alto)
                                  c_mod = (
                                      min(255, int(d["color"][0] * 0.5 + random.randint(0, 50))), 
                                      min(255, int(d["color"][1] * 0.8 + random.randint(50, 100))), 
                                      min(255, int(d["color"][2] + random.randint(100, 200)))  
                                  )
                                  r = random.randint(1, max(2, thick_fire // 3))
                                  cv2.circle(self._trail_canvas, (px+ox, py+oy), r, c_mod, -1)
                                  cv2.circle(self._cam_trail_canvas, (cpx+ox, cpy+oy), r, c_mod, -1)

                          elif self.active_effect == "effect_nebula":
                              # 🌌 NEBULOSA ESTELAR - Poeira estelar colorida e pontilhada suave
                              num_stars = random.randint(10, 25)
                              thick_neb = max(10, int(base_thickness * 2.5))
                              
                              for _ in range(num_stars):
                                  t = random.random()
                                  px = int(p1[0] * (1-t) + p2[0] * t)
                                  py = int(p1[1] * (1-t) + p2[1] * t)
                                  cpx = int(pc1[0] * (1-t) + pc2[0] * t)
                                  cpy = int(pc1[1] * (1-t) + pc2[1] * t)
                                  
                                  # Distribuição gaussiana para parecer concentração no centro
                                  ox = int(random.gauss(0, thick_neb / 2))
                                  oy = int(random.gauss(0, thick_neb / 2))
                                  
                                  col = (
                                      min(255, d["color"][0] + random.randint(0, 100)),
                                      min(255, d["color"][1] + random.randint(0, 80)),
                                      min(255, d["color"][2] + random.randint(0, 100))
                                  )
                                  
                                  # Algumas estrelas maiores e brilhantes (núcleo branco)
                                  if random.random() > 0.9:
                                      cv2.circle(self._trail_canvas, (px+ox, py+oy), random.randint(2, 4), (255, 255, 255), -1)
                                      cv2.circle(self._cam_trail_canvas, (cpx+ox, cpy+oy), random.randint(2, 4), (255, 255, 255), -1)
                                  else:
                                      cv2.circle(self._trail_canvas, (px+ox, py+oy), random.randint(1, 2), col, -1)
                                      cv2.circle(self._cam_trail_canvas, (cpx+ox, cpy+oy), random.randint(1, 2), col, -1)

                          elif self.active_effect == "effect_plasma":
                              # ⚡ RAIO PLASMÁTICO - Descargas elétricas caóticas ligando os pontos
                              if dist > 5:
                                  for _ in range(3): # Gera múltiplas ramificações
                                      pt_ant_p = p1
                                      pt_ant_c = pc1
                                      segments = int(dist // 5) + 2
                                      
                                      plasma_color = (
                                          min(255, d["color"][0] + 150),
                                          min(255, d["color"][1] + 100),
                                          min(255, d["color"][2] + 50)
                                      )
                                      
                                      for s in range(1, segments + 1):
                                          t = s / segments
                                          jitter = int(base_thickness * 1.5)
                                          
                                          if s == segments:
                                              px, py = p2[0], p2[1]
                                              cpx, cpy = pc2[0], pc2[1]
                                          else:
                                              px = int(p1[0] * (1-t) + p2[0] * t) + random.randint(-jitter, jitter)
                                              py = int(p1[1] * (1-t) + p2[1] * t) + random.randint(-jitter, jitter)
                                              cpx = int(pc1[0] * (1-t) + pc2[0] * t) + random.randint(-jitter, jitter)
                                              cpy = int(pc1[1] * (1-t) + pc2[1] * t) + random.randint(-jitter, jitter)
                                          
                                          thick_plasma = random.randint(1, max(3, int(base_thickness//2)))
                                          cv2.line(self._trail_canvas, pt_ant_p, (px, py), plasma_color, thick_plasma)
                                          cv2.line(self._cam_trail_canvas, pt_ant_c, (cpx, cpy), plasma_color, thick_plasma)
                                          
                                          # Brilho central fino (raios brancos)
                                          if random.random() > 0.5:
                                              cv2.line(self._trail_canvas, pt_ant_p, (px, py), (255, 255, 255), 1)
                                              cv2.line(self._cam_trail_canvas, pt_ant_c, (cpx, cpy), (255, 255, 255), 1)
                                          
                                          pt_ant_p = (px, py)
                                          pt_ant_c = (cpx, cpy)

                      elif len(hist) >= 2:
                          # ─── Modo padrão (sem efeito): rastro temporário fino ───
                          pts_list = [(h[0], h[1]) for h in hist]
                          pts_cam = np.array(pts_list, dtype=np.float32).reshape(-1, 1, 2)
                          pts_proj = cv2.perspectiveTransform(pts_cam, self._H)
                          
                          for i in range(len(pts_proj) - 1):
                              p1 = tuple(pts_proj[i][0].astype(int))
                              p2 = tuple(pts_proj[i+1][0].astype(int))
                              thickness = int(1 + (i / len(pts_proj)) * 4)
                              cv2.line(projector_frame, p1, p2, d["color"], thickness)

                      # Bolinha removida: o usuário não quer o rastreador exposto na vida real
                      pass

            # ─── JOGO INTERATIVO: ATAQUE DE INSETOS ───
            if self.active_effect == "effect_insects":
                if getattr(self, '_insects', None) is None:
                    self._insects = []
                    self._game_over_state = {}
                
                now = time.time()
                
                # Gerenciador global de reinicialização (Game Over compartilhado)
                if getattr(self, '_game_reset_at', 0) > 0:
                    delta = now - self._game_reset_at
                    if delta < 2.5: # 2.5 segundos de luto/tela vermelha
                        # Desenhar Tela de Reinicialização Global se alguém perdeu
                        px, py, cx, cy = self._losing_obj_pos
                        if True: # Removido pisca-pisca para teste
                            cv2.putText(projector_frame, "VOCE PERDEU!", (int(px - 110), int(py - 30)), cv2.FONT_HERSHEY_DUPLEX, 1.2, Config.RED, 4)
                            cv2.putText(projector_frame, "REINICIANDO...", (int(px - 80), int(py + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.WHITE, 2)
                            if self.yolo_enabled:
                                cv2.putText(monitor_frame, "GAME OVER", (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, Config.RED, 2)
                        
                        # IMPORTANTE: Usar continue para não fechar o app, apenas pular o resto do processamento
                        continue 
                    else:
                        self._game_reset_at = 0 # Reinicia o jogo!
                        self._insects = [] # Limpa sobreviventes do round anterior
                
                alive_dets = all_dets # No novo modo, todos estão vivos até alguém perder o round 
                
                # Spawn aleatório das bordas da área de projeção
                if alive_dets and len(self._insects) < min(15, len(alive_dets) * 5): 
                    if random.random() < 0.1: 
                        edge = random.choice(["top", "bottom", "left", "right"])
                        if edge == "top": ix, iy = random.randint(0, 640), -10
                        elif edge == "bottom": ix, iy = random.randint(0, 640), 490
                        elif edge == "left": ix, iy = -10, random.randint(0, 480)
                        else: ix, iy = 650, random.randint(0, 480)
                        self._insects.append({"x": float(ix), "y": float(iy), "s": random.uniform(2.5, 6.0)})
                        
                dead_insects = []
                for i, ins in enumerate(self._insects):
                    if not alive_dets: break
                    
                    # Procura o objeto (presa) mais próximo!
                    best_d = None
                    best_dist = 999999
                    best_px, best_py = None, None
                    
                    for d in alive_dets:
                        cx = d.get("true_cx", d["box"][0] + d["box"][2] // 2)
                        cy = d.get("true_cy", d["box"][1] + d["box"][3] // 2)
                        
                        px, py = cx, cy
                        if self._H is not None: # Distância é baseada no mundo do Projetor
                            c_cam = np.float32([[[cx, cy]]])
                            c_proj = cv2.perspectiveTransform(c_cam, self._H)[0][0]
                            px, py = c_proj[0], c_proj[1]
                            
                        dist = math.hypot(ins["x"] - px, ins["y"] - py)
                        if dist < best_dist:
                            best_dist = dist
                            best_d = d
                            best_px, best_py = px, py
                            
                    if best_d is not None:
                        angle = math.atan2(best_py - ins["y"], best_px - ins["x"])
                        wobble = math.sin(now * 30.0 + i) * 0.4 # Faz eles andarem rebolando
                        
                        ins["x"] += math.cos(angle + wobble) * ins["s"]
                        ins["y"] += math.sin(angle + wobble) * ins["s"]
                        
                        ix, iy = int(ins["x"]), int(ins["y"])
                        
                        # Projetor desenha corpo e patas
                        cv2.circle(projector_frame, (ix, iy), 5, Config.WHITE, -1)
                        cv2.circle(projector_frame, (ix, iy), 7, Config.RED, 1)
                        cv2.line(projector_frame, (ix, iy), (ix+int(math.cos(angle+1.5+wobble)*8), iy+int(math.sin(angle+1.5+wobble)*8)), Config.WHITE, 1)
                        cv2.line(projector_frame, (ix, iy), (ix+int(math.cos(angle-1.5-wobble)*8), iy+int(math.sin(angle-1.5-wobble)*8)), Config.WHITE, 1)
                        
                        # Monitor só desenha se Visão IA estiver ligada
                        if self.yolo_enabled:
                            cv2.circle(monitor_frame, (ix, iy), 3, Config.WHITE, -1)
                            cv2.circle(monitor_frame, (ix, iy), 5, Config.RED, 1)
                        
                        # Colisão! Se a aranha tocar o corpo
                        avg_radius = (best_d["box"][2] + best_d["box"][3]) / 4.0
                        if self._H is not None: avg_radius *= 640 / max(1, Config.WIDTH)
                            
                        if best_dist < max(15, avg_radius * 0.8):
                            self._game_reset_at = now
                            self._losing_obj_pos = (best_px, best_py, cx, cy) # Salva local da morte
                            dead_insects.append(i)
                            break # Para o loop de insetos pois o jogo acabou
                
                # Deleta os insetos suicidas que mataram o jogador
                for i in sorted(dead_insects, reverse=True):
                    self._insects.pop(i)
                    
                # Desenhar Tela de Reinicialização Global se alguém perdeu
                if getattr(self, '_game_reset_at', 0) > 0:
                    px, py, cx, cy = self._losing_obj_pos
                    
                    # PROJETOR: Texto Gigante com Borda Preta (Máxima visibilidade na mesa)
                    # Sombra/Borda
                    cv2.putText(projector_frame, "VOCE PERDEU!", (int(px - 110), int(py - 30)), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,0), 7)
                    cv2.putText(projector_frame, "REINICIANDO...", (int(px - 80), int(py + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 5)
                    # Cor Principal
                    cv2.putText(projector_frame, "VOCE PERDEU!", (int(px - 110), int(py - 30)), cv2.FONT_HERSHEY_DUPLEX, 1.5, Config.RED, 3)
                    cv2.putText(projector_frame, "REINICIANDO...", (int(px - 80), int(py + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, Config.WHITE, 2)
                    
                    if self.yolo_enabled:
                        cv2.putText(monitor_frame, "GAME OVER", (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4)
                        cv2.putText(monitor_frame, "GAME OVER", (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, Config.RED, 2)

            # ─── Compor Trail Canvas no Projetor e Monitor ───
            if self._trail_canvas is not None and getattr(self, '_cam_trail_canvas', None) is not None and getattr(self, 'active_effect', None) is not None:
                # Desvanecimento suave (se configurado)
                if self._trail_fade_rate > 0:
                    fade_mask = self._trail_canvas > 0
                    self._trail_canvas[fade_mask] = np.clip(
                        self._trail_canvas[fade_mask].astype(np.int16) - self._trail_fade_rate, 0, 255
                    ).astype(np.uint8)
                    
                    fade_mask_cam = self._cam_trail_canvas > 0
                    self._cam_trail_canvas[fade_mask_cam] = np.clip(
                        self._cam_trail_canvas[fade_mask_cam].astype(np.int16) - self._trail_fade_rate, 0, 255
                    ).astype(np.uint8)
                
                # Sobrepor o rastro persistente no frame do projetor
                projector_frame = cv2.add(projector_frame, self._trail_canvas)
                
                # NO MONITOR: Só mostra se a Visão IA estiver ligada para debugar, senão limpa
                if self.yolo_enabled:
                    monitor_frame = cv2.add(monitor_frame, self._cam_trail_canvas)

            # ─── FPS ───
            self._calc_fps()

            # ─── Loop de Eventos e Exibição ───
            cv2.imshow("PAINEL DE CONTROLE (PC)", monitor_frame)
            cv2.imshow("SAIDA PROJETOR", projector_frame)

            # Polling de teclado rápido para detectar o X da janela
            key = cv2.waitKey(2) & 0xFF
            
            # CHECAGEM DE MORTE TOTAL: O X da janela do OpenCV (Câmera) ou Tecla Q/ESC
            visible_pc = cv2.getWindowProperty("PAINEL DE CONTROLE (PC)", cv2.WND_PROP_VISIBLE)
            if visible_pc < 1 or key == ord('q') or key == 27:
                self._handle_exit()
                break
            
            if key == ord(' '):  # Espaço = registrar calibração
                self._handle_calibration_record()


        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n[Sistema] Encerrado.")


if __name__ == "__main__":
    app = App()
    app.run()
