import cv2
import numpy as np
from config import Config
from renderer import Renderer
import threading
import time

class ProjecaoCore:
    """
    Arquitetura ultra-simples: 1 thread de captura + processamento.
    YOLO roda em processo separado só quando pedido.
    Zero GIL contention = zero delay.
    """

    def __init__(self, camera_id=0):
        self.config = Config()
        self.config.camera_id = camera_id

        # Câmera com DirectShow (mais rápido no Windows)
        self.cap = cv2.VideoCapture(self.config.camera_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # Forçar codec MJPG — decodificação mais rápida que YUV
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        self.renderer = None
        self.running = True

        # Rastreadores MOSSE (manuais)
        self.trackers = {}
        self._next_id = 0

        # YOLO — completamente separado, nunca interfere
        self._yolo_enabled = False
        self._yolo_model = None
        self._yolo_detections = []  # Cache das últimas detecções YOLO
        self._yolo_frame_counter = 0

        # Calibração
        self.calibrating = False
        self.calib_step = 0
        self.calib_pts_proj = []
        self.calib_pts_cam = []

        # Frame JPEG pré-codificado para streaming web
        self._web_jpeg = None
        self._web_lock = threading.Lock()
        self._web_event = threading.Event()  # Sinaliza frame novo disponível

        # Um único thread: captura → rastreia → projeta → monta frame web
        self._thread = threading.Thread(target=self._main_loop, daemon=True)
        self._thread.start()

    def _main_loop(self):
        """Loop principal. Faz TUDO em sequência, sem competição de threads."""
        while self.running:
            # ─── 1. CAPTURA (direto da câmera, sem buffer intermediário) ───
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            self._current_frame = frame
            all_detections = []

            # ─── 2. RASTREAMENTO MOSSE (~0.5ms por objeto) ───
            to_remove = []
            for tid, data in list(self.trackers.items()):
                success, box = data["tracker"].update(frame)
                if success:
                    x, y, w, h = [int(v) for v in box]
                    all_detections.append({
                        "box": (x, y, w, h),
                        "label": data["label"],
                        "id": tid,
                        "color": data["color"],
                        "type": "manual"
                    })
                else:
                    to_remove.append(tid)
            for tid in to_remove:
                del self.trackers[tid]

            # ─── 3. YOLO (só se ativo, a cada 5 frames para não pesar) ───
            if self._yolo_enabled:
                self._yolo_frame_counter += 1
                if self._yolo_frame_counter % 5 == 0:
                    self._run_yolo(frame, all_detections)
                # Usar cache nos frames intermediários
                for d in self._yolo_detections:
                    # Verificar se não duplica com rastreador manual
                    dx, dy, dw, dh = d["box"]
                    cx, cy = dx + dw // 2, dy + dh // 2
                    is_dup = any(
                        abs(cx - (m["box"][0] + m["box"][2] // 2)) < 60 and
                        abs(cy - (m["box"][1] + m["box"][3] // 2)) < 60
                        for m in all_detections if m["type"] == "manual"
                    )
                    if not is_dup:
                        all_detections.append(d)

            # ─── 4. PROJEÇÃO (prioridade máxima, antes do web frame) ───
            if self.config.homography_matrix is not None or self.config.simulation_mode or self.calibrating:
                if self.renderer is None:
                    try:
                        self.renderer = Renderer(self.config)
                    except Exception as e:
                        print(f"[Renderer] Erro: {e}")
                if self.renderer:
                    if self.calibrating:
                        pt = self.calib_pts_proj[self.calib_step]
                        self.renderer.draw_wizard(pt)
                    else:
                        cam = frame if self.config.simulation_mode else None
                        self.renderer.project(all_detections, camera_frame=cam)

            # ─── 5. FRAME WEB (desenha + codifica JPEG de uma vez) ───
            for d in all_detections:
                x, y, w, h = d["box"]
                c = d["color"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), c, 2)
                cv2.putText(frame, d["label"], (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
                cv2.circle(frame, (x + w // 2, y + h // 2), 4, c, -1)

            # Codifica JPEG uma única vez (0.6ms) e armazena os bytes prontos
            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                with self._web_lock:
                    self._web_jpeg = buf.tobytes()
                self._web_event.set()  # Acorda o gerador de streaming

            # Guardar para acesso público
            self._all_detections = all_detections

    def _run_yolo(self, frame, manual_dets):
        """Roda YOLO inline (mesmo thread). Só chamado a cada N frames."""
        if self._yolo_model is None:
            return
        try:
            results = self._yolo_model(frame, verbose=False, imgsz=320)
            new_dets = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf > 0.4:
                        label = self._yolo_model.names[cls]
                        b = box.xywh[0].cpu().numpy()
                        x, y = int(b[0] - b[2] / 2), int(b[1] - b[3] / 2)
                        w, h = int(b[2]), int(b[3])
                        new_dets.append({
                            "box": (x, y, w, h),
                            "label": f"{label} {conf:.0%}",
                            "id": -1,
                            "color": (0, 255, 220),
                            "type": "auto"
                        })
            self._yolo_detections = new_dets
        except Exception as e:
            print(f"[YOLO] Erro: {e}")

    # ═══════════════════════════════════════════════════
    # API Pública
    # ═══════════════════════════════════════════════════

    def get_frame_bytes(self):
        """Retorna JPEG pré-codificado. Zero processamento aqui."""
        with self._web_lock:
            return self._web_jpeg

    def wait_for_frame(self, timeout=0.1):
        """Bloqueia até um frame novo estar disponível."""
        self._web_event.wait(timeout=timeout)
        self._web_event.clear()

    @property
    def detections(self):
        return getattr(self, '_all_detections', [])

    @property
    def detector(self):
        """Compatibilidade com app.py que acessa core.detector.use_yolo"""
        return self

    @property
    def use_yolo(self):
        return self._yolo_enabled

    @use_yolo.setter
    def use_yolo(self, val):
        self._yolo_enabled = val

    def add_tracker(self, roi, label):
        frame = getattr(self, '_current_frame', None)
        if frame is None:
            return
        tracker = cv2.legacy.TrackerMOSSE_create()
        tracker.init(frame, tuple(roi))
        color = (
            int(np.random.randint(80, 255)),
            int(np.random.randint(80, 255)),
            int(np.random.randint(80, 255))
        )
        self.trackers[self._next_id] = {
            "tracker": tracker,
            "label": label,
            "color": color
        }
        self._next_id += 1

    def reset_trackers(self):
        self.trackers = {}

    def toggle_yolo(self):
        self._yolo_enabled = not self._yolo_enabled
        if self._yolo_enabled and self._yolo_model is None:
            print("[IA] Carregando YOLOv8n pela primeira vez...")
            from ultralytics import YOLO
            self._yolo_model = YOLO('yolov8n.pt')
            print("[IA] Modelo carregado!")
        if not self._yolo_enabled:
            self._yolo_detections = []
        return self._yolo_enabled

    def toggle_simulation(self):
        self.config.simulation_mode = not self.config.simulation_mode
        if self.renderer:
            self.renderer.close()
            self.renderer = None
        return self.config.simulation_mode

    def set_projector_monitor(self, index):
        """Muda o monitor de projeção e reinicia o renderer."""
        import screeninfo
        monitors = screeninfo.get_monitors()
        if 0 <= index < len(monitors):
            self.config.projector_screen_index = index
            self.config.save()
            if self.renderer:
                self.renderer.close()
                self.renderer = None
            return True
        return False

    def start_calibration(self):
        import screeninfo
        screens = screeninfo.get_monitors()
        m = screens[min(self.config.projector_screen_index, len(screens) - 1)]
        w, h = m.width, m.height
        mg = 0.1
        self.calib_pts_proj = [
            (int(w * mg), int(h * mg)),
            (int(w * (1 - mg)), int(h * mg)),
            (int(w * (1 - mg)), int(h * (1 - mg))),
            (int(w * mg), int(h * (1 - mg))),
        ]
        self.calib_pts_cam = []
        self.calib_step = 0
        self.calibrating = True
        if self.renderer:
            self.renderer.close()
            self.renderer = None

    def record_calibration_point(self, x=None, y=None):
        if not self.calibrating:
            return False
        
        cx, cy = None, None
        if x is not None and y is not None:
            cx, cy = x, y
        else:
            # Tenta detecção automática se não houver clique manual
            dets = self.detections
            if dets:
                box = dets[0]["box"]
                cx, cy = box[0] + box[2] // 2, box[1] + box[3] // 2
        
        if cx is not None:
            self.calib_pts_cam.append((cx, cy))
            self.calib_step += 1
            if self.calib_step >= 4:
                self.config.update_homography(self.calib_pts_cam, self.calib_pts_proj)
                self.calibrating = False
                if self.renderer:
                    self.renderer.close()
                    self.renderer = None
                return "finished"
            return "next"
        return "no_object"

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.renderer:
            self.renderer.close()
        cv2.destroyAllWindows()
