import cv2
import numpy as np
from ultralytics import YOLO

class Detector:
    def __init__(self, use_yolo=False):
        self.use_yolo = use_yolo
        self._model = None  # carregado sob demanda quando usuário ligar a IA
        # Rastreadores {id: {tracker, label, color}}
        self.trackers = {}
        self.next_id = 0
        self._yolo_cache = []

    @property
    def model(self):
        """YOLO só é carregado na primeira vez que o usuário ligar a IA."""
        if self._model is None:
            print("[IA] Carregando YOLOv8n...")
            from ultralytics import YOLO
            self._model = YOLO('yolov8n.pt')
        return self._model

    def add_tracker(self, frame, roi, label="Objeto"):
        """MOSSE: o rastreador mais rápido do OpenCV. Opera a ~500fps."""
        tracker = cv2.legacy.TrackerMOSSE_create()  # usa legacy por compatibilidade
        tracker.init(frame, tuple(roi))
        color = (
            int(np.random.randint(80, 255)),
            int(np.random.randint(80, 255)),
            int(np.random.randint(80, 255))
        )
        self.trackers[self.next_id] = {
            "tracker": tracker,
            "label": label,
            "color": color
        }
        self.next_id += 1
        return self.next_id - 1

    def remove_tracker(self, tid):
        if tid in self.trackers:
            del self.trackers[tid]

    def update(self, frame, yolo_every_n=3, frame_count=0):
        """
        Atualiza rastreadores KCF (todo frame) e roda YOLO a cada N frames.
        No i5 sem GPU, yolo_every_n=3 dá ~20fps reais de IA com rastreamento suave.
        """
        detections = []

        # === 1. RASTREADORES KCF (ultra-rápido, todo frame) ===
        to_remove = []
        for tid, data in self.trackers.items():
            success, box = data["tracker"].update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                detections.append({
                    "box": (x, y, w, h),
                    "label": data["label"],
                    "id": tid,
                    "color": data["color"],
                    "type": "manual"
                })
            else:
                to_remove.append(tid)

        for tid in to_remove:
            self.remove_tracker(tid)

        # === 2. YOLO EM SKIP-FRAME INTELIGENTE ===
        if self.use_yolo:
            if frame_count % yolo_every_n == 0:
                # Roda YOLO neste frame e guarda em cache
                new_yolo = []
                results = self.model(frame, verbose=False, imgsz=320)  # imgsz=320: 2x mais rápido que 640
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        if conf > 0.4:
                            label = self.model.names[cls]
                            b = box.xywh[0].cpu().numpy()
                            x = int(b[0] - b[2]/2)
                            y = int(b[1] - b[3]/2)
                            w = int(b[2])
                            h = int(b[3])
                            new_yolo.append({
                                "box": (x, y, w, h),
                                "label": f"{label} {conf:.0%}",
                                "id": -1,
                                "color": (0, 255, 220),
                                "type": "auto"
                            })
                self._yolo_cache = new_yolo

            # Adicionar detecções YOLO do cache (sem duplicatas com rastreadores manuais)
            for det in self._yolo_cache:
                dx, dy, dw, dh = det["box"]
                # Verificar sobreposição com rastreadores manuais
                is_duplicate = any(
                    abs((dx + dw/2) - (mx + mw/2)) < 60 and abs((dy + dh/2) - (my + mh/2)) < 60
                    for mx, my, mw, mh in [d["box"] for d in detections]
                )
                if not is_duplicate:
                    detections.append(det)

        return detections
