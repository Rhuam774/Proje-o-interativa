import cv2

class AdaptiveFPSController:
    """Controlador de FPS adaptativo compartilhado por todos os efeitos"""
    def __init__(self, normal_fps=8, low_fps=2, threshold=1500):
        self.normal_fps = normal_fps
        self.low_fps = low_fps
        self.threshold = threshold
        self.fps_ratio = normal_fps // low_fps
        self.counter = 0
    
    def should_process_frame(self, edge_count=0, canvas=None):
        """
        Determina se deve processar o frame atual baseado na carga
        
        Retorna True se deve processar, False se deve pular
        """
        if edge_count > self.threshold:
            self.counter += 1
            if self.counter % self.fps_ratio != 0:
                # Pula frame - mantém canvas
                if canvas is not None:
                    cv2.putText(canvas, f"FPS: {self.low_fps} | EDGES: {edge_count}", 
                               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                return False
            else:
                # Processa frame
                if canvas is not None:
                    cv2.putText(canvas, f"FPS: {self.low_fps} | EDGES: {edge_count}", 
                               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                return True
        else:
            # FPS normal
            self.counter = 0
            if canvas is not None:
                cv2.putText(canvas, f"FPS: {self.normal_fps} | EDGES: {edge_count}", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            return True
