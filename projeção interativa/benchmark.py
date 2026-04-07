"""
Diagnóstico de Performance — Descobre onde está o gargalo.
Roda 60 frames e mede o tempo de cada etapa.
"""
import cv2
import numpy as np
import time

print("="*60)
print("  DIAGNÓSTICO DE PERFORMANCE")
print("="*60)

# Teste 1: Velocidade pura da câmera
print("\n[Teste 1] Câmera pura (sem processamento)...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Descartar primeiros frames (warm-up)
for _ in range(5):
    cap.read()

start = time.perf_counter()
for _ in range(30):
    ret, frame = cap.read()
elapsed = time.perf_counter() - start
fps_cam = 30 / elapsed
print(f"  Câmera: {fps_cam:.1f} FPS ({elapsed/30*1000:.1f}ms por frame)")
print(f"  Resolução: {frame.shape[1]}x{frame.shape[0]}")

# Teste 2: JPEG encoding
print("\n[Teste 2] Codificação JPEG (qualidade 70)...")
start = time.perf_counter()
for _ in range(30):
    cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
elapsed = time.perf_counter() - start
print(f"  JPEG encode: {elapsed/30*1000:.1f}ms por frame")

# Teste 3: JPEG encoding com resolução reduzida
print("\n[Teste 3] JPEG com frame reduzido (320x240)...")
small = cv2.resize(frame, (320, 240))
start = time.perf_counter()
for _ in range(30):
    cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
elapsed = time.perf_counter() - start
print(f"  JPEG encode (small): {elapsed/30*1000:.1f}ms por frame")

# Teste 4: Tracker MOSSE
print("\n[Teste 4] Rastreador MOSSE...")
tracker = cv2.legacy.TrackerMOSSE_create()
h, w = frame.shape[:2]
roi = (w//4, h//4, w//2, h//2)
tracker.init(frame, roi)
start = time.perf_counter()
for _ in range(30):
    ret, frame = cap.read()
    if ret:
        tracker.update(frame)
elapsed = time.perf_counter() - start
fps_track = 30 / elapsed
print(f"  Câmera + MOSSE: {fps_track:.1f} FPS ({elapsed/30*1000:.1f}ms por frame)")

# Teste 5: cv2.imshow direto (sem web)
print("\n[Teste 5] Exibição direta (cv2.imshow)...")
start = time.perf_counter()
for _ in range(30):
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Benchmark", frame)
        cv2.waitKey(1)
elapsed = time.perf_counter() - start
fps_show = 30 / elapsed
print(f"  Câmera + imshow: {fps_show:.1f} FPS ({elapsed/30*1000:.1f}ms por frame)")

cv2.destroyAllWindows()
cap.release()

# Resumo
print("\n" + "="*60)
print("  RESUMO")
print("="*60)
print(f"  Câmera pura:          {fps_cam:.0f} FPS")
print(f"  Câmera + MOSSE:       {fps_track:.0f} FPS")
print(f"  Câmera + cv2.imshow:  {fps_show:.0f} FPS")
print()
if fps_cam < 15:
    print("  ⚠️  GARGALO: A câmera está muito lenta!")
    print("  → Solução: Reduzir resolução ou trocar backend do OpenCV")
elif fps_track < 15:
    print("  ⚠️  GARGALO: O rastreador está pesado!")
else:
    print("  ✅ Hardware OK! O gargalo deve estar no streaming web (MJPEG).")
    print("  → Solução: Reduzir resolução do stream web e limitar FPS")
print()
