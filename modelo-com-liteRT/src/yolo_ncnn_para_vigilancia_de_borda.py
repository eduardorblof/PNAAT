import cv2
from datetime import datetime
import time
from ultralytics import YOLO
from picamera2 import Picamera2 # <-- IMPORTAR PICAMERA2
 
# --- Configurações do Projeto ---
MODEL_PATH = "yolo11n_ncnn_model"
LOG_FILE = "registro_deteccoes.txt"
PERSON_CLASS_ID = 0
LOG_COOLDOWN_SEC = 5
 

# --- Funções de Ajuda --- 

def log_detection(timestamp):
    """Escreve a detecção em um arquivo de log."""
    log_entry = f"[{timestamp}] - Pessoa Detectada.\n"
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry)
    print(f"Log gravado: {log_entry.strip()}")

# --- Processo Principal --- 

print("Iniciando Detecção de Pessoas (YOLO NCNN + Picamera2)...")
 

# 1. Carrega o modelo NCNN
print("Carregando modelo NCNN...")
model = YOLO(MODEL_PATH)
print("Modelo carregado com sucesso.")
 

# 2. Inicia a câmera (MODO PICAMERA2)
try:
    picam2 = Picamera2()
    # Configura para "preview" (rápido) e um formato que o OpenCV entende
    config = picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    print("Câmera PiCamera iniciada com sucesso.")
except Exception as e:
    print(f"ERRO: Não foi possível iniciar a Picamera2. {e}")
    print("Verifique se o 'Legacy Camera' está DESABILITADO em raspi-config.")
    exit()
 
last_log_time = time.time()
print("Pressione 'q' na janela de vídeo para sair...")
 
try:
    while True:
        # 3. Captura o frame (MODO PICAMERA2)
        # capture_array() já retorna um array numpy pronto
        frame = picam2.capture_array()
       
        # (Opcional, mas recomendado) Converte de RGB (Picamera2) para BGR (OpenCV)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
 
        # 4. Inferência com Ultralytics
        results = model(frame_bgr, verbose=False) # Use o frame BGR
 
        person_detected_in_frame = False
       
        # 5. Processamento das Detecções
        for box in results[0].boxes:
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
           
            if class_id == PERSON_CLASS_ID and confidence > 0.5:
                person_detected_in_frame = True
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
               
                # Desenha no frame BGR
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_bgr, f"Pessoa: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                break
 
        # 6. Registro Condicional (Logging)
        if person_detected_in_frame and (time.time() - last_log_time > LOG_COOLDOWN_SEC):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_detection(timestamp)
            last_log_time = time.time()
 
        # Mostra o vídeo (frame BGR)
        cv2.imshow('YOLO NCNN Edge Detector (Picam2) - Pressione Q', frame_bgr)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
finally:
    # Limpeza
    picam2.stop() # <--- Comando de parada do Picamera2
    cv2.destroyAllWindows()
    print("Script finalizado. Log de detecções salvo em:", LOG_FILE)
    