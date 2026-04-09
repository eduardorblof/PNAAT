# ============================================
# Implementação de YOLO para processamento de vídeo no Colab
# ============================================


#pip install ultralytics
# pip install tqdm

# Importa bibliotecas essenciais
import cv2
from ultralytics import YOLO
from IPython.display import display, HTML, clear_output
from base64 import b64encode
import os
from tqdm.notebook import tqdm # Adicionado para exibir a barra de progresso

# ---
# 1. FAÇA O UPLOAD DE UM ARQUIVO DE VÍDEO NO SEU NOTEBOOK COLAB
# O vídeo deve ter um nome como 'video_exemplo.mp4'
# ---

# Nome do arquivo de vídeo de entrada e saída
video_path = '/content/video_exemplo.mp4'
output_path = '/content/vídeo_de_saída.mp4'

# Verificação: o arquivo de entrada existe?
if not os.path.exists(video_path):
    print(f"Erro: Arquivo '{video_path}' não encontrado. Por favor, faça o upload do vídeo.")
else:
    # Carrega o modelo YOLO pré-treinado
    model = YOLO("yolov8n.pt")

    # Inicializa a captura do vídeo
    cap = cv2.VideoCapture(video_path)

    # Obtém as propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Cria um objeto VideoWriter para salvar o novo vídeo
    # Usando o codec 'XVID' que costuma ser mais confiável no Colab
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Iniciando o processamento de {total_frames} frames...")

    # Loop para processar cada frame com uma barra de progresso
    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Realiza a detecção de objetos
            results = model(frame)

            # Renderiza os resultados no frame
            annotated_frame = results[0].plot()

            # Escreve o frame processado no arquivo de saída
            out.write(annotated_frame)
            pbar.update(1)

    print("Processamento concluído! Salvando o vídeo.")

    # Libera os recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # ============================================
    # EXIBE O VÍDEO PROCESSADO NO NOTEBOOK
    # ============================================

    def show_video(file_path):
        if not os.path.exists(file_path):
            print(f"Erro: Arquivo de saída '{file_path}' não foi criado corretamente.")
            return

        mp4 = open(file_path, 'rb').read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        display(HTML(f""" 
            <video width=700 controls>
                  <source src="{data_url}" type="video/mp4">
            </video>
          """))

    show_video(output_path)