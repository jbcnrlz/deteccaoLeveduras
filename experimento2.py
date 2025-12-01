import numpy as np
from cellpose import models, core, io, plot
import matplotlib.pyplot as plt
import cv2 # Necessário para a conversão de canais
import os # Necessário para criar diretório

def main():
    # Cria um diretório para salvar as leveduras, se ele não existir
    output_dir = "leveduras_segmentadas"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. ESPECIFICAÇÃO DO MODELO
    model = models.CellposeModel(gpu=True, model_type='cyto') 
    
    img_path = 'fonte2.jpg'
    img = io.imread(img_path)
    print(f"Shape da imagem original: {img.shape}")

    # 2. CONFIGURAÇÃO DE CANAIS PARA IMAGEM MONOCROMÁTICA (CAMPO CLARO)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    channels = [0, 0] # (canal_de_segmentacao, canal_do_nucleo)

    # 3. AJUSTE DOS PARÂMETROS
    flow_threshold = 0.2 
    cellprob_threshold = 0.2 
    tile_norm_blocksize = 0 

    print("Executando segmentação com Cellpose...")
    masks, flows, styles = model.eval(
        img_gray, 
        channels=channels,
        batch_size=32, 
        flow_threshold=flow_threshold, 
        cellprob_threshold=cellprob_threshold, 
        normalize={"tile_norm_blocksize": tile_norm_blocksize}
    )
    print("Segmentação concluída.")

    # Exibe a segmentação geral
    fig = plt.figure(figsize=(12,5))
    plot.show_segmentation(fig, img, masks, flows[0])
    plt.tight_layout()
    plt.show()

    # O número de leveduras é o número de rótulos únicos, exceto o 0 (fundo)
    unique_levedura_ids = np.unique(masks)
    total_leveduras = len(unique_levedura_ids) - 1 
    print(f"\nContagem total de leveduras segmentadas: {total_leveduras}")

    levedura_count = 0
    # Itera sobre cada ID de levedura (começando do 1, pois 0 é o fundo)
    for levedura_id in unique_levedura_ids:
        if levedura_id == 0: # Ignora o fundo
            continue

        levedura_count += 1

        # Cria uma máscara binária para a levedura atual
        # A máscara terá 255 (branco) onde a levedura está, e 0 (preto) em outro lugar
        individual_mask = (masks == levedura_id).astype(np.uint8) * 255

        # Encontra os contornos da levedura individual para obter a caixa delimitadora (bounding box)
        # RETR_EXTERNAL para pegar apenas o contorno externo, CHAIN_APPROX_SIMPLE para compressão
        contours, _ = cv2.findContours(individual_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Pega o maior contorno (se houver múltiplos, embora não deva acontecer com máscara individual)
            cnt = max(contours, key=cv2.contourArea) 
            
            # Obtém as coordenadas da caixa delimitadora (x, y, largura, altura)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Adiciona um pequeno preenchimento (padding) para garantir que a célula não seja cortada
            padding = 5 
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(img.shape[1], x + w + padding)
            y_end = min(img.shape[0], y + h + padding)

            # Recorta a levedura da imagem original (RGB) ou da imagem em escala de cinza
            # cropped_levedura_rgb = img[y_start:y_end, x_start:x_end]
            cropped_levedura_gray = img_gray[y_start:y_end, x_start:x_end]

            # Salva a levedura recortada
            output_filename = os.path.join(output_dir, f"levedura_{levedura_id:04d}.png")
            cv2.imwrite(output_filename, cropped_levedura_gray) # Salvando a versão em escala de cinza
            # Se quiser salvar a versão RGB, use: cv2.imwrite(output_filename, cropped_levedura_rgb)
            
            print(f"Salva {output_filename}")
        else:
            print(f"Não foi possível encontrar contorno para a levedura ID: {levedura_id}")

    print(f"\nTodas as {levedura_count} leveduras foram salvas no diretório '{output_dir}'.")

if __name__ == '__main__':
    main()