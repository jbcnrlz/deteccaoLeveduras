import numpy as np
import cv2
import matplotlib.pyplot as plt
from cellpose import models, core, io, plot
from cellpose.io import imread # Cellpose tem um leitor de imagem robusto
import math
from collections import defaultdict

def analisar_morfologia_com_cellpose(caminho_imagem):
    """
    Segmenta colônias usando Cellpose e calcula características morfológicas.
    
    :param caminho_imagem: Caminho para o arquivo da imagem da colônia (ex: 'colonia.jpeg').
    :return: Uma lista de dicionários com as características das colônias.
    """
    
    # --- 1. Segmentação com Cellpose ---
    
    print("Iniciando segmentação com Cellpose...")
    
    # Carrega a imagem
    img = imread(caminho_imagem)
    
    # Inicializa o modelo Cellpose - 'cyto' é o modelo de citoplasma, bom para formas arredondadas/irregulares
    # Usar device=cuda se disponível, mas CPU funciona.
    model = models.CellposeModel(gpu=True, model_type='cyto') 
    
    # Executa a avaliação da segmentação.
    # 'diameter=None' deixa o modelo estimar o tamanho dos objetos.
    # 'channels=[0,0]' é para imagens em escala de cinza ou onde o canal de cor não importa.
    masks, flows, styles = model.eval(img, 
                                    diameter=None, 
                                    channels=[0,0], 
                                    flow_threshold=0.8, # Limite de fluxo para separar objetos próximos
                                    cellprob_threshold=0.0) # Limite de probabilidade

    # 'masks' é uma matriz 2D onde cada valor inteiro positivo é uma colônia diferente.
    
    # --- 2. Extração de Características (usando OpenCV nas Máscaras) ---
    
    # Prepara a imagem para visualização
    img_display = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Converte de RGB (Cellpose) para BGR (OpenCV)
    
    # Inicializa a lista de resultados e o contador
    resultados = []
    colony_ids = np.unique(masks)
    
    # Itera sobre cada ID de colônia segmentada (excluindo o 0, que é o fundo)
    for i, obj_id in enumerate(colony_ids):
        if obj_id == 0:
            continue
            
        # Cria uma máscara binária para a colônia atual
        colony_mask = (masks == obj_id).astype(np.uint8) * 255
        
        # Encontra o contorno da colônia segmentada
        contornos, _ = cv2.findContours(colony_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # O Cellpose deve gerar apenas um contorno por ID, então pegamos o primeiro (e único)
        if not contornos:
            continue

        contorno = contornos[0]
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)

        # Filtro de ruído: Ignora objetos muito pequenos que o Cellpose pode ter detectado
        MIN_AREA_THRESHOLD = 100 
        if area < MIN_AREA_THRESHOLD:
            continue
            
        # --- Cálculo das Características ---
        
        # 1. Circularidade (1.0 = Círculo Perfeito)
        circularidade = (4 * math.pi * area) / (perimetro * perimetro) if perimetro > 0 else 0
        
        # 2. Fator de Rugosidade (Roughness Factor) - Indicador de irregularidade da borda
        casco_convexo = cv2.convexHull(contorno)
        area_casco_convexo = cv2.contourArea(casco_convexo)
        # Relação próxima a 1.0 indica forma lisa e convexa
        fator_rugosidade = area / area_casco_convexo if area_casco_convexo > 0 else 0
        
        # 3. Proporção Aspecto
        x, y, w, h = cv2.boundingRect(contorno)
        proporcao_aspecto = w / h if h > 0 else 0

        resultados.append({
            'ID': i,
            'Area_pixels': round(area, 2),
            'Perimetro_pixels': round(perimetro, 2),
            'Circularidade': round(circularidade, 3), 
            'Fator_Rugosidade': round(fator_rugosidade, 3),
            'Proporcao_Aspecto': round(proporcao_aspecto, 3)
        })
        
        # --- Visualização: Desenha o contorno na imagem original ---
        cv2.drawContours(img_display, [contorno], 0, (0, 255, 0), 2) # Desenha o contorno em verde
        cv2.putText(img_display, str(i), (x, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # --- 3. Resultados ---
    
    total_colonias = len(resultados)
    print(f"\n--- Cellpose Concluído ---")
    print(f"Total de Colônias Segmentadas e Analisadas: {total_colonias}\n")

    print("Resumo das Primeiras Colônias (Para ver a rugosidade):")
    
    # Ordena por área para analisar as maiores primeiro
    resultados_ordenados = sorted(resultados, key=lambda x: x['Area_pixels'], reverse=True)
    
    for res in resultados_ordenados[:10]:
        print(f"Colônia {res['ID']:02d} | Área: {res['Area_pixels']} px | Circularidade: {res['Circularidade']} | Rugosidade: {res['Fator_Rugosidade']}")
    if total_colonias > 10:
        print("...")

    # Exibe a imagem com os contornos (em RGB, pois o Matplotlib espera RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    plt.title(f"Segmentação Cellpose (Modelo Cyto). Total: {total_colonias} Colônias.")
    plt.axis('off')
    plt.show()
    
    return resultados_ordenados

# --- Execução do Código ---
# Certifique-se de que 'colonia.jpeg' está na mesma pasta.
caracteristicas_cellpose = analisar_morfologia_com_cellpose('colonia2.jpg')