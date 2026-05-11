import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# --- FATOR DE CALIBRAÇÃO (PRECISA SER OBTIDO POR MEIO DE UMA LÂMINA CALIBRADA) ---
MICRONS_PER_PIXEL = 0.035  # Exemplo: 0.035 micrômetros por pixel
# ---------------------------------------------------------------------------------

def analisar_e_visualizar_levedura(caminho_imagem, microns_por_pixel):
    """
    Analisa uma imagem de levedura e gera uma visualização detalhada com as medidas.
    """
    if not os.path.exists(caminho_imagem):
        print(f"Erro: Arquivo não encontrado no caminho: {caminho_imagem}")
        return None, None

    # 1. Carregar a imagem em escala de cinza e RGB (para visualização colorida)
    img_gray = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE) 
    img_bgr = cv2.imread(caminho_imagem, cv2.IMREAD_COLOR) # Para desenhar sobre ela
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Matplotlib usa RGB

    # 2. Tratamento de Contraste (CLAHE) para binarização
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted_img = clahe.apply(img_gray)
    
    # 3. Limiarização
    _, binary_mask = cv2.threshold(contrasted_img, 80, 255, cv2.THRESH_BINARY_INV) 

    # 4. Encontrar Contornos
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback para Otsu se não encontrar nada com o limiar fixo
        _, binary_mask_otsu = cv2.threshold(contrasted_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_mask_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or cv2.contourArea(max(contours, key=cv2.contourArea)) < 50:
             print("Nenhum contorno válido encontrado.")
             return None, None
             
    contour = max(contours, key=cv2.contourArea)
    
    # Filtra contornos muito pequenos (provavelmente ruído)
    if cv2.contourArea(contour) < 50: 
        print("O contorno encontrado é muito pequeno (provavelmente ruído).")
        return None, None
    
    caracteristicas = {}

    # --- CÁLCULO DAS CARACTERÍSTICAS ---
    area_pixels = cv2.contourArea(contour)
    area_microns2 = area_pixels * (microns_por_pixel ** 2)
    caracteristicas['Área'] = f"{area_microns2:.2f} µm²"
    
    perimetro_pixels = cv2.arcLength(contour, closed=True)
    perimetro_microns = perimetro_pixels * microns_por_pixel
    caracteristicas['Perímetro'] = f"{perimetro_microns:.2f} µm"

    circularidade = (4 * np.pi * area_pixels) / (perimetro_pixels ** 2) if perimetro_pixels > 0 else 0.0
    caracteristicas['Circularidade'] = f"{circularidade:.3f}"
    
    solidez = area_pixels / cv2.contourArea(cv2.convexHull(contour)) if cv2.contourArea(cv2.convexHull(contour)) > 0 else 0.0
    caracteristicas['Solidez'] = f"{solidez:.3f}"

    eixo_maior_microns = 'N/A'
    eixo_menor_microns = 'N/A'
    relacao_aspecto = 'N/A'
    angulacao_graus = 'N/A'
    
    center_ellipse = (0,0)
    axes_ellipse = (0,0)
    angle_ellipse = 0

    if len(contour) >= 5:
        # (center), (axes), (angle)
        (center_ellipse, axes_ellipse, angle_ellipse) = cv2.fitEllipse(contour)

        eixo_menor_pixels = min(axes_ellipse)
        eixo_maior_pixels = max(axes_ellipse)

        eixo_maior_microns = eixo_maior_pixels * microns_por_pixel
        eixo_menor_microns = eixo_menor_pixels * microns_por_pixel
        
        relacao_aspecto = eixo_maior_pixels / eixo_menor_pixels
        angulacao_graus = angle_ellipse # Já está em graus (0-180)
        
        caracteristicas['Comprimento do Eixo Maior'] = f"{eixo_maior_microns:.2f} µm"
        caracteristicas['Comprimento do Eixo Menor'] = f"{eixo_menor_microns:.2f} µm"
        caracteristicas['Relação de Aspecto'] = f"{relacao_aspecto:.2f}"
        caracteristicas['Angulação do Eixo Principal'] = f"{angulacao_graus:.2f}°"
        
    M = cv2.moments(contour)
    cx, cy = 0, 0
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        caracteristicas['Centroide'] = f"({cx}, {cy}) px"
    else:
        caracteristicas['Centroide'] = "N/A"
    
    # --- VISUALIZAÇÃO ---
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"Análise de Levedura: {os.path.basename(caminho_imagem)}", fontsize=16)

    # Subplot 1: Imagem Original com Elipse e Contorno
    axes[0].imshow(img_rgb)
    axes[0].set_title('Levedura e Análise de Forma')
    axes[0].axis('off')

    # Desenhar contorno na imagem RGB
    cv2.drawContours(img_rgb, [contour], -1, (0, 255, 0), 2) # Contorno verde

    # Desenhar elipse de ajuste na imagem RGB
    
    if len(contour) >= 5:
        # A elipse em OpenCV é (centro), (eixos_largura, eixos_altura), angulo
        # Note que center e axes são tuplas (float, float)
        # Convertemos para int para drawEllipse
        cv2.ellipse(img_rgb, (int(center_ellipse[0]), int(center_ellipse[1])), 
                    (int(axes_ellipse[0]/2), int(axes_ellipse[1]/2)), # OpenCV usa raios, não diâmetros
                    angle_ellipse, 0, 360, (255, 0, 0), 2) # Elipse azul

        # Desenhar centroide
        cv2.circle(img_rgb, (int(center_ellipse[0]), int(center_ellipse[1])), 5, (255, 255, 0), -1) # Centroide amarelo
    
    axes[0].imshow(img_rgb) # Exibe a imagem com as sobreposições

    # Subplot 2: Máscara Binária e Tabela de Medidas
    axes[1].imshow(binary_mask, cmap='gray')
    axes[1].set_title('Máscara Binária e Medidas')
    axes[1].axis('off')

    # Adicionar as características como texto no subplot da máscara
    text_y_offset = 0.1
    for key, value in caracteristicas.items():
        axes[1].text(0.05, 1 - text_y_offset, f"{key}: {value}", 
                     transform=axes[1].transAxes, fontsize=10, 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        text_y_offset += 0.08

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta o layout para o título
    plt.show()

    return caracteristicas, img_rgb # Retorna as características e a imagem com sobreposições

# --- EXECUÇÃO ---
if __name__ == '__main__':
    caminho_imagem_levedura = 'leveduras_segmentadas/levedura_0043.png'
    
    # IMPORTANTE: Use o fator de calibração correto da sua configuração!
    fator_calibracao_microns_por_pixel = 0.035 

    caracteristicas_extraidas, imagem_visualizada = analisar_e_visualizar_levedura(
        caminho_imagem_levedura, 
        fator_calibracao_microns_por_pixel
    )

    if caracteristicas_extraidas:
        print("\n--- RESUMO DAS CARACTERÍSTICAS EXTRAÍDAS ---")
        for key, value in caracteristicas_extraidas.items():
            print(f"{key}: {value}")
        print("------------------------------------------")