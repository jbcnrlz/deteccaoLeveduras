import numpy as np, cv2, torch, os, sys, json, pandas as pd
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from typing import List, Dict, Any
import torch.optim as optim
from tqdm import tqdm
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from dataset.yeastDataset import YeastDataset


def get_model(num_classes: int = 2, pretrained: bool = True):
    """Carregar modelo Mask R-CNN"""
    model = maskrcnn_resnet50_fpn(pretrained=pretrained)
    
    # Substituir o classificador de caixas
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Substituir o classificador de máscaras
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model

class YeastFeatureExtractor:
    """Extrator de características para leveduras"""
    def __init__(self):
        self.feature_names = [
            'area', 'perimeter', 'circularity', 'aspect_ratio', 
            'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity',
            'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation',
            'solidity', 'elongation'
        ]
    
    def extract_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Extrair características de uma única levedura"""
        features = {}
        
        # Características morfológicas
        morph_features = self._extract_morphological_features(mask)
        features.update(morph_features)
        
        # Características de intensidade
        intensity_features = self._extract_intensity_features(image, mask)
        features.update(intensity_features)
        
        # Características de textura
        texture_features = self._extract_texture_features(image, mask)
        features.update(texture_features)
        
        return features
    
    def _extract_morphological_features(self, mask: np.ndarray) -> Dict[str, float]:
        """Extrair características morfológicas"""
        # Encontrar contornos
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {key: 0.0 for key in ['area', 'perimeter', 'circularity', 
                                       'aspect_ratio', 'solidity', 'elongation']}
        
        # Maior contorno
        contour = max(contours, key=cv2.contourArea)
        
        # Área e perímetro
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Circularidade
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # Retângulo delimitador
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Área convexa e solidez
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Alongamento
        elongation = (w - h) / (w + h) if (w + h) > 0 else 0
        
        return {
            'area': float(area),
            'perimeter': float(perimeter),
            'circularity': float(circularity),
            'aspect_ratio': float(aspect_ratio),
            'solidity': float(solidity),
            'elongation': float(elongation)
        }
    
    def _extract_intensity_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Extrair características de intensidade"""
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        
        # Aplicar máscara
        masked_pixels = gray_image[mask > 0]
        
        if len(masked_pixels) == 0:
            return {
                'mean_intensity': 0.0,
                'std_intensity': 0.0,
                'min_intensity': 0.0,
                'max_intensity': 0.0
            }
        
        return {
            'mean_intensity': float(np.mean(masked_pixels)),
            'std_intensity': float(np.std(masked_pixels)),
            'min_intensity': float(np.min(masked_pixels)),
            'max_intensity': float(np.max(masked_pixels))
        }
    
    def _extract_texture_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Extrair características de textura usando GLCM"""
        from skimage.feature import graycomatrix, graycoprops
        
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        
        # Aplicar máscara
        masked_image = np.zeros_like(gray_image)
        masked_image[mask > 0] = gray_image[mask > 0]
        
        # Calcular GLCM
        try:
            glcm = graycomatrix(masked_image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                               symmetric=True, normed=True)
            
            # Calcular propriedades de textura
            contrast = np.mean(graycoprops(glcm, 'contrast'))
            dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
            homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
            energy = np.mean(graycoprops(glcm, 'energy'))
            correlation = np.mean(graycoprops(glcm, 'correlation'))
            
            return {
                'contrast': float(contrast),
                'dissimilarity': float(dissimilarity),
                'homogeneity': float(homogeneity),
                'energy': float(energy),
                'correlation': float(correlation)
            }
        except:
            return {
                'contrast': 0.0,
                'dissimilarity': 0.0,
                'homogeneity': 0.0,
                'energy': 0.0,
                'correlation': 0.0
            }

class YeastAnalyzer:
    """Analisador de leveduras com Mask R-CNN"""
    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_extractor = YeastFeatureExtractor()
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Carregar modelo treinado"""
        self.model = get_model(num_classes=2, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def train(self, data_dir: str, output_dir: str = "./yeast_model", 
              num_epochs: int = 10, annotation_dir: str = None):
        """Treinar o modelo"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Dataset e DataLoader
        dataset = YeastDataset(data_dir, annotation_dir=annotation_dir)
        data_loader = DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=2,
            collate_fn=self.collate_fn
        )
        
        # Modelo
        self.model = get_model(num_classes=2, pretrained=True)
        self.model.to(self.device)
        
        # Otimizador
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        
        # Loop de treinamento
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for images, targets in progress_bar:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets if t]
                
                if not targets:  # Pular se não há targets
                    continue
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                total_loss += losses.item()
                progress_bar.set_postfix(loss=losses.item())
            
            print(f"Epoch {epoch+1} finished, Loss: {total_loss/len(data_loader):.4f}")
            
            # Salvar checkpoint
            if (epoch + 1) % 5 == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(output_dir, f"yeast_model_epoch_{epoch+1}.pth")
                )
        
        # Salvar modelo final
        torch.save(
            self.model.state_dict(),
            os.path.join(output_dir, "yeast_model_final.pth")
        )
        
        print(f"Modelo salvo em: {output_dir}")
    
    def collate_fn(self, batch):
        """Função para colatar batch com targets variáveis"""
        return tuple(zip(*batch))
    
    def predict(self, image_path: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Fazer predição em uma imagem"""
        if self.model is None:
            raise ValueError("Modelo não carregado. Chame load_model() primeiro.")
        
        # Carregar e pré-processar imagem
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
        
        original_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
        
        # Predição
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        prediction = predictions[0]
        
        # Filtrar por confiança
        masks = []
        scores = []
        boxes = []
        
        if 'scores' in prediction and len(prediction['scores']) > 0:
            keep = prediction['scores'] > confidence_threshold
            
            if 'masks' in prediction:
                masks = prediction['masks'][keep].cpu().numpy()
            if 'scores' in prediction:
                scores = prediction['scores'][keep].cpu().numpy()
            if 'boxes' in prediction:
                boxes = prediction['boxes'][keep].cpu().numpy()
        
        # Extrair características
        features = []
        for i, mask in enumerate(masks):
            # Binarizar máscara
            binary_mask = (mask[0] > 0.5).astype(np.uint8)
            feature = self.feature_extractor.extract_features(image, binary_mask)
            feature['confidence'] = float(scores[i])
            features.append(feature)
        
        return {
            "masks": masks,
            "scores": scores,
            "boxes": boxes,
            "features": features,
            "image_path": image_path,
            "original_image": original_image
        }
    
    def analyze_directory(self, image_dir: str, output_dir: str = "./results"):
        """Analisar todas as imagens em um diretório"""
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        image_files = [f for f in os.listdir(image_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp'))]
        
        for img_file in tqdm(image_files, desc="Processando imagens"):
            try:
                img_path = os.path.join(image_dir, img_file)
                result = self.predict(img_path)
                
                # Salvar resultados
                result_file = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_results.json")
                with open(result_file, 'w') as f:
                    json.dump({
                        "image_path": result["image_path"],
                        "num_yeasts": len(result["features"]),
                        "features": result["features"]
                    }, f, indent=4, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
                
                results.append(result)
                
                # Visualizar resultados
                self.visualize_result(result, os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_visualization.jpg"))
                
            except Exception as e:
                print(f"Erro ao processar {img_file}: {str(e)}")
        
        # Gerar relatório
        self.generate_summary_report(results, output_dir)
        
        return results
    
    def visualize_result(self, result: Dict[str, Any], output_path: str):
        """Visualizar e salvar resultados"""
        image = result["original_image"]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Desenhar bounding boxes e máscaras
        for i, (box, mask) in enumerate(zip(result["boxes"], result["masks"])):
            # Bounding box
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Máscara
            binary_mask = (mask[0] > 0.5).astype(np.uint8)
            colored_mask = np.zeros_like(image)
            colored_mask[binary_mask > 0] = [0, 255, 0]  # Verde
            image = cv2.addWeighted(image, 1, colored_mask, 0.3, 0)
            
            # ID da levedura
            cv2.putText(image, f"{i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Salvar imagem
        cv2.imwrite(output_path, image)
    
    def generate_summary_report(self, results: List[Dict[str, Any]], output_dir: str):
        """Gerar relatório resumido"""
        all_features = []
        
        for result in results:
            for feature in result["features"]:
                feature["image"] = os.path.basename(result["image_path"])
                all_features.append(feature)
        
        if not all_features:
            print("Nenhuma característica extraída.")
            return
        
        # DataFrame com todas as características
        df = pd.DataFrame(all_features)
        
        # Salvar dados completos
        df.to_csv(os.path.join(output_dir, "all_yeast_features.csv"), index=False)
        
        # Estatísticas resumidas
        summary = df.describe()
        summary.to_csv(os.path.join(output_dir, "summary_statistics.csv"))
        
        print(f"Relatório gerado em: {output_dir}")
        print(f"Total de leveduras analisadas: {len(df)}")
        print(f"Número de imagens processadas: {len(results)}")
