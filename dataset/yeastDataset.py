import os, json, cv2, torch, numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class YeastDataset(Dataset):
    """Dataset personalizado para imagens de leveduras"""
    def __init__(self, root_dir, transforms=None, annotation_dir=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.annotation_dir = annotation_dir
        
        # Listar todas as imagens
        self.imgs = [f for f in os.listdir(root_dir) 
                    if f.endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp'))]
        
        # Carregar anotações se disponíveis
        self.annotations = self._load_annotations() if annotation_dir else None
        
    def _load_annotations(self):
        """Carregar anotações do diretório"""
        annotations = {}
        if self.annotation_dir and os.path.exists(self.annotation_dir):
            for img_name in self.imgs:
                base_name = os.path.splitext(img_name)[0]
                annotation_path = os.path.join(self.annotation_dir, f"{base_name}.json")
                
                if os.path.exists(annotation_path):
                    with open(annotation_path, 'r') as f:
                        annotations[img_name] = json.load(f)
        return annotations
    
    def __getitem__(self, idx):
        # Carregar imagem
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # Converter para tensor
        img = F.to_tensor(img)
        
        # Se temos anotações, usá-las
        if self.annotations and img_name in self.annotations:
            annotation = self.annotations[img_name]
            boxes = []
            masks = []
            labels = []
            
            for obj in annotation.get('objects', []):
                # Caixas delimitadoras
                boxes.append(obj['bbox'])
                
                # Máscaras (se disponíveis)
                if 'segmentation' in obj:
                    mask = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)
                    # Implementar decodificação de máscaras se necessário
                    masks.append(mask)
                
                labels.append(1)  # Classe 1 para levedura
            
            if boxes:
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
                
                if masks:
                    masks = torch.as_tensor(masks, dtype=torch.uint8)
                
                target = {
                    "boxes": boxes,
                    "labels": labels,
                    "image_id": torch.tensor([idx]),
                    "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                    "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
                }
                
                if masks:
                    target["masks"] = masks
                    
                return img, target
        
        # Se não há anotações, retornar imagem vazia
        return img, {}
    
    def __len__(self):
        return len(self.imgs)
