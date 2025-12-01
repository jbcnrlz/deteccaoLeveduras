from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import torch

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = F.to_pil_image(img)
        axs[0, i].imshow(img)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

def main():
    
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms = weights.transforms()

    output = []    
    original_images = []
    for i in ['fonte.jpg','fonte2.jpg']:
        im = Image.open(i)
        original_images.append(im)
        output.append(transforms(im))

    model = maskrcnn_resnet50_fpn(weights=weights, progress=False)
    model = model.eval()
    out = model(output)
    print(out)

    # Desenhar bounding boxes nas imagens ORIGINAIS
    images_with_boxes = []
    score_threshold = 0.5
    
    for i, (original_img, output) in enumerate(zip(original_images, out)):
        # Converter imagem original para tensor uint8
        img_tensor = (F.to_tensor(original_img) * 255).byte()
        
        # Filtrar detecções por score
        high_score_indices = output['scores'] > score_threshold
        
        if high_score_indices.any():
            # Desenhar bounding boxes na imagem ORIGINAL
            boxes = output['boxes'][high_score_indices]
            img_with_boxes = draw_bounding_boxes(
                img_tensor,
                boxes=boxes,
                width=4,
                colors="red"
            )
        else:
            # Se não houver detecções, usar imagem original
            img_with_boxes = img_tensor
        
        images_with_boxes.append(img_with_boxes)

    # Mostrar resultados
    show(images_with_boxes)


if __name__ == "__main__":
    main()