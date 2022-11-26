## Proyecto YOLO V5: Aplicando un conjunto de datos persoanalizados y modelos de atención.

## Integrantes:

Jhon Harold Pineda Dorado
Miguel Caycedo Saa
Nombre: Harold Muñoz


## Definición: 

Herramienta para la detección de objetos usando la librería Yolo V5 y la aplicación de la "transferencia de conocimiento"; este material forma parte de la entrega final de materia "Visión Computacional con Deep Learning". 

Los objetos seleccionados para la tarea de detección son:

0 - Ambulance (1000 unds train - 200 unds test)
1 - Bus (1000 unds train - 200 unds test)
2 - Motorcycle (1000 unds train - 200 unds test)
3 - Train (1000 unds train - 200 unds test)
4 - Truck (1000 unds train - 200 unds test)

Cada imágen posee las respectivas anotaciones (bounding box) para cada clase.

## Despliegue:


El proyecto principal de YOLO V5 se debe obtener del repositorio de la empresa Ultralytics (https://github.com/ultralytics/yolov5)

Por lo anterior se debe clonar con base en la siguiente instrucción:

FOLDER_CLONE = "YOLOv5m"
!git clone https://github.com/ultralytics/yolov5  $FOLDER_CLONE


Puede consultar el ejemplo de la implementación en el archivo:

YOLOv5_MODULOS_ATENCION.ipynb 

Este se encuentra ubicado en el siguiente enlace:

https://drive.google.com/file/d/1bt3Exx7l_RNJG1sUk5EnpZVntWCzUi6S/view?usp=sharing


# Instalación de dependencias:

!pip install -qr requirements.txt  # install dependencies (ignore errors)
!pip install -q roboflow


# Importación de librerias:

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shutil
import torch
from IPython.display import Image, clear_output  # to display images
from utils.downloads import attempt_download  # to download models/datasets

# Instalación de la libreria de Grad-Cam

!pip install grad-cam

## DataSet:

El conjunto de datos se obtubo mediante la plataforma de Roboflow y se utilizó el formato de exportar datos para "YOLOv5 PyTorch".

Tenga en cuenta que la implementación de Ultralytics requiere un archivo YAML que defina dónde están sus datos de entrenamiento y de prueba. Al exportar los datos de Roboflow también descargamos este formato para nosotros


## Modifiación del proyecto original YOLO V5 para configurar los modelos de atención:

Editamos el archivo FOLDER_PROJECT+'/models/custom_model.yaml' con los siguientes parámetros:

#Parameters
nc: 5  # number of classes
depth_multiple: 1.0 #0.67  # model depth multiple
width_multiple: 1.0 # 0.75  # layer channel multiple

Donde n=5 es el número de clases de nuestro proyecto


# YOLOv5 + SE

Editamos el archivo FOLDER_PROJECT+'/models/common.py' agregando al final la siguiente clase:

class SENet(nn.Module):
    def __init__(self, c1,  ratio=16):
        super(SENet, self).__init__()
        #c*1*1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)

----------------------------------------------------------------


Realizamos una copia del archivo FOLDER_PROJECT+'/models/custom_model.yaml' y lo nombramos como FOLDER_PROJECT+'/models/model_SE.yaml'


Editamos el archivo FOLDER_PROJECT+'/models/model_SE.yaml' incluyendo la línea de código [-1, 1, SENet, [1024]]:

#YOLOv5 v6.0 backbone
backbone:
  #[from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SENet, [1024]], #SE standart
   [-1, 1, SPPF, [1024, 5]],  # 9
  ]


----------------------------------------------------------------


# YOLOv5 + CBAM


Editamos el archivo  FOLDER_PROJECT+'/models/common.py' agregando al final la siguiente clase:


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveMaxPool2d(1)

        self.fc1=nn.Conv2d(in_planes, in_planes//ratio, 1, bias=False)
        self.relu1=nn.ReLU()
        self.fc2=nn.Conv2d(in_planes//ratio, in_planes, 1, bias=False)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        avg_out=self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out=self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out=avg_out+max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1=nn.Conv2d(2, 1, kernel_size, padding=3, bias=False) #kernel size = 7 Padding isn 3: (n-7+1)+2P = n
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        avg_out=torch.mean(x, dim=1, keepdim=True)
        max_out,_=torch.max(x, dim=1, keepdim=True)
        x=torch.cat([avg_out, max_out], dim=1)
        x=self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channelIn):
        super(CBAM, self).__init__()
        self.channel_attention=ChannelAttention(channelIn)
        self.spatial_attention=SpatialAttention()

    def forward(self, x):
        out=self.channel_attention(x)*x
        #print('outchannels:{}'.format(out.shape))
        out=self.spatial_attention(out)*out
        return out


Realizamos una copia del archivo FOLDER_PROJECT+'/models/custom_model.yaml' y lo nombramos como FOLDER_PROJECT+'/models/model_CBAM.yaml'

Editamos el archivo FOLDER_PROJECT+'/models/model_CBAM.yaml':

#YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)
   [-1, 1, CBAM, [256]],

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)
   [-1, 1, CBAM, [512]],

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)
   [-1, 1, CBAM, [1024]],

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]

## Entrenamiento del Modelo YOLO V5

Luego de las modificaciones anteriores se realiza el entrenamiento del modelo de YOLO V5.

Se pueden descargar los modelos compilados desde la siguientes URLs:

- Ruta de los mejores pesos de entrenamiento YOLOv5m:

http://208.76.80.10/models_dl/best_yolov5m.pt


- Ruta de los mejores pesos de entrenamiento YOLOv5m + Módulo de atención SE:

http://208.76.80.10/models_dl/best_yolov5m_se.pt


- Ruta de los mejores pesos de entrenamiento YOLOv5m+Módulo de atención CBAM

http://208.76.80.10/models_dl/best_yolov5m_cbam.pt



## Grad-CAM

Para poder poner en marcha la herramienta de Grad-Cam es preciso codificar las siguientes lineas:

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
##import torch    
import cv2
##import numpy as np
import requests
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
#from PIL import Image

COLORS = np.random.uniform(0, 255, size=(80, 3))

def parse_detections(results):
    detections = results.pandas().xyxy[0]
    detections = detections.to_dict()
    boxes, colors, names = [], [], []

    for i in range(len(detections["xmin"])):
        confidence = detections["confidence"][i]
        if confidence < 0.2:
            continue
        xmin = int(detections["xmin"][i])
        ymin = int(detections["ymin"][i])
        xmax = int(detections["xmax"][i])
        ymax = int(detections["ymax"][i])
        name = detections["name"][i]
        category = int(detections["class"][i])
        color = COLORS[category]

        boxes.append((xmin, ymin, xmax, ymax))
        colors.append(color)
        names.append(name)
    return boxes, colors, names


def draw_detections(boxes, colors, names, img):
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            color, 
            2)

        cv2.putText(img, name, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return img

def renormalize_cam_in_bounding_boxes(boxes, colors, names, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1] 
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    image_with_bounding_boxes = draw_detections(boxes, colors, names, eigencam_image_renormalized)
    return image_with_bounding_boxes    


# Evaluar la clase 0: Ambulance con Grad-Cam (tomando una de las imagenes)

from PIL import Image, ImageDraw
img = np.array(Image.open('data/images/0_ambulance_1.jpg'))
img = cv2.resize(img, (640, 640))
rgb_img = img.copy()
img = np.float32(img)/255
transform = transforms.ToTensor()
tensor = transform(img).unsqueeze(0)

path_yolo='/content/drive/MyDrive/YOLOv5m'
#path_own_model='yolov5s_gun'
best_weights='runs/train/yolov5m_results/weights/best.pt'
model=torch.hub.load(path_yolo,'custom', best_weights,source='local')
model.eval()
model.cpu()
target_layers=[model.model.model.model[-2]]

results=model([rgb_img])
boxes, colors, names = parse_detections(results)
detections = draw_detections(boxes, colors, names, rgb_img.copy())
Image.fromarray(detections)


--------------------------------------------------------------------------





## Conclusiones: 

* El modelo de yolo v5, ha demostrado tener un mejor rendimiento para detectar trenes y ambulancias, ambas con una precision del 74%, por otro lado se debe destacar que el modelo tiene limitaciones al momento de predecir camiones, cuya precisión es de apenas el 52%.

* Se observa que el modelo yolov5 en las imagenes que etiquetó adecuadamente como ambulancias, la concentración de la red abarca especificamente el lugar donde se ubica el objeto, y al implementar el boundding box la concentracion se hace mas evidente sobre el objeto.

* El modelo para la categoria de buses se ha mostrado tener concentración en general sobre el objeto, aunque en una de las imagenes de prueba el modelo solo ubicó un pequeño mapa de calor sobre el objeto cuando se hacía uso del bounding box.

* En la categoria de motocicletas el modelo fue capaz de fijar su atencion en el lugar del objeto, sin embargo se observa que los bounding box tienden a abarcar zonas correspondientes a la posicion del conductor.

* A pesar de que es una de las categorias que mejor es predicha, se observa que hay ocasiones en donde la atencion de la red no se encuentra especificamente en el lugar donde se encuentra ubicado el objeto.

* Apesar de ser los camiones la categoria de menos precision, el modelo logra fijar su atención en zonas en las cuales el objeto se encuentra ubicado, asi mismo, los bounding box estan especificamente en el lugar del objeto.


