# Product Detection 

##### Summary Table

|      | |
| ---------- |-------------------|
| **Author**       | Ultralytics LLC|
| **Title**        | YOLO v.5 real time object detection |
| **Topics**       | Ứng dụng trong computer vision, sử dụng thuật toán chính là CNN|
| **Descriptions** | Input sẽ là các tấm hình và file .txt có tên tương ứng và chứa 5 thông số của object. đầu tiên là ```<class object>``` và ```<x, y, width, height>``` của bounding box chứa vật. khi train xong sẽ trả ra output là file trọng số ```weights```. Ta sẽ sử dụng trọng số ```weights``` đã train để predict bounding box và class của các object trong hình|
| **Links**        | https://github.com/ultralytics/yolov5|
| **Framework**    | PyTorch|
| **Pretrained Models**  | 
| **Datasets**     |Mô hình được train với bộ dữ liệu SKU110K. 
| **Level of difficulty**|Sử dụng nhanh và dễ, có thể train lại với tập dữ liệu khác tốc độ tùy thuộc vào phần cứng và hình ảnh input|


[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VUwsyUc9PRuDfqxSfzai9QKavedlSVMh#scrollTo=_BK-VVOJpsjz)


# Introduction
  Trong kho lưu trữ này, tôi giới thiệu một ứng dụng của phiên bản YOLO mới nhất, tức là YOLOv5, để phát hiện các mặt hàng có trong kệ cửa hàng bán lẻ. Ứng dụng này có thể được sử dụng để theo dõi số lượng hàng tồn kho bằng cách sử dụng hình ảnh của các mặt hàng trên kệ.
  
  ![Caption for the picture.](https://raw.githubusercontent.com/shayanalibhatti/Retail-Store-Item-Detection-using-YOLOv5/master/results.png)
  
# Dataset
[SKU110K Dataset]( http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz)
Tập dữ liệu SKU110k dựa trên hình ảnh của các đối tượng bán lẻ trong bối cảnh đông đúc. Nó cung cấp hình ảnh tập hợp đào tạo, xác thực và thử nghiệm và các tệp .csv tương ứng chứa thông tin về vị trí hộp giới hạn của tất cả các đối tượng trong các hình ảnh đó. Tệp .csv có thông tin hộp giới hạn đối tượng được viết trong các cột sau:

image_name,x1,y1,x2,y2,class,image_width,image_height

Trong đó x1, y1 là tọa độ trên cùng bên trái của hộp giới hạn và x2, y2 là tọa độ dưới cùng bên phải của hộp giới hạn, phần còn lại của các tham số là tự giải thích. Dưới đây là ví dụ về các tham số của hình ảnh train_0.jpg cho một hộp giới hạn. Có một số hộp giới hạn cho mỗi hình ảnh, một hộp cho mỗi đối tượng.

train_0.jpg, 208, 537, 422, 814, object, 3024, 3024

Trong tập dữ liệu SKU110k, chúng tôi có 2940 hình ảnh trong tập hợp train, 8232 hình ảnh trong tập hợp validation và 587 hình ảnh trong tập test. Mỗi hình ảnh có thể có số lượng đối tượng khác nhau, do đó, số lượng hộp giới hạn khác nhau.

### Preprocessing

Xử lý trước hình ảnh bao gồm thay đổi kích thước chúng thành 416x416x3. Điều này được thực hiện trên nền tảng của Roboflow. Hình ảnh được chú thích, thay đổi kích thước được hiển thị trong hình dưới đây:

![Caption for the picture.](https://raw.githubusercontent.com/shayanalibhatti/Retail-Store-Item-Detection-using-YOLOv5/master/roboflow_data_image_annotated.jpg)

### Automatic Annotation

Trên trang web Roboflow.ai, tệp .csv chú thích hộp giới hạn và hình ảnh từ tập huấn luyện được tải lên và dịch vụ chú thích của Roboflow.ai tự động vẽ hộp giới hạn trên hình ảnh bằng cách sử dụng chú thích được cung cấp trong tệp .csv như được hiển thị trong hình trên.

### Data Generation

 70–20–10 training-validation-test set split
 
### Hardware Used

Trained on Google Colab with GPU Tesla T4

# Code

I started by cloning YOLOv5 and installing the dependencies mentioned in requirements.txt file

```sh
!git clone https://github.com/ultralytics/yolov5  # clone repo
!pip install -r yolov5/requirements.txt  # install dependencies
%cd yolov5
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
```

#### Create file Data.yaml

```sh
%%writetemplate /content/yolov5/models/data.yaml

train: /content/drive/MyDrive/product_data /train/images
val: /content/drive/MyDrive/product_data /valid/images

# number of classes
nc: 1

# class names
names: ['Object']
```


#### Custom model.yaml

```sh
%%writetemplate /content/yolov5/models/custom_yolov5s.yaml

# parameters
nc: {num_classes}  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, BottleneckCSP, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 6, BottleneckCSP, [1024, False]],  # 9
  ]

# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, BottleneckCSP, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

## Train

Now I start the training process. I defined the image size (img) to be 416x416, batch size 32 and the model is run for 300 epochs. If we dont define weights, they are initialized randomly. 

```sh
%%time
%cd /content/yolov5/
!python train.py --img 416 --batch 32 --epochs 300 --data '/content/yolov5/models/data.yaml' --cfg ./models/custom_yolov5s.yaml --weights '' --name yolov5s_results --nosave  --cache
```


#### Observations
Visualize important evaluation metrics after the model has been trained using the following code:

```sh
from utils.plots import plot_results  # plot results.txt as results.png
Image(filename='/content/yolov5/runs/train/yolov5s_results2/results.png', width=1000)
```
The following 3 parameters are commonly used for object detection tasks: · GIoU is the Generalized Intersection over Union which tells how close to the ground truth our bounding box is. · Objectness shows the probability that an object exists in an image. Here it is used as loss function. · mAP is the mean Average Precision telling how correct are our bounding box predictions on average. It is area under curve of precision-recall curve. It is seen that Generalized Intersection over Union (GIoU) loss and objectness loss decrease both for training and validation. Mean Average Precision (mAP) however is at 0.7 for bounding box IoU threshold of 0.5. Recall stands at 0.8 as shown below:

![Caption for the picture.](https://raw.githubusercontent.com/shayanalibhatti/Retail-Store-Item-Detection-using-YOLOv5/master/observations.png)
## Results

Các hình ảnh sau đây cho thấy kết quả của thuật toán YOLOv5 được đào tạo để vẽ các hộp giới hạn trên các đối tượng. Kết quả là khá tốt.

![Caption for the picture.](https://raw.githubusercontent.com/shayanalibhatti/Retail-Store-Item-Detection-using-YOLOv5/master/result1.jpg)

Hình ảnh bộ thử nghiệm gốc (bên trái) và các hộp giới hạn được vẽ bởi YOLOv5 (bên phải)

## Conclusion

Ứng dụng phát hiện đối tượng bán lẻ này có thể được sử dụng để theo dõi hàng tồn kho trên kệ hàng hoặc cho một khái niệm cửa hàng thông minh, nơi mọi người chọn đồ và tự động bị tính phí. Kích thước trọng lượng nhỏ và tốc độ khung hình tốt của YOLOv5 sẽ mở đường trở thành lựa chọn hàng đầu cho các nhiệm vụ phát hiện đối tượng thời gian thực dựa trên hệ thống nhúng.



