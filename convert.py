import os
import xml.etree.ElementTree as ET
import random
from shutil import copy2

# ============ 配置区域 ============
DATA_PATH = "."  # 数据集所在目录（脚本放在数据集文件夹里运行）
IMAGES_DIR = "images"      # 原始图片文件夹
ANNOTATIONS_DIR = "annotations"  # XML标注文件夹

# 类别映射（根据实际数据调整，这3个是安全帽数据集的类别）
CLASSES = ['helmet']

# 数据集划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
# ================================

def convert_xml_to_yolo(xml_path, img_width, img_height):
    """将XML标注转换为YOLO格式的归一化坐标"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    yolo_labels = []
    for obj in root.findall('object'):
        # 获取类别
        class_name = obj.find('name').text
        if class_name not in CLASSES:
            continue
        class_id = CLASSES.index(class_name)
        
        # 获取边界框坐标
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # 转换为YOLO格式：中心点坐标 + 宽高，都归一化到0~1
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_labels

# 创建输出目录
os.makedirs("yolo_dataset/images/train", exist_ok=True)
os.makedirs("yolo_dataset/images/val", exist_ok=True)
os.makedirs("yolo_dataset/images/test", exist_ok=True)
os.makedirs("yolo_dataset/labels/train", exist_ok=True)
os.makedirs("yolo_dataset/labels/val", exist_ok=True)
os.makedirs("yolo_dataset/labels/test", exist_ok=True)

# 获取所有图片文件
image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 随机打乱并划分
random.shuffle(image_files)
total = len(image_files)
train_end = int(total * TRAIN_RATIO)
val_end = int(total * (TRAIN_RATIO + VAL_RATIO))

train_files = image_files[:train_end]
val_files = image_files[train_end:val_end]
test_files = image_files[val_end:]

print(f"总共 {total} 张图片")
print(f"训练集: {len(train_files)} 张")
print(f"验证集: {len(val_files)} 张")
print(f"测试集: {len(test_files)} 张")

# 处理所有图片
for split, file_list in [('train', train_files), ('val', val_files), ('test', test_files)]:
    for img_file in file_list:
        # 复制图片
        src_img = os.path.join(IMAGES_DIR, img_file)
        dst_img = os.path.join(f"yolo_dataset/images/{split}", img_file)
        copy2(src_img, dst_img)
        
        # 找到对应的XML文件
        xml_file = img_file.replace('.jpg', '.xml').replace('.png', '.xml').replace('.jpeg', '.xml')
        xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)
        
        if not os.path.exists(xml_path):
            print(f"警告: 找不到 {xml_path}")
            continue
        
        # 获取图片尺寸
        from PIL import Image
        img = Image.open(src_img)
        img_width, img_height = img.size
        
        # 转换标注
        yolo_labels = convert_xml_to_yolo(xml_path, img_width, img_height)
        
        # 保存TXT文件
        txt_file = img_file.rsplit('.', 1)[0] + '.txt'
        txt_path = os.path.join(f"yolo_dataset/labels/{split}", txt_file)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_labels))
        
        print(f"已处理: {img_file}")

# 生成data.yaml配置文件
yaml_content = f"""
path: {os.path.abspath('yolo_dataset')}
train: images/train
val: images/val
test: images/test

nc: {len(CLASSES)}
names: {CLASSES}
"""

with open('yolo_dataset/data.yaml', 'w') as f:
    f.write(yaml_content)

print("\n✅ 转换完成！")
print(f"数据集位置: yolo_dataset/")
print(f"配置文件: yolo_dataset/data.yaml")
