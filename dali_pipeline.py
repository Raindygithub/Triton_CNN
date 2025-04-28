# 新建dali_pipeline.py
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch

# 修改 dali_pipeline.py 中的读取部分
@pipeline_def(batch_size=256, num_threads=8, device_id=0)
def cifar10_lmdb_pipeline(lmdb_path, is_training=True):
    # 使用通用文件读取器（假设LMDB中存储的是键值对：键为文件名，值为序列化的图像和标签）
    files, labels = fn.readers.file(
        file_root=lmdb_path,
        random_shuffle=is_training,
        name="Reader"
    )
    
    # 解码图像
    images = fn.decoders.image(
        files,
        device="mixed" if torch.cuda.is_available() else "cpu",
        output_type=types.RGB
    )
    
    # 3. 数据增强 -------------------------------------
    # 随机裁剪（训练集使用）
    images = fn.random_resized_crop(
        images,
        size=[32, 32],
        random_area=[0.8, 1.0],
        random_aspect_ratio=[0.9, 1.1]
    )
    
    # 随机水平翻转（50%概率）
    images = fn.flip(
        images,
        horizontal=fn.random.coin_flip(probability=0.5)
    )
    
    # 4. 归一化处理 -----------------------------------
    images = fn.crop_mirror_normalize(
        images,
        mean=[0.4914*255, 0.4822*255, 0.4465*255],  # CIFAR-10均值
        std=[0.2023*255, 0.1994*255, 0.2010*255],    # CIFAR-10标准差
        dtype=types.FLOAT,
        output_layout=types.NCHW  # 输出形状为[Batch, Channel, Height, Width]
    )
    
    # 5. 标签处理 -------------------------------------
    labels = labels.gpu()  # 将标签转移到GPU
    labels = fn.squeeze(labels, axes=[])  # 去除冗余维度
    
    return images, labels