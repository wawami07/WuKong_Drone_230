import os 
import numpy as np
import json
import shutil 

def mkdir(file_path):
    """创建目录（若不存在），增加日志便于排查"""
    if not os.path.exists(file_path):
        os.makedirs(file_path) 
        print(f"已创建目录: {file_path}")

def read_pcd(filepath):
    """读取PCD文件（支持ASCII和二进制格式，解决解码错误）"""
    if not os.path.exists(filepath):
        print(f"错误：PCD文件不存在 -> {filepath}")
        return np.empty((0, 4), dtype=np.float32)

    with open(filepath, 'rb') as f:
        # 1. 解析PCD头部，确定数据格式
        header = {}
        while True:
            line = f.readline().strip()
            if not line:
                continue
            # 头部是ASCII文本，解码获取格式信息
            try:
                line_str = line.decode('utf-8')
            except UnicodeDecodeError:
                print(f"错误：PCD头部无效 -> {filepath}")
                return np.empty((0, 4), dtype=np.float32)
            
            if line_str.startswith('DATA'):
                header['format'] = line_str.split()[-1]  # 记录格式（ascii/binary）
                break
            # 提取点数（用于二进制格式计算）
            if ':' in line_str:
                key, val = line_str.split(':', 1)
                if key.strip().lower() == 'width':
                    header['width'] = int(val.strip())
                elif key.strip().lower() == 'height':
                    header['height'] = int(val.strip())

        # 2. 按格式读取点云数据
        if header.get('format') == 'ascii':
            # ASCII格式：逐行解析x,y,z,intensity
            points = []
            while True:
                line = f.readline().strip()
                if not line:
                    break
                try:
                    parts = line.decode('utf-8', errors='ignore').split()
                    if len(parts) >= 4:
                        points.append([
                            float(parts[0]), float(parts[1]), 
                            float(parts[2]), float(parts[3])
                        ])
                except ValueError:
                    print(f"警告：跳过无效ASCII行 -> {line}")
            return np.array(points, dtype=np.float32).reshape(-1, 4)
        
        elif header.get('format') in ['binary', 'binary_compressed']:
            # 二进制格式：直接按字节读取（float32类型）
            total_points = header.get('width', 0) * header.get('height', 0)
            if total_points == 0:
                print(f"警告：PCD点数为0 -> {filepath}")
                return np.empty((0, 4), dtype=np.float32)
            # 每个点4个float32字段（x,y,z,intensity），共 4*4=16字节
            data = np.fromfile(f, dtype=np.float32, count=total_points * 4)
            return data.reshape(-1, 4)
        
        else:
            print(f"警告：不支持的PCD格式 -> {header.get('format')}")
            return np.empty((0, 4), dtype=np.float32)

def pcd2npy(pcdfolder, npyfolder):
    """将scene/lidar下的PCD转为NPY（OpenPCDet需要）"""
    mkdir(npyfolder)
    if not os.path.exists(pcdfolder):
        print(f"错误：PCD目录不存在 -> {pcdfolder}")
        return

    # 只处理.pcd文件，避免其他文件干扰
    pcd_files = [f for f in os.listdir(pcdfolder) if f.lower().endswith('.pcd')]
    if len(pcd_files) == 0:
        print(f"警告：scene/lidar下无.pcd文件 -> {pcdfolder}")
        return

    for pcd_file in pcd_files:
        filename = os.path.splitext(pcd_file)[0]
        pcd_path = os.path.join(pcdfolder, pcd_file)
        npy_path = os.path.join(npyfolder, f"{filename}.npy")

        # 读取并保存NPY
        point_cloud = read_pcd(pcd_path)
        if point_cloud.shape[0] == 0:
            print(f"跳过空文件 -> {pcd_file}")
            continue
        np.save(npy_path, point_cloud)
        print(f"已保存NPY：{filename}.npy")

def json2txt(json_folder, txt_folder):
    """将scene/label下的JSON转为TXT标签（OpenPCDet格式）"""
    mkdir(txt_folder)
    if not os.path.exists(json_folder):
        print(f"错误：JSON目录不存在 -> {json_folder}")
        return

    # 只处理.json文件
    json_files = [f for f in os.listdir(json_folder) if f.lower().endswith('.json')]
    if len(json_files) == 0:
        print(f"警告：scene/label下无.json文件 -> {json_folder}")
        return

    for json_file in json_files:
        filename = os.path.splitext(json_file)[0]
        json_path = os.path.join(json_folder, json_file)
        txt_path = os.path.join(txt_folder, f"{filename}.txt")

        # 读取JSON并解析标签
        try:
            with open(json_path, 'r', encoding='utf8') as fp:
                json_data = json.load(fp)
        except json.JSONDecodeError:
            print(f"跳过无效JSON -> {json_file}")
            continue

        # 清空TXT避免重复写入
        with open(txt_path, 'w') as f:
            pass

        # 写入标签（格式：x y z dx dy dz yaw 类别）
        for obj in json_data:
            try:
                line = f"{obj['psr']['position']['x']} " \
                       f"{obj['psr']['position']['y']} " \
                       f"{obj['psr']['position']['z']} " \
                       f"{obj['psr']['scale']['x']} " \
                       f"{obj['psr']['scale']['y']} " \
                       f"{obj['psr']['scale']['z']} " \
                       f"{obj['psr']['rotation']['z']} " \
                       f"{obj['obj_type']}\n"
                with open(txt_path, 'a') as f:
                    f.write(line)
            except KeyError as e:
                print(f"JSON缺少字段 {e} -> {json_file}")
        print(f"已保存TXT：{filename}.txt")

def gen_train_test_split(pcdfolder, output_dir, test_ratio=0.1):
    """生成训练集/测试集列表（train.txt/val.txt）"""
    mkdir(output_dir)
    if not os.path.exists(pcdfolder):
        print(f"错误：PCD目录不存在 -> {pcdfolder}")
        return

    # 从PCD文件名获取样本名（确保与标签对应）
    sample_names = [os.path.splitext(f)[0] for f in os.listdir(pcdfolder) if f.lower().endswith('.pcd')]
    if len(sample_names) == 0:
        print(f"错误：无PCD样本可划分 -> {pcdfolder}")
        return

    # 固定随机种子，结果可复现
    np.random.seed(42)
    shuffled = np.random.permutation(sample_names)
    test_size = int(len(shuffled) * test_ratio)
    train = shuffled[test_size:]
    test = shuffled[:test_size]

    # 写入train.txt和val.txt
    with open(os.path.join(output_dir, "train.txt"), 'w') as f:
        f.write('\n'.join(train))
    with open(os.path.join(output_dir, "val.txt"), 'w') as f:
        f.write('\n'.join(test))

    print(f"\n划分完成：训练集{len(train)}个，测试集{len(test)}个")
    print(f"训练集列表：{os.path.join(output_dir, 'train.txt')}")
    print(f"测试集列表：{os.path.join(output_dir, 'val.txt')}")

def gen_scene_dataset(scene_root, output_root, test_ratio=0.1):
    """
    核心函数：仅处理scene目录，生成OpenPCDet格式数据集
    scene_root: scene目录路径（必须包含lidar和label子目录）
    output_root: 最终数据集输出路径（生成points/labels/ImageSets）
    """
    print(f"=== 开始处理scene目录：{scene_root} ===")
    # 1. 明确scene下的关键路径（直接定位，不遍历其他目录）
    pcd_dir = os.path.join(scene_root, "lidar")       # scene的PCD目录
    json_label_dir = os.path.join(scene_root, "label") # scene的JSON标签目录
    points_dir = os.path.join(output_root, "points")   # 输出NPY点云
    labels_dir = os.path.join(output_root, "labels")   # 输出TXT标签
    imagesets_dir = os.path.join(output_root, "ImageSets") # 输出训练测试列表

    # 2. 检查scene目录结构是否正确
    if not os.path.exists(pcd_dir):
        print(f"错误：scene下缺少lidar目录 -> {pcd_dir}")
        return
    if not os.path.exists(json_label_dir):
        print(f"错误：scene下缺少label目录 -> {json_label_dir}")
        return

    # 3. 逐步生成数据集
    pcd2npy(pcd_dir, points_dir)                  # PCD转NPY
    json2txt(json_label_dir, labels_dir)          # JSON转TXT
    gen_train_test_split(pcd_dir, imagesets_dir, test_ratio) # 划分训练测试集

    print(f"\n=== 处理完成！输出目录：{output_root} ===")
    print(f"点云（NPY）：{points_dir}")
    print(f"标签（TXT）：{labels_dir}")
    print(f"训练测试列表：{imagesets_dir}")

if __name__ == '__main__':
    # -------------------------- 请根据你的实际路径修改这两个参数 --------------------------
    SCENE_ROOT = "/home/ubuntu/SUSTechPOINTS/data/scene"  # 你的scene目录（必须含lidar和label）
    OUTPUT_ROOT = "/home/ubuntu/SUSTechPOINTS/data/scene_dataset"  # 最终输出目录（可自定义）
    TEST_RATIO = 0.1  # 测试集比例（10%测试，90%训练，可调整）
    # -----------------------------------------------------------------------------------

    # 仅处理scene目录，执行生成
    gen_scene_dataset(SCENE_ROOT, OUTPUT_ROOT, TEST_RATIO)

