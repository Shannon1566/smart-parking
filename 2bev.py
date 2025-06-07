# 读取图片
import cv2
import numpy as np
import os

import numpy as np
import cv2

def get_bev(img, K, T, bev_res=0.005, bev_x_range=(-3, 3), bev_y_range=(1.2, 5)):
    """
    根据相机内参 K 和外参 T，从图像生成鸟瞰图。
    
    参数:
    - img: 原始图像 (H x W x 3)
    - K: 相机内参 (3x3)
    - T: 相机外参 (4x4)，表示从世界坐标到相机坐标的变换矩阵
    - bev_res: BEV图像分辨率（米/像素）
    - bev_x_range: 前后方向范围（单位：米）
    - bev_y_range: 左右方向范围（单位：米）
    
    返回:
    - bev_img: 鸟瞰图（BEV图像）
    """
    # 生成BEV图像网格
    x_min, x_max = bev_x_range
    y_min, y_max = bev_y_range
    xs = np.arange(x_min, x_max, bev_res)
    ys = np.arange(y_min, y_max, bev_res)
    xs, ys = np.meshgrid(xs, ys)
    zs = np.zeros_like(xs)

    # 拼接为形状(N, 3) 的世界坐标点 (Z=0)
    world_points = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)  # (N, 3)
    world_points_h = np.concatenate([world_points, np.ones((world_points.shape[0], 1))], axis=1)  # (N, 4)

    # 世界坐标转相机坐标
    cam_points_h = (T @ world_points_h.T).T  # (N, 4)
    cam_points = cam_points_h[:, :3]

    # 过滤掉Z <= 0的点（在相机后方）
    mask = cam_points[:, 2] > 0
    cam_points = cam_points[mask]

    # 相机坐标投影到像素坐标
    pixels = (K @ cam_points.T).T  # (N, 3)
    pixels = pixels[:, :2] / pixels[:, 2:3]  # 像素坐标 (x/z, y/z)

    # 过滤出图像内的像素
    h, w = img.shape[:2]
    x_pix = pixels[:, 0]
    y_pix = pixels[:, 1]
    in_img = (x_pix >= 0) & (x_pix < w) & (y_pix >= 0) & (y_pix < h)

    pixels = pixels[in_img]
    world_points_valid = world_points[mask][in_img]

    # 把图像上的颜色采样过来
    sampled_colors = img[y_pix[in_img].astype(np.int32), x_pix[in_img].astype(np.int32)]

    # 创建BEV图像并填入颜色
    bev_h = int((y_max - y_min) / bev_res)
    bev_w = int((x_max - x_min) / bev_res)
    bev_img = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)

    # 映射 world_points_valid 到 bev 图像像素坐标
    u = ((world_points_valid[:, 0] - x_min) / bev_res).astype(np.int32)
    v = ((world_points_valid[:, 1] - y_min) / bev_res).astype(np.int32)

    # 因为图像坐标系是从上到下，所以我们需要上下翻转y轴
    v = bev_h - 1 - v

    valid_mask = (u >= 0) & (u < bev_w) & (v >= 0) & (v < bev_h)
    bev_img[v[valid_mask], u[valid_mask]] = sampled_colors[valid_mask]

    return bev_img


if __name__ == "__main__":
    # 检查图片文件
    img_path = 'img/test1.jpg'
    
    if not os.path.exists(img_path):
        print(f"错误: 图片文件 '{img_path}' 不存在")
        print("当前工作目录:", os.getcwd())
        exit(1)
    
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"错误: 无法读取图片文件 '{img_path}'")
        exit(1)
    
    print(f"成功加载图片，尺寸: {img.shape}")

    # 相机内参矩阵
    K = np.array([
        [1672.20958685379, 0, 967.185727369945],
        [0, 1677.94918818452, 531.923779475263],
        [0, 0, 1]
    ])

    # 外参变换矩阵 (相机到世界坐标系的变换)
    T = np.array([
        [0.998786310536022, 0.0492481623782281, 0.000724146538345100, -0.0334929733014370],
        [0.00965195661778823, -0.181288572393349, -0.983382577257208, 0.450112772104900],
        [-0.0482985053525433, 0.982196045615105, -0.181543885489876, 0.455995433493428],
        [0, 0, 0, 1]
    ])
    
    # 平移地面高度
    T[2][3] = T[2][3] - 1
    
    print("开始生成鸟瞰图...")
    bev = get_bev(img, K, T)
    
    # 确保输出目录存在
    os.makedirs('img', exist_ok=True)
    
    # cv2.imshow('Original Image', img)
    # cv2.imshow('Bird Eye View', bev)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # 保存结果
    output_path = 'img/test1_bev.png'
    cv2.imwrite(output_path, bev)
    print(f"鸟瞰图已保存到: {output_path}")
