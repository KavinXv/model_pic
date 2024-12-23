import cv2
import os
# 用于文件操作
import shutil

def histogram_equalization_color(input_folder, output_folder):
    # 遍历输入文件夹下的所有子文件夹和文件
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".jpg"):
                # 构建原始图像路径
                img_path = os.path.join(root, file)

                # 读取图像
                img = cv2.imread(img_path)


                # 将图像从 BGR 颜色空间转换到 YUV 颜色空间
                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

                # 对 Y 通道（亮度）进行直方图均衡化
                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

                # 将图像从 YUV 颜色空间转换回 BGR 颜色空间
                img_equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                # 构建输出文件夹路径
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)

                # 如果输出文件夹不存在，创建它
                os.makedirs(output_dir, exist_ok=True)

                # 构建输出图像路径，并在文件名后加上 "1"
                file_name, file_ext = os.path.splitext(file)
                output_file_name = f"{file_name}_1{file_ext}"
                output_img_path = os.path.join(output_dir, output_file_name)

                # 保存图像
                cv2.imwrite(output_img_path, img_equalized)

                print(f"图像保存到: {output_img_path}")

if __name__ == "__main__":
    # 输入原始图像所在文件夹
    input_folder = r'D:\vscode_code\Deep_learning\DATASET\data2_2\train' 

    # 输出图像保存位置
    output_folder =  r'D:\vscode_code\Deep_learning\DATASET\data2_2_1\train' 

    # 调用函数
    histogram_equalization_color(input_folder, output_folder)
