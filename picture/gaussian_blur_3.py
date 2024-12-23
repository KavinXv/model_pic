import cv2
import os

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    # 对图像进行高斯模糊处理
    # 对图像中的每个像素点，以其周围像素的加权平均值来替代原始值

    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred_image

def process_images_with_blur(input_folder, output_folder, kernel_size=(9, 9), sigma=10.0):
    # 遍历输入文件夹下的所有子文件夹和文件
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".jpg"): 
                # 构建原始图像路径
                img_path = os.path.join(root, file)

                # 读取图像
                img = cv2.imread(img_path)


                # 对图像进行高斯模糊处理
                blurred_img = apply_gaussian_blur(img, kernel_size, sigma)

                # 构建输出文件夹路径
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)

                # 如果输出文件夹不存在，创建它
                os.makedirs(output_dir, exist_ok=True)

                # 构建输出图像路径，并在文件名后加上 "3"
                file_name, file_ext = os.path.splitext(file)
                output_file_name = f"{file_name}_3{file_ext}"
                output_img_path = os.path.join(output_dir, output_file_name)

                # 保存图像
                cv2.imwrite(output_img_path, blurred_img)

                print(f"图像保存到: {output_img_path}")

if __name__ == "__main__":
    # 输入原始图像所在文件夹
    input_folder = r'D:\vscode_code\Deep_learning\DATASET\data2_2\test'  # 替换为你输入的大文件夹路径

    # 输出图像保存位置
    output_folder = r'D:\vscode_code\Deep_learning\DATASET\data2_2_3\test'  # 替换为你输出的大文件夹路径

    # 参数
    kernel_size = (5, 5) 
    sigma = 0  # 标准差，0 

    # 调用函数
    process_images_with_blur(input_folder, output_folder, kernel_size, sigma)
