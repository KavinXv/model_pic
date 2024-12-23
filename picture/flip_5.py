import cv2
import os

def flip_image_horizontal(image):
    # 对图像进行左右翻转

    flipped_image = cv2.flip(image, 1)  # 1表示水平翻转
    return flipped_image

def process_images_with_flip(input_folder, output_folder):
    # 遍历输入文件夹下的所有子文件夹和文件
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".jpg"):  
                # 构建原始图像路径
                img_path = os.path.join(root, file)

                # 读取图像
                img = cv2.imread(img_path)


                # 对图像进行左右翻转
                flipped_img = flip_image_horizontal(img)

                # 构建输出文件夹路径
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)

                # 如果输出文件夹不存在，创建它
                os.makedirs(output_dir, exist_ok=True)

                # 构建输出图像路径，并在文件名后加上 "5"
                file_name, file_ext = os.path.splitext(file)
                output_file_name = f"{file_name}_5{file_ext}"
                output_img_path = os.path.join(output_dir, output_file_name)

                # 保存图像
                cv2.imwrite(output_img_path, flipped_img)

                print(f"图像保存到: {output_img_path}")

if __name__ == "__main__":
    # 输入原始图像所在文件夹
    input_folder = r'D:\vscode_code\Deep_learning\DATASET\data2_2\test'  # 替换为你输入的大文件夹路径

    # 输出图像保存位置
    output_folder = r'D:\vscode_code\Deep_learning\DATASET\data2_2_5\test'  # 替换为你输出的大文件夹路径

    # 调用函数
    process_images_with_flip(input_folder, output_folder)
