from PIL import Image
import os
objects_folder = "C:\\Users\\12178\\Desktop\\大论文代码\\第四章内容\\bear\\object_mask"
image_name = "frame_00021"
object_path = os.path.join(objects_folder, image_name + '.png')
objects = Image.open(object_path) if os.path.exists(object_path) else None
print(objects.size)  # (985, 729)
image_data_list = list(objects.getdata())
print(len(image_data_list))  # 985 * 729 = 718065
print(image_data_list[1])  # 0
print(image_data_list[5])  # 0
print(image_data_list[6])  # 9
print(image_data_list[130])  # 9
print(image_data_list[131])  # 3

