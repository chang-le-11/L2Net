from PIL import Image
import os


size = 128
stride = 32


def image_cut(image_name, save_path, size, stride, save_name):
    # image_name:图像路径
    # save_path:保存地址
    # size:裁剪的图像大小
    # stride:步长
    # save_name:图像保存编号
    img = Image.open(image_name)
    width = img.size[0]
    height = img.size[1]
    w1 = (width - size[0]) // stride + 1
    w2 = (height -size[1]) // stride + 1
    if width > size[0] and height > size[1]:
        for i in range(w1):
            for j in range(w2):
                cut_name = os.path.join(save_path, str(save_name)+'.png')
                save_name += 1
                cimg = img.crop(box=(i*stride, j*stride, i*stride+size[0], j*stride+size[1]))
                cimg.save(cut_name)
    else:
        print("size is error!")
    print(save_name)
    return save_name


print("_______IR________")
path_name = "E:\\my_methods\\MSRS\\ir"
save_path = "E:\\my_methods\\train_data\\ir"

save_name = 1
for item in os.listdir(path_name):
    image_name = os.path.join(path_name, item)
    save_name = image_cut(image_name, save_path, [size, size], stride, save_name)


print("_______VIS_______")
path_name = "E:\\my_methods\\MSRS\\vi"
save_path = "E:\\my_methods\\train_data\\vis"

save_name = 1
for item in os.listdir(path_name):
    image_name = os.path.join(path_name, item)
    save_name = image_cut(image_name, save_path, [size, size], stride, save_name)



