from PIL import Image
from transformer import transform
from commandline_helper import get_args

iterations, base_img,output_path, style_imgs = get_args()

content_image = Image.open(base_img)
height, width = content_image.size


for style_img_src, nr in zip(style_imgs, range(len(style_imgs))):
    style_image = Image.open(style_img_src).resize((height, width))
    content_image = transform(content_image, style_image, height, width, iterations)
    content_image.save(output_path+"output_"+str(nr)+".jpg")

