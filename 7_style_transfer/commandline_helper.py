import sys

def get_args():

    args = sys.argv

    if len(args) < 5:
        print("To few arguments")
        print("Use: style.py no_of_iterations base_image output_path style_images...")
        sys.exit()

    iterations = int(args[1])
    base_img = args[2]
    output_path = args[3]
    style_imgs = args[4:]

    return iterations, base_img,output_path, style_imgs

