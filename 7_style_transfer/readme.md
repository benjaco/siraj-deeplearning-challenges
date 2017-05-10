# Generate Art Challenge
A response to Siraj's Generate Art Challenge found here: https://www.youtube.com/watch?v=Oex0eWoU7AQ

The video shows how to style a image using another image, the challenge was to to style a image using to other images

I have created a program there can take multiple style image and put a easy to use command line interface on top of it

The demo code from Siraj simplified formula is: `base_image + style_image = new_image`

But I got a gray image if I runned: `base_image + style_image + style_image_2`

So I ended up with splitting it up like: `base_image + style_image = output` then `output + style_image_n = output` for the other images


`Use: style.py no_of_iterations base_image output_path style_images...`

## demo

`python style.py 5 images/hugo.jpg /output/ images/styles/gothic.jpg images/styles/picasso.jpg images/styles/marilyn.jpg`

![Image Flow](https://raw.githubusercontent.com/benjaco/siraj-deeplearning-challenges/master/7_style_transfer/results/demo.png)

