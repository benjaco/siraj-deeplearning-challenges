import numpy as np
import time

from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16
from PIL import Image

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave


def image_to_array(content_image):
    content_array = np.asarray(content_image, dtype='float32')
    content_array = np.expand_dims(content_array, axis=0)
    return content_array


def transform_color_and_flip_colorchannels(content_array):
    content_array[:, :, :, 0] -= 103.939
    content_array[:, :, :, 1] -= 116.779
    content_array[:, :, :, 2] -= 123.68
    return content_array[:, :, :, ::-1]


def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram


def style_loss(style, combination, height, width):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height * width
    return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x, height, width):
    a = backend.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))


def eval_loss_and_grads(x, f_outputs, height, width):
    x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values

def transform(content_img, style_img, height, width, iterations):
    content_array = image_to_array(content_img)
    style_array = image_to_array(style_img)

    content_array = transform_color_and_flip_colorchannels(content_array)
    style_array = transform_color_and_flip_colorchannels(style_array)

    content_image = backend.variable(content_array)
    style_image = backend.variable(style_array)
    combination_image = backend.placeholder((1, height, width, 3))

    input_tensor = backend.concatenate([content_image,
                                        style_image,
                                        combination_image], axis=0)

    model = VGG16(input_tensor=input_tensor, weights='imagenet',
                  include_top=False)

    layers = dict([(layer.name, layer.output) for layer in model.layers])

    content_weight = 0.025
    style_weight = 5.0
    total_variation_weight = 1.0

    loss = backend.variable(0.)

    layer_features = layers['block2_conv2']
    content_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]

    loss += content_weight * backend.sum(backend.square(combination_features - content_image_features))


    feature_layers = ['block1_conv2', 'block2_conv2',
                      'block3_conv3', 'block4_conv3',
                      'block5_conv3']

    for layer_name in feature_layers:
        layer_features = layers[layer_name]
        style_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_features, combination_features, height, width)
        loss += (style_weight / len(feature_layers)) * sl

    loss += total_variation_weight * total_variation_loss(combination_image, height, width)


    grads = backend.gradients(loss, combination_image)

    outputs = [loss]
    outputs += grads
    f_outputs = backend.function([combination_image], outputs)

    evaluator = Evaluator(f_outputs, height, width)

    x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))

        x = x.reshape((height, width, 3))
        x = x[:, :, ::-1]
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = np.clip(x, 0, 255).astype('uint8')

    img = Image.fromarray(x)
    return img




class Evaluator(object):

    def __init__(self, f_outputs, height, width):
        self.loss_value = None
        self.grads_values = None
        self.f_outputs = f_outputs
        self.height = height
        self.width = width

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x, self.f_outputs, self.height, self.width)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values