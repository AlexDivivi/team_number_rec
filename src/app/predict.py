import torch
import torch.nn as nn
import imageio
from skimage import color
from skimage.transform import rescale
import numpy as np


class Predict:
    def __init__(self, image_file):
        self.load_model = torch.load('../../model/model.pt')
        self.model = self.load_model.eval()
        self.image_scaled = self.read_image(image_file)
        self.prediction = self.predict(self.model, self.image_scaled)

    def predict(self, model, input):
        input = torch.as_tensor(input, device=torch.device("cpu"), dtype=torch.float)
        pr = model(input)
        sf = nn.Softmax(dim=1)
        vec = sf(pr)
        vec = vec.squeeze()
        print(vec)
        nx = vec.detach().numpy()
        number = np.where(nx == max(nx))[0][0]
        print(number)
        return number

    def read_image(self, image_file):
        image = imageio.imread(image_file)
        image = color.rgb2gray(image)
        image = rescale(image=image, scale=0.03125, multichannel=False, anti_aliasing=True)
        image = image.reshape(1, 256) * (-1)
        image += 0.5
        image /= np.abs(image).max()
        return image
