from app import Preprocessing
from app import Train
from app import Predict
import os

# numbers = Preprocessing('numbers')
# numbers.load_data('zip.csv', name='numbers', sep=' ', header=None)
# numbers.cleanup('numbers', drop=257)
# numbers.split_data(target=0)
# train = Train(numbers, epoch=150000)

for i in range(0, 2):
    print(f'{i} : ')
    prd = Predict(f'../../images/written{i}.png')

for i in range(3, 6):
    print(f'{i} : ')
    prd = Predict(f'../../images/written{i}.png')

for i in range(7, 10):
    print(f'{i} : ')
    prd = Predict(f'../../images/written{i}.png')

image = 1
prd = Predict(f'../../images/written{image}.png')
os.system(f'say {prd.prediction}')
