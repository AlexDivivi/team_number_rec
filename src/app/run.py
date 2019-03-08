from app import Preprocessing
from app import Train

numbers = Preprocessing('numbers')
numbers.load_data('zip.csv', name='numbers', sep=' ', header=None)
numbers.cleanup('numbers', drop=257)
numbers.split_data(target=0)
train = Train(numbers, epoch=20000)
