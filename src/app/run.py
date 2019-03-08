from app import Preprocessing
from app import Train

forest = Preprocessing('forest')
forest.load_data('train_cover_type.csv', name='forest')
forest.clean_dataset('forest')
forest.split_data(target='Cover_Type')
feature_len = len(forest.splits['X_train'][0])
train = Train(forest)
