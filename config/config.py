DATA_PATH=r"C:\Users\Ahmed\Desktop\deployNoteBook"
TRAIN_FILE=r'\datasets\train.csv'
TEST_FILE=r'\datasets\test.csv'

TARGET ='survived'

NUMERICAL_FEATURES= ['pclass', 'age', 'sibsp', 'parch', 'fare']

CATEGORICAL_FEATURES= ['sex', 'cabin', 'embarked', 'title']

SAVED_MODEL_PATH=r"C:\Users\Ahmed\Desktop\deployNoteBook\trained_models"


NUMERICAL_FEATURES_WITH_NA = ['age', 'fare']

CATEGORICAL_FEATURES_WITH_NA = ['cabin', 'embarked']