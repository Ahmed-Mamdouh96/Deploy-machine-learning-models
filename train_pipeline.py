from json import load
from re import I

from yaml import safe_dump
from config import config
from processing.data_management import load_dataset,save_pipeline
import pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from predict import make_prediction

def run_training():

    train = load_dataset(config.TRAIN_FILE)
   


    y=train[config.TARGET]
    train.drop(config.TARGET, axis=1,inplace=True)

    pipeline.pipeline.fit(train[config.NUMERICAL_FEATURES],y)

    #predict=pipeline.pipeline.predict(test[config.NUMERICAL_FEATURES])

    print('train accuracy: {}'.format(accuracy_score(y, pipeline.pipeline.predict(train[config.NUMERICAL_FEATURES]))))

    save_pipeline(pipeline.pipeline)

if __name__ == '__main__':
    run_training()
    test=load_dataset(config.TEST_FILE)
    y_test=test[config.TARGET]
    test.drop(config.TARGET, axis=1,inplace=True)
    

    result = make_prediction(test)
    print(result['model_name'])

    
    print('test accuracy: {}'.format(accuracy_score(y_test,result['prediction'] )))

