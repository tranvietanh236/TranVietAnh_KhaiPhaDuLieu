from src.loaddata import data_loader, split_data
from src.prepprocess_data import data_processor
from src.train_model import model_train
from src.evalute_model import model_evalutor, plotter


if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    data = data_loader.load_data(url)
    data =  data_processor.preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data.split_data(data)
    model = model_train.train_model(X_train, y_train)
    model_evalutor.evaluate_model(model, X_test, y_test)
