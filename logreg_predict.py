from logreg_train import *
import sys

def generate_csv(y_pred):
    with open("houses.csv", "w+") as file:
        file.write(f"Index,Hogwarts House\n")
        for ind_pred_y in range(len(y_pred)):
            file.write(f"{ind_pred_y},{y_pred[ind_pred_y]}\n")

def predict_1(dataset_train, dataset_test):
    log_model, list_w, dict_house = logreg_train(dataset_train)
    data, _, X = prep_data(dataset_test)
    y_pred = log_model.predict(data, X, list_w)
    y_pred = convert_int_to_classname(dict_house, y_pred)
    generate_csv(y_pred)

if __name__ == '__main__':
    if (len(sys.argv) != 2 or sys.argv[1].split('.\\')[-1] != "dataset_test.csv"):
        exit(1)
    predict_1( "dataset_train.csv", sys.argv[1])

