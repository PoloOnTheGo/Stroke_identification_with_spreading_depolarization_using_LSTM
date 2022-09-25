from tensorflow.keras.models import model_from_json


def load_trained_model():
    json_file = open("model-bw.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights("model-bw.h5")
    print("Loaded model from disk")
    return model


def run_experiment(model):
    # load data
    validation_data_folder = ""
    validation_X, validation_y = load_data(validation_data_folder)

    score = evaluate_model(validation_X, validation_y, model)
    score = score * 100.0
    print('>#%d: %.3f' % (r + 1, score))


if __name__ == "__main__":
    trained_model = load_trained_model()
    run_experiment(trained_model)
