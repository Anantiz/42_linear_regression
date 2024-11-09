from regression import LinearRegression
import sys

def train_save_model():
    model = LinearRegression()
    # Check if the user gave us a file to read from
    ac = len(sys.argv)
    save_file = 'ft_linear_model.csv'
    if ac >= 3:
        data_file = sys.argv[2]
        model.load_data(data_file)
    else:
        model.load_data()
    ret = model.train()

    if not ret:
        print("Error: Model could not be trained", file=sys.stderr)
        sys.exit(1)
    if ac >= 4:
        save_file = sys.argv[3]
        model.save_model(save_file)
    else:
        model.save_model()


def load_evaluate_model():
    model = LinearRegression()
    ac = len(sys.argv)
    if ac >= 3:
        model.load_model(sys.argv[2])
    else:
        model.load_model()
    if not model.trained:
        print("Error: Model could not be loaded", file=sys.stderr)
        sys.exit(1)
    if ac >= 4:
        test_data = sys.argv[3]
        model.load_data(test_data)
    else:
        model.load_data()
    if model.x_data is None or model.y_data is None:
        print("Error: Data could not be loaded", file=sys.stderr)
        sys.exit(1)
    model.compare_data()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:\n\
            python3 script.py train data_file[='data_csv'] save_file[='ft_linear_model.csv']\n\
            python3 script.py test model_weights[='ft_linear_model.csv'] data_file_to_compare[='data_csv']\n", file=sys.stderr)
        sys.exit(1)
    try:
        mode = sys.argv[1]
        if mode == 'train':
            train_save_model()
        elif mode == 'test':
            load_evaluate_model()
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)
