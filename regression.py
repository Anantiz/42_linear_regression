import sys
import math
import matplotlib.pyplot as plt

EPSILON = 1e-18

def mean(data):
    return sum(data) / len(data)

def variance(data):
    mu = mean(data)
    return sum([(x - mu)**2 for x in data]) / len(data)

def standard_deviation(data):
    mu = mean(data)
    return math.sqrt(sum([(x - mu)**2 for x in data]) / len(data))

class LinearRegression():
    """
    Reminders:
        R^2    = (Var(mean) - Var(line)) / Var(mean)
        Var(a) = sum((a_i - mean)^2)
    """
    def __init__(self):
        """ Nothing to do here """
        # First order polynomials only (Very unexciting but that's what the assignment asks for)
        self.k = 0
        self.x_coefficient = 0
        self.trained = False
        self.lr = 0.01 # Learning rate
        self.x_data = None
        self.y_data = None
        self.labels = ('X', 'Y')

    def save_model(self, file_path='model.csv'):
        """
        Save the model to a file
        """
        if not self.trained:
            print("Error: Model has not been trained yet, cannot save", file=sys.stderr)
            return False
        try:
            with open(file_path, 'w') as f:
                f.write(f"{self.k},{self.x_coefficient}")
                print(f"Model saved to `{file_path}`")
        except:
            print(f"Error: Could not write to file `{file_path}`", file=sys.stderr)
            return False
        return True

    def load_model(self, file_path='model.csv'):
        """
        Load the model from a file
        """
        try:
            with open(file_path, 'r') as f:
                raw_data = f.read().split(',')
                if len(raw_data) > 2:
                    raise RuntimeError("Too many values in file")
                self.k = float(raw_data[0])
                self.x_coefficient = float(raw_data[1])
                if self.k is None or self.x_coefficient is None:
                    raise ValueError
                self.trained = True
        except FileNotFoundError:
            print(f"Error: File `{file_path}` not found", file=sys.stderr)
            return False
        except ValueError:
            print("Error: File contents could not be parsed as float", file=sys.stderr)
            return False
        except RuntimeError as e:
            print(e, file=sys.stderr)
            return False
        return True

    def load_data(self, file_path='data.csv') -> tuple[list, list] | None:
            """
            File must be a csv file with two columns
            The first column is the x values and the second column is the y values
            """
            try:
                with open(file_path, 'r') as f:
                    # Assume the first line is the header
                    # Assume corect format for now, error handling will be added later
                    raw_data = f.readlines()
                    headers = raw_data[0].split(',')
                    self.labels = (headers[0], headers[1])
                    self.x_data = [int(i.split(',')[0]) for i in raw_data[1:]] # Skip the header
                    self.y_data = [int(i.split(',')[1]) for i in raw_data[1:]] # Skip the header
            except FileNotFoundError:
                print(f"Error: File `{file_path}` not found", file=sys.stderr)
                return False
            except ValueError:
                print("Error: File contents could not be parsed as whole-integer", file=sys.stderr)
                return False
            return True

    def clear_data(self):
        """ To reclaim memory"""
        self.x_data = None
        self.y_data = None

    def compare_data(self):
        R = self.Rsquared()
        print(f"R^2: {R}\nThis is the precisions/acuracy of the model, 1 is perfect, 0 is as good as the mean, non-normalized value means the model sucks")
        predicted = [self.predict(x) for x in self.x_data]
        plt.plot(self.x_data, self.y_data, 'ro', label='Data')
        plt.plot(self.x_data, predicted, 'b-', label='Model-prediction')
        plt.title("Data vs Model-prediction")
        plt.xlabel(self.labels[0])
        plt.ylabel(self.labels[1])
        plt.show()

    def train(self, training_cycles_count=10_000, reset=True, normalize=True):
        """
        Train the model with the data in the file
        """
        def fit_polynomial(training_cycles_count, x_data, y_data) -> bool:
            """ Fit polynomial for the given data"""
            m = len(x_data)
            Nr = 1 / m
            update_ratio = self.lr * Nr
            print(f"Starting training with {training_cycles_count} cycles on {m} data points")
            print(f"X data: {x_data}")
            for i in range(training_cycles_count):
                grad_k = sum([(self.predict(x) - y) for x, y in zip(x_data, y_data)])
                grad_x_coef = sum([(self.predict(x) - y) * x for x, y in zip(x_data, y_data)])
                self.k -= update_ratio * grad_k
                self.x_coefficient -= update_ratio * grad_x_coef

        if reset:
            self.k = 0
            self.x_coefficient = 0
            self.trained = False
        if self.x_data is None:
            print("Error: No data loaded, cannot train", file=sys.stderr)
            return False
        if normalize == False:
            fit_polynomial(training_cycles_count, self.x_data, self.y_data)
        else:
            mean_x = mean(self.x_data)
            dev_x = standard_deviation(self.x_data)
            norm_x = [(x-mean_x) / dev_x for x in self.x_data]
            fit_polynomial(training_cycles_count, norm_x, self.y_data)
        self.trained = True
        print(f"Model trained: k={self.k}, x_coefficient={self.x_coefficient}, R^2={self.Rsquared()}")
        return True

    def Rsquared(self):
        """
        Literally just the formula from wikipedia:
            https://en.wikipedia.org/wiki/Coefficient_of_determination#Definitions
        In non-math terms, what we are doing is measuring how much better/worse
        our predictions are versus just taking the mean(average) value of the data set.

        If R == 1 we have a perfect model
        If R == 0 we have a model that is no better than just taking the mean
        """
        if self.x_data is None:
            print("Error, no data loaded, cannot estimate Rsquared", file=sys.stderr)
            return -1
        y_mean = sum(self.y_data) / len(self.y_data)
        SSres = sum([(y - self.predict(x))**2 for x,y in zip(self.x_data, self.y_data)])
        SStot = sum([(y - y_mean)**2 for y in self.y_data])
        if SStot < EPSILON:
            return 1
        R = 1 - (SSres / SStot)
        return R

    def predict(self, x):
        return self.k + self.x_coefficient * x
