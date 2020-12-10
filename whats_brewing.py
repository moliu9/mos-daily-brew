# import statements
import numpy as np
import csv
from sklearn import preprocessing
from sklearn.naive_bayes import CategoricalNB

# initializing the LabelEncoders
le_temp = preprocessing.LabelEncoder()
le_weather = preprocessing.LabelEncoder()
le_energy = preprocessing.LabelEncoder()
le_mood = preprocessing.LabelEncoder()
le_shop = preprocessing.LabelEncoder()
le_drink = preprocessing.LabelEncoder()


def load_data(filename: str) -> list:
    """ parse in data by iterating through every row of the csv file
        input: filename of the csv file
        output: list of lists containing string values of the four features & two labels"""
    file = open(filename, encoding="utf8")
    reader = csv.reader(file)
    temp, weather, energy, mood, coffee_shop, coffee_drink = ([] for _ in range(6))
    for row in reader:
        temp.append(row[0])
        weather.append(row[1])
        energy.append(row[2])
        mood.append(row[3])
        coffee_shop.append(row[4])
        coffee_drink.append(row[5])
    return [temp, weather, energy, mood, coffee_shop, coffee_drink]


def encode_features():
    """ encode the features' string values into numeric representations
        input: None
        output: a 2D numpy array with size (number_of_rows, 4) of encoded features """

    # unpack the data by splitting different features into separate lists
    data = load_data('cool_beans.csv')
    temp = data[0]
    weather = data[1]
    energy = data[2]
    mood = data[3]

    # converting strings into numeric representation using respective LabelEncoder
    temp_encoded = le_temp.fit_transform(temp)
    weather_encoded = le_weather.fit_transform(weather)
    energy_encoded = le_energy.fit_transform(energy)
    mood_encoded = le_mood.fit_transform(mood)

    # combine features into array form
    return np.column_stack((temp_encoded, weather_encoded, energy_encoded, mood_encoded))


def encode_labels():
    """ encode the labels' string features into numeric representations
        input: None
        output: lists of the shop and drink labels, encoded """

    # unpack the data by splitting different labels into separate lists
    data = load_data('cool_beans.csv')
    coffee_shop = data[4]
    coffee_drink = data[5]

    # convert string labels into numeric representation
    shop_label_encoded = le_shop.fit_transform(coffee_shop)
    drink_label_encoded = le_drink.fit_transform(coffee_drink)
    return [shop_label_encoded, drink_label_encoded]


def train_shop_model():
    """ create and train the NB model for coffee shop selection
        input: None (acquire features and label from functions above)
        output: Categorical NB classifier fitted with coffee shop labels """

    # obtain encoded features and labels
    features = encode_features()
    shop_label = encode_labels()[0]

    # create and fit the Categorical NB model
    cnb_shop = CategoricalNB()
    cnb_shop.fit(features, shop_label)
    return cnb_shop


def predict_shop_model(temp_input: str, weather_input: str, energy_input: str, mood_input: str) -> str:
    """ predict coffee shop selection using trained NB model
        input: user inputs of features
        output: name of the coffee shop predicted """

    cnb_shop = train_shop_model()

    # transform string inputs into numeric representations
    temp = le_temp.transform([temp_input])  # why is the label unseen?
    weather = le_weather.transform([weather_input])
    energy = le_energy.transform([energy_input])
    mood = le_mood.transform([mood_input])

    shop_pred = le_shop.inverse_transform(cnb_shop.predict([[temp[0], weather[0], energy[0], mood[0]]]))
    return shop_pred[0]


def train_drink_model():
    """ create and train the NB model for coffee drink selection
        input: None (acquire features and label from functions above)
        output: Categorical NB classifier fitted with coffee shop labels """

    # obtain features and labels
    features = encode_features()
    drink_label = encode_labels()[1]

    # create and fit the Categorical NB model
    cnb_drink = CategoricalNB()
    cnb_drink.fit(features, drink_label)
    return cnb_drink


def predict_drink_model(temp_input: str, weather_input: str, energy_input: str, mood_input: str) -> str:
    """ predict coffee drink selection using trained NB model
        input: user inputs of features
        output: name of the coffee drink predicted """

    cnb_drink = train_drink_model()

    # transform string inputs into numeric representations
    temp = le_temp.transform([temp_input])  # why is the label unseen?
    weather = le_weather.transform([weather_input])
    energy = le_energy.transform([energy_input])
    mood = le_mood.transform([mood_input])

    drink_pred = le_drink.inverse_transform(cnb_drink.predict([[temp[0], weather[0], energy[0], mood[0]]]))
    return drink_pred[0]


def execute_classifier():
    """ allows user to call the classifier to predict shop and drink from user input data """

    # prompt the user for string inputs
    temp_input = input('What is the temperature today?')
    weather_input = input('What is the weather like?')
    energy_input = input('What is your energy level?')
    mood_input = input('How are you feeling?')

    # use the NB models to predict shop and drink
    predicted_shop = predict_shop_model(temp_input, weather_input, energy_input, mood_input)
    predicted_drink = predict_drink_model(temp_input, weather_input, energy_input, mood_input)
    print('Good Morning Mo! I think you should get a {0} from {1} today'.format(predicted_drink, predicted_shop))
