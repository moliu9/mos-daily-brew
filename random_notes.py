def classifier(temperature_input: str, weather_input: str, energy_input: str, mood_input: str):

    # obtain encoded features and labels
    features = encode_features()
    coffee_shop_label, coffee_drink_label,  = encode_labels()

    # predict the coffee shop
    mnb1 = MultinomialNB()  # justification of this distribution?

    mnb1.fit(features, coffee_shop_label)
    shop_prediction = mnb1.predict([[temperature_input, weather_input, energy_input, mood_input]])
    shop = le1.inverse_transform(shop_prediction)

    # predict the coffee drink
    mnb2 = MultinomialNB()

    mnb2.fit(features, coffee_drink_label)
    drink_prediction = mnb2.predict([[temperature_input, weather_input, energy_input, mood_input]])
    drink = le2.inverse_transform(drink_prediction)

    # return the predicted decoded string values
    return shop[0], drink[0]
