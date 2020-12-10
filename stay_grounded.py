from whats_brewing import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
The purpose of this file is to test and compare the accuracies of three NB models:
Multinomial, Complement, and Categorical
"""


# get the features and labels
features = encode_features()
shop_label, drink_label = encode_labels()

# test accuracy of MultinomialNB
f_train, f_test, sl_train, sl_test = train_test_split(features, shop_label, test_size=0.2, random_state=0)
mnb = MultinomialNB()
mnb.fit(f_train, sl_train)
sl_expect = sl_test
sl_pred = mnb.predict(f_test)
shop_score = accuracy_score(sl_expect, sl_pred, normalize=True)
print('The coffee shop prediction using the MultinomialNB model is {} % accurate'.format(shop_score*100))


# test accuracy of ComplementNB
f_train, f_test, sl_train, sl_test = train_test_split(features, shop_label, test_size=0.2, random_state=0)
comp_nb = ComplementNB()
comp_nb.fit(f_train, sl_train)
sl_expect = sl_test
sl_pred = comp_nb.predict(f_test)
shop_score = accuracy_score(sl_expect, sl_pred, normalize=True)
print('The coffee shop prediction using the ComplementNB model is {} & accurate'.format(shop_score*100))


# test accuracy of CategoricalNB
f_train, f_test, sl_train, sl_test = train_test_split(features, shop_label, test_size=0.2, random_state=0)
cnb = CategoricalNB() # change temperature data into categorical
cnb.fit(f_train, sl_train)
sl_expect = sl_test
sl_pred = cnb.predict(f_test)
shop_score = accuracy_score(sl_expect, sl_pred, normalize=True)
print('The coffee shop prediction using the CategoricalNB model is {} % accurate'.format(shop_score*100))