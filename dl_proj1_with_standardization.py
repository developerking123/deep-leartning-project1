import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("D:/Deep_Learning_Project_One/Deep_Learning_Project_One/sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# baseline model
def create_baseline():
	# create model, write code below
    from keras import models
    from keras import layers
    model = models.Sequential()
    model.add(layers.Dense(60, activation='relu', input_shape=(60,)))
    model.add(layers.Dense(1, activation='sigmoid'))
	
	# Compile model, write code below
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
# evaluate baseline model with standardized dataset
#An effective data preparation scheme for tabular
#data when building neural network models is standardization.
#We use scikit learn for standardization.
#Rather than performing the standardization on the entire dataset,
#it is good practice to train the standardization procedure on the training data
#within the pass of a cross-validation run and to use the trained standardization
#to prepare the “unseen” test fold. This makes standardization a step in model
#preparation in the cross-validation process and it prevents the algorithm having
#knowledge of “unseen” data during evaluation, knowledge that might be passed from
#the data preparation scheme like a crisper distribution.
#We can achieve this in scikit-learn using a Pipeline:

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)#The pipeline is a wrapper that executes one or more models within a pass of
                               #the cross-validation procedure.
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
