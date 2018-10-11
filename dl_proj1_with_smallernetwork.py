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
#fix random seed for reproducability
seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("D:/Deep_Learning_Project_One/Deep_Learning_Project_One/sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)#NumPy arrays are stored as contiguous blocks of memory. They usually have a single datatype(e.g. integers, floats or fixed-length strings) and then the bits in memory are interpreted as values with that datatype.
                                #Creating an array with dtype=object is different. The memory taken by the array now is filled
                                #with pointers to Python objects which are being stored elsewhere in memory 
                                #(much like a Python list is really just a list of pointers to objects, not the objects themselves).


Y = dataset[:,60] # 'Y'(dataset) is in the form of object
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y) #It will model required encoding(1=R and 0=M)on entire dataset
encoded_Y = encoder.transform(Y) #Now Labels are converted in binary form(dtype=int64)
# smaller model
def create_smaller():
	# create model, write code below
    from keras import models
    from keras import layers
    model = models.Sequential()
    model.add(layers.Dense(30, activation='relu', input_shape=(60,))) #We Have single fully connected hidden layer
                                                                      #with half no.of neurons as input variables.
                                                                      
                                    
    model.add(layers.Dense(1, activation='sigmoid')) #In binary classification,we use single neuron 
                                                     #and 'sigmoid' activation function in output layer.
	
	# Compile model, write code below
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy']) # We use optimizar='adam' for gradient descent process
                                        # 'binary_crossentropy' for binary classification problem.
    return model
# evaluate smaller model with standardized dataset
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)#The pipeline is a wrapper that executes one or more models within a pass of
                               #the cross-validation procedure.
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized(Smaller) = : %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
