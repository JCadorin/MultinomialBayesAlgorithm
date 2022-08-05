from traintestdata import *
from classes import *


mnnb = MNNaiveBayes()
mnnb.fit(X,y)
predicted = mnnb.predict(X_test)
print()
print("Predicting Text:")
for x in predicted:
    print()
    print(x)