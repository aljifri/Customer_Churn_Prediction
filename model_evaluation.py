from sklearn.metrics import classification_report, confusion_matrix
from model_training import predictions, y_test

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))
