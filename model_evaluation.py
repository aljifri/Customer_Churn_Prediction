from sklearn.metrics import classification_report, confusion_matrix
from model_training import predictions, y_test

print(confusion_matrix(predictions, y_test))
print(classification_report(predictions, y_test))