from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# True labels
y_true = [0, 0, 1, 1, 1]

# Predicted probabilities of positive class
y_score = [0.1, 0.3, 0.2, 0.7, 0.9]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_score)

print(type(fpr))
print(tpr)
print(thresholds)

# Plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()