from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.evalute_model import plotter


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    classification_rep = classification_report(y_test, y_pred, target_names=['<=50K', '>50K'], output_dict=True)
    print("Classification Report:\n", classification_rep)

    cm = confusion_matrix(y_test, y_pred)
    plotter.plot_confusion_matrix(cm)

    plotter.plot_feature_importance(model, X_test.columns)


