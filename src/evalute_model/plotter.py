import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def plot_feature_importance(model, feature_names):
    feature_importance = model.feature_importances_
    df_importance = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    df_importance = df_importance.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=df_importance, palette="viridis")
    plt.title("Feature Importance")
    plt.show()
