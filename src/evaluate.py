
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


def evaluate_model(model, X_train , y_train,X_test, y_test, model_name="Model"):
    """
    Evaluate a classification model using multiple metrics.

    Returns:
        Dictionary containing evaluation metrics and predictions.
    """

    # Generate predictions
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)

    # Calculate classification metrics
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred, zero_division=0)
    recall_train = recall_score(y_train, y_train_pred, zero_division=0)
    f1_train = f1_score(y_train, y_train_pred, zero_division=0)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    cm_train = confusion_matrix(y_train, y_train_pred)
    # Display evaluation results
    print(f"Evaluation Results: {model_name}")
    print("-" * 50)

    print(f"Test Accuracy:  {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test F1 Score:  {f1:.4f}")

    print(f"Train Accuracy:  {accuracy_train:.4f}")
    print(f"Train Precision: {precision_train:.4f}")
    print(f"Train Recall:    {recall_train:.4f}")
    print(f"Train F1 Score:  {f1_train:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_test_pred,
        target_names=["Canceled", "Not_Canceled"],
        zero_division=0
    ))

    print("\nTrain Classification Report:")
    print(classification_report(
        y_train,
        y_train_pred,
        target_names=["Canceled", "Not_Canceled"],
        zero_division=0
    ))
    

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Canceled", "Not_Canceled"],
        yticklabels=["Canceled", "Not_Canceled"]
    )

    plt.title(f"{model_name} -Test Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm_train,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Canceled", "Not_Canceled"],
        yticklabels=["Canceled", "Not_Canceled"]
    )

    plt.title(f"{model_name} -Train Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.show()

    # Calculate ROC-AUC if probability estimates are available
    roc_auc = None

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_proba)

        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

        plt.figure(figsize=(6, 4))

        plt.plot(
            fpr,
            tpr,
            label=f"ROC-AUC = {roc_auc:.4f}"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.title(f"{model_name} - ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"ROC-AUC:  {roc_auc:.4f}")

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "training_accuracy": accuracy_train,
        "training_precision": precision_train,
        "training_recall": recall_train,
        "training_f1_score": f1_train,
        "training_confusion_matrix": cm_train,  
        "predictions": y_test_pred,
        "predictions_train": y_train_pred
    }
