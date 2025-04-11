from dataclasses import dataclass
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from dsba.preprocessing import preprocess_dataframe, split_features_and_target


@dataclass
class ClassifierEvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: list[list[int]]


def evaluate_classifier(
    classifier: ClassifierMixin, target_column: str, df: pd.DataFrame
) -> ClassifierEvaluationResult:
    df = preprocess_dataframe(df)
    X, y_actual = split_features_and_target(df, target_column)
    y_predicted = classifier.predict(X)

    accuracy = accuracy_score(y_actual, y_predicted)
    precision = precision_score(
        y_actual, y_predicted, average="weighted", zero_division=0
    )
    recall = recall_score(y_actual, y_predicted, average="weighted", zero_division=0)
    f1 = f1_score(y_actual, y_predicted, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(y_actual, y_predicted)

    return ClassifierEvaluationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        confusion_matrix=conf_matrix,
    )


# Display functions :


def visualize_classification_evaluation(result: ClassifierEvaluationResult):
    confusion_maxtrix_fig = plot_confusion_matrix(result)
    plt.show(confusion_maxtrix_fig)
    evaluation_metrics_fig = plot_classification_metrics(result)
    plt.show(evaluation_metrics_fig)


def plot_confusion_matrix(result: ClassifierEvaluationResult) -> plt.Figure:
    confusion_matrix_df = pd.DataFrame(result.confusion_matrix)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(confusion_matrix_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig


def plot_classification_metrics(result: ClassifierEvaluationResult) -> plt.Figure:
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    scores = [result.accuracy, result.precision, result.recall, result.f1_score]
    metric_data = pd.DataFrame({"Metric": metrics, "Score": scores})
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.barplot(data=metric_data, x="Metric", y="Score", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Classification Metrics")
    ax.set_ylabel("Score")
    return fig