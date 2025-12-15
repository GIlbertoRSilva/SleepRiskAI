# src/visual.py

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)


class Visual:
    def __init__(self):
        BASE_DIR = Path(__file__).resolve().parent.parent

        self.MODEL_PATH = BASE_DIR / "models" / "sleep_risk_model.pkl"
        self.FI_PATH = BASE_DIR / "models" / "feature_importance.csv"
        self.DATA_PATH = BASE_DIR / "data" / "processed" / "data_processed.csv"

        self.OUTPUT_DIR = BASE_DIR / "reports" / "figures"
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.TARGET = "SleepRisk"

     
        self.BLUE_MAIN = "#1f4fd8"
        self.BLUE_LIGHT = "#6fa8ff"
        self.BLUE_DARK = "#0b1f66"

        self._load_assets()
        self._predict()

    def _load_assets(self):
        print("Loading model and data...")

        self.model = joblib.load(self.MODEL_PATH)
        self.df = pd.read_csv(self.DATA_PATH)

        self.X = self.df.drop(columns=[self.TARGET])
        self.y = self.df[self.TARGET]

        self.feature_importance = pd.read_csv(self.FI_PATH)

    def _predict(self):
        self.y_pred = self.model.predict(self.X)
        self.y_proba = self.model.predict_proba(self.X)[:, 1]

    def save_classification_report(self):
        report = classification_report(self.y, self.y_pred)

        report_path = self.OUTPUT_DIR / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)

    def save_confusion_matrix(self, normalize=True):
        cm = confusion_matrix(self.y, self.y_pred)

        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(6, 6))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()

        labels = ["No Risk", "High Risk"]
        plt.xticks([0, 1], labels)
        plt.yticks([0, 1], labels)

        for i in range(2):
            for j in range(2):
                plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color=self.BLUE_DARK)

        plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        plt.tight_layout()
        plt.savefig(self.OUTPUT_DIR / "confusion_matrix.png", dpi=300)
        plt.close()

    def save_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y, self.y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color=self.BLUE_MAIN, linewidth=2, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color=self.BLUE_LIGHT)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.OUTPUT_DIR / "roc_curve.png", dpi=300)
        plt.close()

    def save_precision_recall_curve(self):
        precision, recall, _ = precision_recall_curve(self.y, self.y_proba)

        plt.figure(figsize=(7, 6))
        plt.plot(recall, precision, color=self.BLUE_MAIN, linewidth=2)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")

        plt.tight_layout()
        plt.savefig(self.OUTPUT_DIR / "precision_recall_curve.png", dpi=300)
        plt.close()


    def save_feature_importance(self, top_n=15):
        df_plot = self.feature_importance.head(top_n)

        plt.figure(figsize=(8, 6))
        plt.barh(df_plot["feature"], df_plot["importance"], color=self.BLUE_MAIN)
        plt.gca().invert_yaxis()

        plt.title("Top Feature Importances")
        plt.xlabel("Importance")

        plt.tight_layout()
        plt.savefig(self.OUTPUT_DIR / "feature_importance.png", dpi=300)
        plt.close()

    def save_all(self):
        print("Saving all evaluation artifacts...")

        self.save_classification_report()
        self.save_confusion_matrix()
        self.save_roc_curve()
        self.save_precision_recall_curve()
        self.save_feature_importance()

        print(f"All figures saved in: {self.OUTPUT_DIR}")

viz = Visual()
viz.save_all()
