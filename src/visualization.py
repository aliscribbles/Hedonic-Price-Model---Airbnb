"""
Visualization — EDA plots and model comparison charts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================================
# Style Defaults
# ==========================================================================
def set_style():
    """Apply a clean, publication-quality matplotlib style."""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    plt.rcParams.update({
        "figure.figsize": (12, 8),
        "figure.dpi": 120,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


# ==========================================================================
# EDA Plots
# ==========================================================================
def plot_price_distribution(
    df: pd.DataFrame,
    price_col: str = "price",
    log_price_col: str = "log_price",
    save_path: str | None = None,
):
    """Side-by-side histograms of raw and log-transformed price."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if price_col in df.columns:
        axes[0].hist(df[price_col], bins=100, color="#4C72B0", edgecolor="white")
        axes[0].set_title("Price Distribution")
        axes[0].set_xlabel("Price ($)")
        axes[0].set_ylabel("Frequency")

    if log_price_col in df.columns:
        axes[1].hist(df[log_price_col], bins=80, color="#55A868", edgecolor="white")
        axes[1].set_title("Log Price Distribution")
        axes[1].set_xlabel("Log(Price)")
        axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_neighbourhood_analysis(
    df: pd.DataFrame,
    neighbourhood_col: str = "neighbourhood_cleansed",
    price_col: str = "price",
    top_n: int = 20,
    save_path: str | None = None,
):
    """Bar charts of listing count and median price by neighbourhood."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Listing counts
    counts = df[neighbourhood_col].value_counts().head(top_n)
    counts.plot.barh(ax=axes[0], color="#4C72B0")
    axes[0].set_title(f"Top {top_n} Neighbourhoods by Listing Count")
    axes[0].set_xlabel("Number of Listings")
    axes[0].invert_yaxis()

    # Median price
    medians = df.groupby(neighbourhood_col)[price_col].median().sort_values(ascending=False).head(top_n)
    medians.plot.barh(ax=axes[1], color="#C44E52")
    axes[1].set_title(f"Top {top_n} Neighbourhoods by Median Price")
    axes[1].set_xlabel("Median Price ($)")
    axes[1].invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    figsize: tuple = (16, 14),
    save_path: str | None = None,
):
    """Heatmap of the correlation matrix for selected numeric columns."""
    set_style()
    if columns:
        data = df[columns]
    else:
        data = df.select_dtypes(include=[np.number])

    corr = data.corr()
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=False, cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax,
    )
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_boxplot_by_neighbourhood(
    df: pd.DataFrame,
    neighbourhood_col: str = "neighbourhood_cleansed",
    price_col: str = "price",
    log: bool = True,
    save_path: str | None = None,
):
    """Boxplot of price (or log price) grouped by neighbourhood."""
    set_style()
    fig, ax = plt.subplots(figsize=(15, 25))
    col = f"log_{price_col}" if log and f"log_{price_col}" in df.columns else price_col
    sns.boxplot(y=neighbourhood_col, x=col, data=df, ax=ax, orient="h")
    ax.set_title(f"{'Log ' if log else ''}Price by Neighbourhood")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ==========================================================================
# Model Evaluation Plots
# ==========================================================================
def plot_predictions_vs_actual(
    y_true_train,
    y_pred_train,
    y_true_test,
    y_pred_test,
    title: str = "Predictions vs Actual",
    save_path: str | None = None,
):
    """Scatter plots of predicted vs actual for train and test sets."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    axes[0].scatter(y_true_train, y_pred_train, alpha=0.3, s=10, color="#4C72B0")
    axes[0].plot(
        [y_true_train.min(), y_true_train.max()],
        [y_true_train.min(), y_true_train.max()],
        "r--", lw=2,
    )
    axes[0].set_title(f"{title} — Train Set")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")

    axes[1].scatter(y_true_test, y_pred_test, alpha=0.3, s=10, color="#55A868")
    axes[1].plot(
        [y_true_test.min(), y_true_test.max()],
        [y_true_test.min(), y_true_test.max()],
        "r--", lw=2,
    )
    axes[1].set_title(f"{title} — Test Set")
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_feature_importance(
    importances: pd.DataFrame,
    title: str = "Feature Importance",
    top_n: int = 20,
    save_path: str | None = None,
):
    """Horizontal bar chart of top-N feature importances."""
    set_style()
    data = importances.head(top_n).iloc[::-1]  # reverse for horizontal bars

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    ax.barh(data["feature"], data["importance"], color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_regularization_trace(
    trace_df: pd.DataFrame,
    features: list[str] | None = None,
    title: str = "Regularization Trace",
    save_path: str | None = None,
):
    """Line plot of coefficient magnitude vs regularization strength."""
    set_style()
    fig, ax = plt.subplots(figsize=(12, 8))

    cols = features or [c for c in trace_df.columns if c not in ("alpha", "r2")]
    for col in cols[:15]:  # limit to 15 features for readability
        ax.plot(trace_df["alpha"], trace_df[col], label=col, linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Alpha (λ)")
    ax.set_ylabel("Coefficient Value")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "Test R²",
    save_path: str | None = None,
):
    """Bar chart comparing models on a single metric."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
    comparison_df[metric].plot.bar(ax=ax, color=colors[: len(comparison_df)])
    ax.set_title(f"Model Comparison — {metric}")
    ax.set_ylabel(metric)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    # Add value labels on bars
    for i, v in enumerate(comparison_df[metric]):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
