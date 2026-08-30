# src/visualization.py

from collections.abc import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_histogram(
    df: pd.DataFrame,
    x: str,
    *,
    color: str | None = None,
    nbins: int = 30,
    barmode: str = "overlay",
    opacity: float = 0.6,
    title: str | None = None,
    category_orders: dict | None = None,
) -> go.Figure:
    fig = px.histogram(
        df,
        x=x,
        color=color,
        nbins=nbins,
        barmode=barmode,
        opacity=opacity,
        title=title,
        category_orders=category_orders,
    )

    fig.update_layout(
        xaxis_title=x,
        yaxis_title="Count",
    )

    return fig


def plot_count_bar(
    df: pd.DataFrame,
    x: str,
    *,
    color: str | None = None,
    title: str | None = None,
    category_orders: dict | None = None,
) -> go.Figure:
    group_columns = [x]

    if color:
        group_columns.append(color)

    count_df = (
        df.groupby(group_columns, observed=True)
        .size()
        .reset_index(name="Count")
    )

    fig = px.bar(
        count_df,
        x=x,
        y="Count",
        color=color,
        barmode="group",
        text="Count",
        title=title,
        category_orders=category_orders,
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        xaxis_title=x,
        yaxis_title="Count",
    )

    return fig


def plot_survival_rate(
    df: pd.DataFrame,
    x: str,
    *,
    color: str | None = None,
    target: str = "Survived",
    title: str | None = None,
    category_orders: dict | None = None,
    show_count: bool = False,
) -> go.Figure:
    group_columns = [x]

    if color:
        group_columns.append(color)

    survival_df = (
        df.groupby(group_columns, observed=True)[target]
        .agg(rate="mean", count="size")
        .reset_index()
    )

    if show_count:
        survival_df["label"] = survival_df.apply(
            lambda row: f"{row['rate']:.1%}<br>n={row['count']}",
            axis=1,
        )
    else:
        survival_df["label"] = survival_df["rate"].map(
            lambda value: f"{value:.1%}"
        )

    fig = px.bar(
        survival_df,
        x=x,
        y="rate",
        color=color,
        barmode="group",
        text="label",
        title=title,
        category_orders=category_orders,
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        xaxis_title=x,
        yaxis_title="Survival Rate",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.1],
    )

    return fig


def plot_box_distribution(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: str | None = None,
    title: str | None = None,
    category_orders: dict | None = None,
    show_points: bool = True,
) -> go.Figure:
    fig = px.box(
        df,
        x=x,
        y=y,
        color=color,
        points="all" if show_points else False,
        title=title,
        category_orders=category_orders,
    )

    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        showlegend=color is not None and color != x,
    )

    return fig


def plot_survival_heatmap(
    df: pd.DataFrame,
    row: str,
    column: str,
    *,
    target: str = "Survived",
    title: str | None = None,
    row_order: Sequence | None = None,
    column_order: Sequence | None = None,
) -> go.Figure:
    survival_df = (
        df.groupby([row, column], observed=True)[target]
        .mean()
        .reset_index()
    )

    heatmap_df = survival_df.pivot(
        index=row,
        columns=column,
        values=target,
    )

    if row_order is not None:
        heatmap_df = heatmap_df.reindex(row_order)

    if column_order is not None:
        heatmap_df = heatmap_df.reindex(columns=column_order)

    fig = px.imshow(
        heatmap_df,
        text_auto=".1%",
        aspect="auto",
        zmin=0,
        zmax=1,
        title=title,
        labels={
            "x": column,
            "y": row,
            "color": "Survival Rate",
        },
    )

    return fig


def plot_count_heatmap(
    df: pd.DataFrame,
    row: str,
    column: str,
    *,
    title: str | None = None,
) -> go.Figure:
    count_df = pd.crosstab(
        index=df[row],
        columns=df[column],
    )

    fig = px.imshow(
        count_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title=title,
        labels={
            "x": column,
            "y": row,
            "color": "Passenger Count",
        },
    )

    return fig