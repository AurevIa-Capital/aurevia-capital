import pandas as pd


def format_currency(value):
    """Format a number as currency."""
    if pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def format_percent(value):
    """Format a number as percentage."""
    if pd.isna(value):
        return "N/A"
    return f"{value:+.2f}%" if value != 0 else "0.00%"


def get_trend_arrow(value):
    """Get an arrow based on trend direction."""
    if pd.isna(value):
        return ""
    return "↑" if value > 0 else "↓" if value < 0 else ""


def get_trend_color(value):
    """Get CSS class based on trend direction."""
    if pd.isna(value):
        return ""
    return "profit" if value > 0 else "loss" if value < 0 else ""


def format_watch_name(watch_name):
    """Format watch name for display."""
    display_name = watch_name.split("-")[1:] if "-" in watch_name else [watch_name]
    return " ".join(display_name).title().replace("_", " ")
