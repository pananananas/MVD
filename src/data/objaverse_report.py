import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from ydata_profiling import ProfileReport

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---
CSV_INPUT_PATH = "objaverse_eda_results.csv"
PROFILE_REPORT_OUTPUT_PATH = "objaverse_ydata_profile.html"
VISUALIZATIONS_DIR = "objaverse_visualizations"

# Create directory for saving plots if it doesn't exist
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame | None:
    """Loads data from the CSV file."""
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Successfully loaded data from {csv_path}. Shape: {df.shape}")
        # Basic check for expected columns
        expected_cols = [
            "uuid",
            "file_size_bytes",
            "prompt",
            "prompt_length",
            "render_count",
            "average_contrast",
        ]
        if not all(col in df.columns for col in expected_cols):
            logging.warning(
                f"CSV is missing some expected columns. Found: {list(df.columns)}. Expected at least: {expected_cols}"
            )
        return df
    except FileNotFoundError:
        logging.error(f"Error: The file {csv_path} was not found.")
        return None
    except Exception as e:
        logging.error(f"Error loading CSV {csv_path}: {e}")
        return None


def generate_profile_report(df: pd.DataFrame, output_path: str):
    """Generates and saves a ydata-profiling report."""
    if df is None or df.empty:
        logging.warning("DataFrame is empty. Skipping profile report generation.")
        return
    logging.info("Generating ydata-profiling report...")
    try:
        profile = ProfileReport(
            df, title="Objaverse EDA Profile Report", explorative=True
        )
        profile.to_file(output_path)
        logging.info(f"Successfully saved ydata-profiling report to {output_path}")
    except Exception as e:
        logging.error(f"Error generating or saving profile report: {e}")


def create_custom_visualizations(df: pd.DataFrame, output_dir: str):
    """Creates and saves custom visualizations."""
    if df is None or df.empty:
        logging.warning("DataFrame is empty. Skipping custom visualizations.")
        return

    logging.info("Generating custom visualizations...")
    sns.set_theme(style="whitegrid")

    # --- Numerical Distributions ---
    numerical_cols = [
        "file_size_bytes",
        "prompt_length",
        "render_count",
        "average_contrast",
    ]
    for col in numerical_cols:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col.replace('_', ' ').title()}")
            plt.xlabel(col.replace("_", " ").title())
            plt.ylabel("Frequency")
            save_path = os.path.join(output_dir, f"distribution_{col}.png")
            try:
                plt.savefig(save_path)
                logging.info(f"Saved {save_path}")
            except Exception as e:
                logging.error(f"Failed to save plot {save_path}: {e}")
            plt.close()
        else:
            logging.warning(f"Column {col} not found for visualization.")

    # --- Scatter Plots for Relationships ---
    scatter_pairs = [
        ("prompt_length", "file_size_bytes"),
        ("average_contrast", "prompt_length"),
        ("render_count", "file_size_bytes"),
        ("render_count", "average_contrast"),
    ]
    for x_col, y_col in scatter_pairs:
        if x_col in df.columns and y_col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df[x_col], y=df[y_col], alpha=0.5)
            plt.title(
                f"{x_col.replace('_', ' ').title()} vs. {y_col.replace('_', ' ').title()}"
            )
            plt.xlabel(x_col.replace("_", " ").title())
            plt.ylabel(y_col.replace("_", " ").title())
            save_path = os.path.join(output_dir, f"scatter_{x_col}_vs_{y_col}.png")
            try:
                plt.savefig(save_path)
                logging.info(f"Saved {save_path}")
            except Exception as e:
                logging.error(f"Failed to save plot {save_path}: {e}")
            plt.close()
        else:
            logging.warning(
                f"One or both columns ({x_col}, {y_col}) not found for scatter plot."
            )

    # --- Bar Plot: Mean average_contrast per render_count ---
    if "render_count" in df.columns and "average_contrast" in df.columns:
        plt.figure(figsize=(10, 6))
        mean_contrast_by_renders = (
            df.groupby("render_count")["average_contrast"].mean().reset_index()
        )
        sns.barplot(
            x="render_count",
            y="average_contrast",
            data=mean_contrast_by_renders,
            palette="viridis",
        )
        plt.title("Mean Average Contrast per Render Count")
        plt.xlabel("Render Count")
        plt.ylabel("Mean Average Contrast")
        save_path = os.path.join(
            output_dir, "barplot_mean_contrast_vs_render_count.png"
        )
        try:
            plt.savefig(save_path)
            logging.info(f"Saved {save_path}")
        except Exception as e:
            logging.error(f"Failed to save plot {save_path}: {e}")
        plt.close()
    else:
        logging.warning(
            "Columns 'render_count' or 'average_contrast' not found for bar plot."
        )

    # --- Word Cloud for Prompts ---
    if "prompt" in df.columns and df["prompt"].notna().any():
        logging.info("Generating word cloud for prompts...")
        # Ensure prompts are strings and handle potential NaN/float values gracefully
        prompts_text = " ".join(df["prompt"].astype(str).dropna())
        if prompts_text.strip():  # Check if there's any text to process
            try:
                wordcloud = WordCloud(
                    width=800, height=400, background_color="white"
                ).generate(prompts_text)
                plt.figure(figsize=(12, 8))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.title("Word Cloud of Prompts")
                save_path = os.path.join(output_dir, "wordcloud_prompts.png")
                plt.savefig(save_path)
                logging.info(f"Saved {save_path}")
                plt.close()
            except Exception as e:
                logging.error(f"Error generating word cloud: {e}")
        else:
            logging.warning(
                "No valid text found in 'prompt' column for word cloud generation."
            )
    else:
        logging.warning(
            "Column 'prompt' not found or contains no text data for word cloud."
        )

    logging.info("Custom visualizations generation complete.")


def main():
    logging.info(
        "Starting Objaverse EDA Report script (Phase 2: Profiling and Visualization)..."
    )

    df = load_data(CSV_INPUT_PATH)

    if df is not None:
        generate_profile_report(df, PROFILE_REPORT_OUTPUT_PATH)
        create_custom_visualizations(df, VISUALIZATIONS_DIR)
    else:
        logging.error("Failed to load data. Exiting report generation.")

    logging.info("--- Objaverse EDA Report script finished. ---")


if __name__ == "__main__":
    main()
