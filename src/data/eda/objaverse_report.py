import logging
import os
import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from ydata_profiling import ProfileReport

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

CSV_INPUT_PATH = "objaverse_eda_results.csv"
PROFILE_REPORT_OUTPUT_PATH = "objaverse_ydata_profile.html"
VISUALIZATIONS_DIR = "objaverse_visualizations"
TOPIC_MODELING_OUTPUT_PATH = "objaverse_lda_topics.csv"  # New constant for LDA results

os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Successfully loaded data from {csv_path}. Shape: {df.shape}")
        expected_cols = [
            "uuid",
            "file_size_bytes",
            "prompt",
            "prompt_length",
            "render_count",
            "average_contrast",
        ]
        missing_expected = [col for col in expected_cols if col not in df.columns]
        if missing_expected:
            logging.warning(
                f"CSV is missing some expected columns: {missing_expected}. Found columns: {list(df.columns)}"
            )
        else:
            logging.info(f"All expected columns found: {expected_cols}")

        # Log details for key columns
        key_cols_to_check = [
            "file_size_bytes",
            "average_contrast",
            "render_count",
            "prompt",
        ]
        for col in key_cols_to_check:
            if col in df.columns:
                logging.info(f"Column '{col}' details:")
                logging.info(f"  dtype: {df[col].dtype}")
                logging.info(f"  Number of NaNs: {df[col].isna().sum()}")
                if pd.api.types.is_numeric_dtype(df[col]):
                    logging.info(f"  Basic stats:\n{df[col].describe()}")
                else:
                    logging.info(
                        f"  Unique values (sample): {df[col].unique()[:5]}"
                    )  # Show a few unique values for non-numeric
            else:
                logging.warning(f"Key column '{col}' not found in DataFrame.")

        return df
    except FileNotFoundError:
        logging.error(f"Error: The file {csv_path} was not found.")
        return None
    except Exception as e:
        logging.error(f"Error loading CSV {csv_path}: {e}")
        return None


def generate_profile_report(df: pd.DataFrame, output_path: str):
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
    if df is None or df.empty:
        logging.warning("DataFrame is empty. Skipping custom visualizations.")
        return

    logging.info("--- DataFrame Info before custom visualizations ---")
    df_info_str_buffer = io.StringIO()
    df.info(buf=df_info_str_buffer)
    logging.info(df_info_str_buffer.getvalue())
    logging.info("---------------------------------------------------")

    # Log descriptive statistics for key numerical columns
    key_numerical_cols = [
        "file_size_bytes",
        "average_contrast",
        "render_count",
        "prompt_length",
    ]
    for col in key_numerical_cols:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                logging.info(
                    f"Descriptive statistics for '{col}':\n{df[col].describe()}"
                )
                if df[col].isna().any():
                    logging.warning(
                        f"Column '{col}' contains {df[col].isna().sum()} NaN values. This might affect calculations or plots."
                    )
            else:
                logging.warning(
                    f"Column '{col}' is not numeric (dtype: {df[col].dtype}). Statistics and some plots might fail or be incorrect."
                )
        else:
            logging.warning(
                f"Key numerical column '{col}' not found for pre-visualization checks."
            )

    # Check 'prompt' column specifically for text processing
    if "prompt" in df.columns:
        logging.info(
            f"'prompt' column: dtype={df['prompt'].dtype}, NaNs={df['prompt'].isna().sum()}, Sample unique values: {df['prompt'].unique()[:3]}"
        )
    else:
        logging.warning(
            "'prompt' column not found, word cloud and LDA will be skipped."
        )

    logging.info("Generating custom visualizations...")
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # --- Calculate and Log Key Statistics ---
    if "average_contrast" in df.columns and pd.api.types.is_numeric_dtype(
        df["average_contrast"]
    ):
        mean_contrast = df["average_contrast"].mean()
        std_contrast = df["average_contrast"].std()
        logging.info(
            f"Average Contrast: {mean_contrast:.4f}, Standard Deviation: {std_contrast:.4f}"
        )
    else:
        mean_contrast, std_contrast = None, None
        logging.warning(
            "Could not calculate statistics for average_contrast (not found or not numeric)."
        )

    if "file_size_bytes" in df.columns and pd.api.types.is_numeric_dtype(
        df["file_size_bytes"]
    ):
        mean_filesize = df["file_size_bytes"].mean()
        std_filesize = df["file_size_bytes"].std()
        total_filesize = df["file_size_bytes"].sum()
        logging.info(
            f"Average File Size: {mean_filesize:.2f} bytes, Standard Deviation: {std_filesize:.2f} bytes, Total Size: {total_filesize} [MB]"
        )
    else:
        mean_filesize, std_filesize, total_filesize = None, None, None
        logging.warning(
            "Could not calculate statistics for file_size_bytes (not found or not numeric)."
        )

    if "prompt_length" in df.columns and pd.api.types.is_numeric_dtype(df["prompt_length"]):
        mean_prompt_length = df["prompt_length"].mean()
        std_prompt_length = df["prompt_length"].std()
        logging.info(
            f"IMPORTANT: IMPORTANT:IMPORTANT:IMPORTANT:IMPORTANT:IMPORTANT:IMPORTANT:IMPORTANT:IMPORTANT:IMPORTANT:IMPORTANT:IMPORTANT: Average Prompt Length: {mean_prompt_length:.2f}, Standard Deviation: {std_prompt_length:.2f}"
        )
    else:
        mean_prompt_length, std_prompt_length = None, None
        logging.warning("Could not calculate statistics for prompt_length (not found or not numeric).")

    numerical_cols = [
        "file_size_bytes",
        "prompt_length",
        "average_contrast",
    ]
    for col in numerical_cols:
        if col in df.columns:
            plt.figure(figsize=(14, 5.5))
            plot_data = df[col].dropna()  # Data for plotting and CSV
            sns.histplot(plot_data, kde=True, kde_kws={'gridsize': 500})

            # title = f"Distribution of {col.replace('_', ' ').title()}"
            
            # Add lines for mean and std dev if applicable
            if col == "average_contrast" and mean_contrast is not None and std_contrast is not None:
                # title += f" ($\mu={mean_contrast:.2f}, \sigma={std_contrast:.2f}$)"
                plt.axvline(mean_contrast, color='r', linestyle='--', linewidth=2, label='$\mu$')
                plt.axvline(mean_contrast + std_contrast, color='g', linestyle=':', linewidth=2, label='$\mu \pm \sigma$')
                plt.axvline(mean_contrast - std_contrast, color='g', linestyle=':', linewidth=2)
                plt.legend()
            elif col == "file_size_bytes" and mean_filesize is not None and std_filesize is not None:
                # mean_filesize_mb = mean_filesize / (1024 * 1024)
                # std_filesize_mb = std_filesize / (1024 * 1024)
                # title += f" ($\mu={mean_filesize_mb:.2f}MB, \sigma={std_filesize_mb:.2f}MB$)"
                
                # Set x-axis limit for file_size_bytes plot
                cap_value_bytes = 0.25 * 1e9 # 0.25 * 10^9 bytes, as per user change
                current_xlim = plt.gca().get_xlim() # Keep this to get the original right limit if needed
                # Ensure we don't set xlim to something that hides all data if max data is less than cap
                # And also respect if data naturally ends before the cap. Start x-axis at 0.
                plt.xlim(left=0, right=min(current_xlim[1], cap_value_bytes))
                logging.info(f"Set x-axis for file_size_bytes distribution from 0 to {min(current_xlim[1], cap_value_bytes) / (1024*1024):.2f} MB.")

                # Format x-axis ticks to display in MB
                ticks_in_bytes = plt.gca().get_xticks()
                tick_labels_mb = [f'{tick / (1024*1024):.0f}' for tick in ticks_in_bytes] # Display as integer MB
                plt.xticks(ticks=ticks_in_bytes, labels=tick_labels_mb)
                plt.xlabel("File Size (MB)") # Update x-axis label

                if total_filesize is not None:
                    logging.info(
                        f"Total data size (all files): {total_filesize / (1024 * 1024 * 1024):.2f} GB"
                    )

            elif col == "prompt_length" and mean_prompt_length is not None and std_prompt_length is not None:
                # title += f" ($\mu={mean_prompt_length:.2f}, \sigma={std_prompt_length:.2f}$)" # Titles are off
                plt.axvline(mean_prompt_length, color='r', linestyle='--', linewidth=2, label='$\mu$')
                plt.axvline(mean_prompt_length + std_prompt_length, color='g', linestyle=':', linewidth=2, label='$\mu \pm \sigma$')
                plt.axvline(mean_prompt_length - std_prompt_length, color='g', linestyle=':', linewidth=2)
                plt.legend()

            # General x-label for plots not specifically handled above or if specific handling removed
            if col != "file_size_bytes":
                plt.xlabel(col.replace("_", " ").title())
            plt.ylabel("Frequency")
            save_path_img = os.path.join(output_dir, f"distribution_{col}.png")
            save_path_csv = os.path.join(output_dir, f"distribution_{col}_data.csv")
            try:
                plt.savefig(save_path_img, dpi=600)
                logging.info(f"Saved {save_path_img} (600 dpi)")
                # Save data to CSV
                plot_data.to_csv(save_path_csv, index=False, header=[col])
                logging.info(f"Saved plot data to {save_path_csv}")
            except Exception as e:
                logging.error(f"Failed to save plot/data for {col}: {e}")
            plt.close()
        else:
            logging.warning(f"Column {col} not found for distribution visualization.")

    # --- Bar Plot for render_count ---
    if "render_count" in df.columns:
        plt.figure(figsize=(14, 6))
        # Create a DataFrame of value counts for the bar plot
        render_count_data = df["render_count"].value_counts().sort_index().reset_index()
        render_count_data.columns = ["render_count", "frequency"]

        sns.barplot(
            x="render_count", y="frequency", data=render_count_data, palette="viridis"
        )
        # plt.title("Distribution of Render Count (Column Chart)")
        plt.xlabel("Render Count")
        plt.ylabel("Frequency (Number of Occurrences)")
        plt.tight_layout()  # Adjust layout
        save_path_img = os.path.join(output_dir, "distribution_render_count_bar.png")
        save_path_csv = os.path.join(
            output_dir, "distribution_render_count_bar_data.csv"
        )
        try:
            plt.savefig(save_path_img, dpi=600)
            logging.info(f"Saved {save_path_img} (600 dpi)")
            # Save data to CSV
            render_count_data.to_csv(save_path_csv, index=False)
            logging.info(f"Saved plot data to {save_path_csv}")
        except Exception as e:
            logging.error(f"Failed to save plot/data for render_count bar chart: {e}")
        plt.close()
    else:
        logging.warning("Column render_count not found for bar chart visualization.")

    scatter_pairs = [
        ("prompt_length", "file_size_bytes"),
        ("average_contrast", "prompt_length"),
        ("render_count", "file_size_bytes"),
        ("render_count", "average_contrast"),
    ]
    for x_col, y_col in scatter_pairs:
        if x_col in df.columns and y_col in df.columns:
            plt.figure(figsize=(14, 5.5))
            sns.scatterplot(x=df[x_col], y=df[y_col], alpha=0.5)
            # plt.title(
            #     f"{x_col.replace('_', ' ').title()} vs. {y_col.replace('_', ' ').title()}"
            # )
            plt.xlabel(x_col.replace("_", " ").title())
            plt.ylabel(y_col.replace("_", " ").title())
            save_path_img = os.path.join(output_dir, f"scatter_{x_col}_vs_{y_col}.png")
            save_path_csv = os.path.join(
                output_dir, f"scatter_{x_col}_vs_{y_col}_data.csv"
            )
            try:
                plt.savefig(save_path_img, dpi=600)
                logging.info(f"Saved {save_path_img} (600 dpi)")
                # Save data to CSV
                df[[x_col, y_col]].dropna().to_csv(save_path_csv, index=False)
                logging.info(f"Saved plot data to {save_path_csv}")
            except Exception as e:
                logging.error(
                    f"Failed to save plot/data for scatter {x_col} vs {y_col}: {e}"
                )
            plt.close()
        else:
            logging.warning(
                f"One or both columns ({x_col}, {y_col}) not found for scatter plot."
            )

    # --- Bar Plot: Mean average_contrast per render_count ---
    if "render_count" in df.columns and "average_contrast" in df.columns:
        plt.figure(figsize=(14, 5.5))
        mean_contrast_by_renders_data = (
            df.groupby("render_count")["average_contrast"].mean().reset_index()
        )
        sns.barplot(
            x="render_count",
            y="average_contrast",
            data=mean_contrast_by_renders_data,
            palette="viridis",
        )
        # plt.title("Mean Average Contrast per Render Count")
        plt.xlabel("Render Count")
        plt.ylabel("Mean Average Contrast")
        save_path_img = os.path.join(
            output_dir, "barplot_mean_contrast_vs_render_count.png"
        )
        save_path_csv = os.path.join(
            output_dir, "barplot_mean_contrast_vs_render_count_data.csv"
        )
        try:
            plt.savefig(save_path_img, dpi=600)
            logging.info(f"Saved {save_path_img} (600 dpi)")
            # Save data to CSV
            mean_contrast_by_renders_data.to_csv(save_path_csv, index=False)
            logging.info(f"Saved plot data to {save_path_csv}")
        except Exception as e:
            logging.error(
                f"Failed to save plot/data for mean contrast vs render count: {e}"
            )
        plt.close()
    else:
        logging.warning(
            "Columns 'render_count' or 'average_contrast' not found for bar plot."
        )

    stopwords = set(
        [
            "create",
            "image",
            "the",
            "and",
            "made",
            "a",
            "it",
            "has",
            "a",
            "an",
            "the",
            "this",
            "that",
            "these",
            "makes",
            "of",
            "should",
            "shape",
            "with",
            "overall",
            "without",
            "and",
            "or",
            "not",
            "but",
            "if",
            "else",
            "elif",
            "while",
            "for",
            "in",
            "to",
            "it",
            "as",
            "from",
            "by",
            "on",
            "off",
            "up",
            "down",
            "left",
            "right",
            "center",
            "top",
            "bottom",
            "front",
            "back",
            "be",
            "giving",
            "have",
            "object",
            "slightly",
            "three",
            "dimensional",
            "edge",
            "creating",
            "shade",
            "possibly",
            "texture",
            "is",
            "are",
            "model",
            "render",
            "rendering",
            "style",
            "detailed",
            "realistic",
            "view",
            "angle",
            "high",
            "quality",
            "low",
            "poly",
            "game",
            "asset",
            "art",
            "abstract",
            "background",
            "light",
            "lighting",
            "shadow",
            "color",
            "surface",
            "material",
            "minimalist",
            "futuristic",
            "vintage",
            "concept",
        ]
    )
    # --- Word Cloud for Prompts ---
    if "prompt" in df.columns and df["prompt"].notna().any():
        logging.info("Generating word cloud for prompts...")
        # Ensure prompts are strings and handle potential NaN/float values gracefully
        prompts_text = " ".join(df["prompt"].astype(str).dropna())
        if prompts_text.strip():  # Check if there's any text to process
            try:
                wordcloud = WordCloud(
                    width=800, height=400, background_color="white", stopwords=stopwords
                ).generate(prompts_text)
                plt.figure(figsize=(14, 7))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                # plt.title("Word Cloud of Prompts")
                save_path = os.path.join(output_dir, "wordcloud_prompts.png")
                plt.savefig(save_path, dpi=600)
                logging.info(f"Saved {save_path} with increased resolution (600 dpi)")
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

    # --- Topic Modeling for Prompts using LDA ---
    if "prompt" in df.columns and df["prompt"].notna().any():
        logging.info("Starting topic modeling for prompts using LDA...")
        prompts_for_lda = df["prompt"].astype(str).dropna().tolist()

        if len(prompts_for_lda) > 10:  # Basic check for sufficient data
            try:
                # Use the same stopwords list as defined for the word cloud
                # The 'stopwords' variable should be in scope from the Word Cloud section above.
                # If Word Cloud section is moved or stopwords definition changes, this might need adjustment.

                vectorizer = CountVectorizer(
                    max_df=0.95,
                    min_df=2,
                    stop_words=list(
                        stopwords
                    ),  # Directly use the 'stopwords' set from word cloud part
                    lowercase=True,
                )
                dtm = vectorizer.fit_transform(prompts_for_lda)
                feature_names = vectorizer.get_feature_names_out()

                num_topics = 20
                num_top_words = 10
                logging.info(
                    f"Attempting to find {num_topics} topics, with {num_top_words} top words each."
                )

                lda = LatentDirichletAllocation(
                    n_components=num_topics, random_state=42, learning_method="online"
                )
                lda.fit(dtm)

                logging.info(f"Top words for {num_topics} topics found via LDA:")
                topics_data = []
                for topic_idx, topic_weights in enumerate(lda.components_):
                    top_words_indices = topic_weights.argsort()[
                        : -num_top_words - 1 : -1
                    ]
                    top_words_list = [feature_names[i] for i in top_words_indices]
                    topic_label = f"Topic #{topic_idx + 1}"
                    logging.info(f"{topic_label}: {', '.join(top_words_list)}")
                    topics_data.append(
                        {
                            "Topic_ID": topic_label,
                            "Top_Words": ", ".join(top_words_list),
                        }
                    )

                topics_df = pd.DataFrame(topics_data)
                try:
                    topics_df.to_csv(TOPIC_MODELING_OUTPUT_PATH, index=False)
                    logging.info(
                        f"Successfully saved LDA topics to {TOPIC_MODELING_OUTPUT_PATH}"
                    )
                except Exception as e:
                    logging.error(f"Failed to save LDA topics CSV: {e}")

            except Exception as e:
                logging.error(f"Error during LDA topic modeling: {e}")
        else:
            logging.warning(
                f"Not enough prompts ({len(prompts_for_lda)}) for meaningful topic modeling. Skipping."
            )
    else:
        logging.warning(
            "Column 'prompt' not found or contains no text data for topic modeling."
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
