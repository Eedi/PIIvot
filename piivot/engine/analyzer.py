import warnings
from os import PathLike
from typing import List

import pandas as pd
from datasets import Dataset
from tqdm.autonotebook import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset

DEFAULT_DEVICE = "cpu"


class Analyzer:
    """Analyzer Engine."""

    punctuation_chars = [
        ".",
        ",",
        "!",
        "?",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "'",
        '"',
        ";",
        ":",
    ]

    def __init__(
        self,
        pretrained_model_name_or_path: str | PathLike[str],
        device: str = DEFAULT_DEVICE,
        surpress_warnings: bool = True,
    ):
        if surpress_warnings:
            warnings.filterwarnings("ignore", module=".*token_classification")

        self.model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.token_classifier = pipeline(
            task="ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="first",
            device=device,
        )

    def remove_trailing_punctuation(self, data: str, end_index: int) -> int:
        while end_index > 0 and data[end_index - 1] in Analyzer.punctuation_chars:
            end_index -= 1

        if end_index > 2 and data[end_index - 2 : end_index] == "'s":
            end_index = end_index - 2

        return end_index

    def remove_proceeding_space_and_punctuation(self, data: str, start_index: int):
        substring = data[start_index:]

        non_whitespace_index = len(substring) - len(substring.lstrip())
        start_index = start_index + non_whitespace_index

        while (
            start_index <= len(data) and data[start_index] in Analyzer.punctuation_chars
        ):
            start_index += 1

        return start_index

    def group_data(self, group, data_columns: List[str]):
        """Group the data based on the provided grouping criteria by joining preceding and trailing data columns into combined_{data_column}.

        Args:
            group (pd.DataFrame): A DataFrame from a df.groupby() call.
            data_columns (List[str]): A list of column names in the DataFrame that contain the data to be grouped and analyzed.

        Returns:
            pd.DataFrame: The group with window_start, window_end, and combined columns for each data_column.
        """
        window_starts = []
        window_ends = []
        combined_messages = []
        for data_column in data_columns:
            for i in range(len(group)):
                previous_message = group[data_column].iloc[i - 1] if i > 0 else ""
                message = group[data_column].iloc[i]
                next_message = (
                    group[data_column].iloc[i + 1] if i < len(group) - 1 else ""
                )

                if previous_message:
                    window_start = len(previous_message) + 1
                    window_end = len(previous_message) + 1 + len(message)
                    combined_message = f"{previous_message} {message}"
                else:
                    window_start = 0
                    window_end = len(message)
                    combined_message = message

                if next_message:
                    combined_message = f"{combined_message} {next_message}"

                window_starts.append(window_start)
                window_ends.append(window_end)
                combined_messages.append(combined_message)

            group[f"window_start_{data_column}"] = window_starts
            group[f"window_end_{data_column}"] = window_ends
            group[f"combined_{data_column}"] = combined_messages

        return group

    def process_batch(
        self,
        df_batch: pd.DataFrame,
        labels_column: str,
        original_data_column: str,
        context_groups: List[str],
    ) -> pd.DataFrame:
        """Process a batch of labels. This involves adjusting indices for windowing (if context_groups were provided),
        removing preceding whitespace, and removing punctuation.

        Args:
            df_batch (pd.DataFrame): The batch of the DataFrame to be processed. This can be the entire DataFrame.
            labels_column (str): The name of the column where the labels will be stored.
            original_data_column (str): The original name of the column containing the data to be analyzed.
            context_groups (List[str]): A list of column names to group data for analysis. If provided,
                prior and proceeding data_columns will be combined and used as input to the model.

        Returns:
            pd.DataFrame: A DataFrame containing the processed batch. The specified labels column will be
            updated with new labels, and the original data column will be analyzed according to the context groups.
        """
        for i in range(len(df_batch)):
            row = df_batch.iloc[i]
            trimmed_labels = []

            if context_groups:
                window_start = row[f"window_start_{original_data_column}"]
                window_end = row[f"window_end_{original_data_column}"]

                for label in row[labels_column]:
                    if label["end"] > window_start and label["start"] < window_end:
                        # Neccessary for deberta based models
                        label_start = self.remove_proceeding_space_and_punctuation(
                            row[original_data_column],
                            max(label["start"] - window_start, 0),
                        )

                        # Optional. This is a choice to offload a few known edge-cases from the anonymization call. In theory, any un-removed punctuation should be maintained by the gpt anonymization.
                        label_end = self.remove_trailing_punctuation(
                            row[original_data_column],
                            min(
                                label["end"] - window_start,
                                len(row[original_data_column]),
                            ),
                        )

                        # If a labeled span is just punctuation or whitespace, label_end could be less than label_start
                        if label_end > label_start:
                            trimmed_labels.append(
                                (label_start, label_end, label["entity_group"])
                            )
            else:
                for label in row[labels_column]:
                    label_start = self.remove_proceeding_space_and_punctuation(
                        row[original_data_column], label["start"]
                    )
                    label_end = self.remove_trailing_punctuation(
                        row[original_data_column], label["end"]
                    )
                    if label_end > label_start:
                        trimmed_labels.append(
                            (label_start, label_end, label["entity_group"])
                        )

            df_batch.at[row.name, labels_column] = trimmed_labels

        return df_batch

    # TODO could optionally provide labels to evaluate the analyze method
    def analyze(
        self,
        df: pd.DataFrame,
        data_columns: List[str],
        context_groups: List[str] = None,
        batch_size: int = 16,
        use_tqdm=True,
    ) -> pd.DataFrame:
        """Analyze specified data columns of the given DataFrame for entities with possible PII by processing.

        Args:
            df (pd.DataFrame): The input DataFrame to be analyzed.
            data_columns (List[str]): A list of column names in the DataFrame that contain the data to be analyzed.
            context_groups (List[str], optional): A list of column names to group data for analysis. If provided,
                prior and proceeding data_columns will be combined and used as input to the model. Default is None.
            batch_size (int, optional): The number of rows to process in each batch to the GPU. Default is 16.
            use_tqdm (bool, optional): Whether to display a progress bar using tqdm. Default is True.

        Returns:
            pd.DataFrame: A DataFrame containing the results of the analysis. New columns in the form '{data_column}_labels' are
            appended to the original DataFrame and contain lists of spans in the form (start_index, end_index, label_name) of the newly
            identified entities.
        """
        input_length = len(data_columns)
        for i, data_column in enumerate(reversed(data_columns)):
            original_i = input_length - 1 - i
            if data_column not in df.columns:
                warnings.warn(
                    f"Column '{data_column}' does not exist in the input dataframe. Skipping..."
                )
                data_columns.pop(original_i)

        original_data_columns = data_columns

        if context_groups:
            df = (
                df.groupby(context_groups)
                .apply(lambda group: self.group_data(group, data_columns))
                .reset_index(drop=True)
            )

            data_columns = [f"combined_{data_column}" for data_column in data_columns]

        ds = Dataset.from_pandas(df[data_columns])

        if use_tqdm:
            zipped_data_columns = tqdm(
                zip(original_data_columns, data_columns), desc="Columns"
            )
        else:
            zipped_data_columns = zip(original_data_columns, data_columns)

        for original_data_column, data_column in zipped_data_columns:
            new_label = f"{original_data_column}_labels"

            if use_tqdm:
                batches = tqdm(
                    self.token_classifier(
                        KeyDataset(ds, data_column), batch_size=batch_size
                    ),
                    total=len(df),
                    desc="Batches",
                )
            else:
                batches = self.token_classifier(
                    KeyDataset(ds, data_column), batch_size=batch_size
                )

            results = []
            for predictions in batches:
                results.extend([predictions])

            df[new_label] = results
            print("Predictions complete. Processing model output...")

            df = self.process_batch(df, new_label, original_data_column, context_groups)

            if context_groups:
                df.drop(
                    columns=[
                        data_column,
                        f"window_start_{original_data_column}",
                        f"window_end_{original_data_column}",
                    ],
                    inplace=True,
                )

        return df
