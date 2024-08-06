import json
from pydantic import ValidationError
import torch
import re
from transformers import pipeline
import pandas as pd
from typing import List, Optional
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline, TokenClassificationPipeline
import warnings
from tqdm.autonotebook import tqdm

DEFAULT_DEVICE = 'cpu'

class Analyzer():
    '''Analyzer Engine.'''
    
    def __init__(self, 
                 pretrained_model_name_or_path, 
                 device=DEFAULT_DEVICE, 
                 surpress_warnings=True):
        if surpress_warnings:
            warnings.filterwarnings("ignore", module=".*token_classification")

        self.model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.token_classifier = pipeline(task="ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="first", device=device)

    def remove_trailing_punctuation(self, data: str, end_index: int) -> int:
        if end_index > 1 and data[end_index - 1] in ['.', ',', '!', '?']:
            substring = data[:end_index]
            pattern = r'[!?\.,]+$'
            cleaned_substring = re.sub(pattern, '', substring)
            
            end_index = len(cleaned_substring)
        
        if end_index > 2 and data[end_index - 2:end_index] == "'s":
            end_index = end_index - 2
        
        return end_index
    
    def remove_proceeding_space(self, data: str, start_index: int):
        substring = data[start_index:]

        non_whitespace_index = len(substring) - len(substring.lstrip())
        new_start_index = start_index + non_whitespace_index
        
        return new_start_index
    
    def group_data(self,
                   group, 
                   data_columns: List[str]):
        window_starts = []
        window_ends = []
        combined_messages = []
        for data_column in data_columns:
            for i in range(len(group)):
                previous_message = group[data_column].iloc[i - 1] if i > 0 else ''
                message = group[data_column].iloc[i]
                next_message = group[data_column].iloc[i + 1] if i < len(group) - 1 else ''

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
            
            group[f'window_start_{data_column}'] = window_starts
            group[f'window_end_{data_column}'] = window_ends
            group[f'combined_{data_column}'] = combined_messages
            
        return group
    
    def process_batch(self, df_batch, labels_column, original_data_column):
        for i in range(len(df_batch)):
            row = df_batch.iloc[i]
            trimmed_labels = []
            window_start = row[f'window_start_{original_data_column}']
            window_end = row[f'window_end_{original_data_column}']

            for label in row[labels_column]:
                if label['end'] > window_start and label['start'] < window_end:

                    # Neccessary for deberta based models
                    label_start = self.remove_proceeding_space(row[original_data_column], 
                                                               max(label['start'] - window_start, 
                                                                   0))
                    
                    # Optional. This is a choice to offload a few known edge-cases from the anonymization call. In theory, any un-removed punctuation should be maintained by the gpt anonymization.
                    label_end = self.remove_trailing_punctuation(row[original_data_column], 
                                                                 min(label['end'] - window_start, 
                                                                     len(row[original_data_column])))
                    
                    # If a labeled span is just punctuation or whitespace, label_end could be less than label_start
                    if label_end > label_start:
                        trimmed_labels.append((label_start, 
                                               label_end, 
                                               label['entity_group']))
            
            df_batch.at[row.name, labels_column] = trimmed_labels

        return df_batch

    # TODO could optionally provide labels to evaluate the analyze method
    def analyze(self, 
                df: pd.DataFrame, 
                data_columns: List[str],
                context_groups: List[str] = ['FlowGeneratorSessionInterventionId'],
                batch_size: int = 16,
                use_tqdm=True) -> pd.DataFrame:
        
        data_columns = [col if col in df.columns else warnings.warn(f"Column '{data_column}' does not exist in the input dataframe. Skipping...") for col in data_columns]
        original_data_columns = data_columns

        if context_groups:
            df = df.groupby(context_groups).apply(lambda group: self.group_data(group, data_columns)).reset_index(drop=True)
            
            data_columns = [f"combined_{data_column}" for data_column in data_columns]

        ds = Dataset.from_pandas(df[data_columns])
        
        if use_tqdm:
            data_columns = tqdm(data_columns, desc="Columns")

        for original_data_column, data_column in zip(original_data_columns, data_columns):
            new_label = f"{original_data_column}_labels"

            if use_tqdm:
                batches = tqdm(self.token_classifier(KeyDataset(ds, data_column), batch_size=batch_size), total=len(df), desc="Batches")
            else:
                batches = self.token_classifier(KeyDataset(ds, data_column), batch_size=batch_size)

            results = []
            for predictions in batches:
                results.extend([predictions])
            
            df[new_label] = results
            print("Predictions complete. Processing model output...")
            
            df = self.process_batch(df,
                                    new_label,
                                    original_data_column)
            
            df.drop(columns=[data_column], inplace=True)
                        
        return df
        