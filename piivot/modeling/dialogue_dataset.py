import json
import os
import pandas as pd
from torch.utils.data import Dataset
import datatest as dt
import ast

from piivot.utils.immutable import global_immutable

# from settings import DATA_PATH

def extract_flow_message_id(row):
    meta_data = json.loads(row['example_metadata'])
    return meta_data.get('FlowGeneratorSessionInterventionMessageId', None)

def extract_flow_id(row):
    meta_data = json.loads(row['example_metadata'])
    return meta_data.get('FlowGeneratorSessionInterventionId', None)

def get_annotation_scheme_prefixes(scheme):
    match scheme:
        case "IOB2":
            return ['B-','I-']
        case "IO":
            return ['I-']


class DialogueDataset(Dataset):
    def __init__(self, config): #augmented_nonpii, augmented_pii, add_synthetic)
        self.config = config
        self.__create_dataset__()
        self.len = len(self.data)

    # def __init__(self, df):
    #     dt.validate(df.columns, {'FlowGeneratorSessionInterventionId', 'FlowGeneratorSessionInterventionMessageId', 'Sequence', 'Message', 'pos_labels'})
    #     self.data = df
        
    #     self.len = len(self.data)

    def __len__(self):
        return self.len
    
    def __create_dataset__(self):
        # TODO change this to a path to a local file.
        # if augmented_nonpii:
        #     dialogue_path = os.path.join(f"{DATA_PATH}/augmented-dialogue_070424.csv") #TODO Not up to date
        # else:
        dialogue_path = os.path.join("/final_extract/extract-tutor-student-dialogues-202407261503-f3cly7v85j7vstbyo7z279te2disqgyx/extract-tutor-student-dialogues-202407261503-f3cly7v85j7vstbyo7z279te2disqgyx-PUBLIC/data/labeled-dialogue.csv")

        self.data = pd.read_csv(dialogue_path)
        if not self.config.params.add_synthetic and 'synthetic' in self.data.columns:
            self.data = self.data[~self.data.synthetic]
        if not self.config.params.augmented_non_pii and 'is_augmented' in self.data.columns:
            self.data = self.data[~self.data.is_augmented]

        self.data = self.data[self.data.manually_labeled].reset_index(drop=True)
        self.data['label'] = self.data['label'].apply(ast.literal_eval)

        self.data['label'] = self.data['label'].apply(lambda x: [label for label in x if label[2] not in self.config.params.exclude_labels])

        self.labels = set(label_name for labels in self.data['label'] for _, _, label_name in labels)
        self.labels = set(f"{prefix}{label}" for label in self.labels for prefix in get_annotation_scheme_prefixes(self.config.params.label_scheme))
        self.labels.add('O')
        self.labels = sorted(self.labels)
        self.config.ids_to_labels = self.labels
        # TODO find a better way to do this
        # global_immutable.LABELS_TO_IDS = {k: v for v, k in enumerate(self.labels)}
        # global_immutable.IDS_TO_LABELS = {v: k for v, k in enumerate(self.labels)}
