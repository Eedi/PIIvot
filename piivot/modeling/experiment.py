"""A module for managing experiments."""

import json
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from pydantic import ValidationError
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from itertools import product

from piivot.utils import Config
from piivot.utils import Persistence
from .BERTDialogueDataset import BERTDialogueDataset, MultiSentenceBERTDialogueDataset
from .Tracker import WandbTracker
from .DialogueTrainer import DialogueTrainer
from .DialogueEvaluator import DialogueEvaluator
from piivot.utils.immutable import global_immutable
from piivot.utils.console import console
from piivot.utils import set_seed
from .tokenizer_factory import create_tokenizer
from .dataset_factory import create_dataset

class Experiment:
    """A class to manage experiments."""

    def __init__(
        self,
        results_dirpath: Path,
        config_filepath: Path,
        resume_checkpoint_filename: str | None = None,
        test: bool = False
    ):
        """Initializes the Experiment class.

        Given the experiment details it sets up persistence, loads the config, and sets up neccessary precursors to running an experiment.

        Args:
            results_dirpath (Path): The path to the results directory which will contain
            the individual experiment directories.
            config_filepath (Path): The path to the config file.
            resume_checkpoint_filename (str, optional): The name of the checkpoint file
            to resume training from. Defaults to None.

        """

        self.__setup__(results_dirpath, config_filepath, resume_checkpoint_filename, test)

        if not self.persistence.already_exists or global_immutable.rerun == True:
            self.__load_data__()
            self.__load_hyperparameters__()
            
            self.persistence.save_config(
                config=self.config
            )

    def __setup__(self, results_dirpath, config_filepath, resume_checkpoint_filename, test):
        console.rule(
            f"Step 1: Loading config and setting up persistence for file {config_filepath.name}"
        )
        
        candidate_exp_dirname = f"{config_filepath.stem}_final" if test else config_filepath.stem


        self.persistence = Persistence(
            results_dirpath=results_dirpath,
            candidate_exp_dirname=candidate_exp_dirname,
            resume_checkpoint_filename=resume_checkpoint_filename,
            test=test,
        )

        with open(config_filepath, "r") as f:
            raw_data = json.load(f)

        try:
            self.config = Config(**raw_data)
            
            if test:
                self.update_config_for_testing()
        except ValidationError as e:
            print(e)

        global_immutable.SEED = self.config.experiment.seed
        set_seed(global_immutable.SEED)

        print(f"DEVICE: {global_immutable.DEVICE}")

    def update_config_for_testing(self):
        self.config.input_data.split = False
        self.config.experiment.trainer.use_tqdm = True

    def create_and_save_subset(self, dataset: Dataset, indicies: List[int], name: str) -> Subset:
        self.persistence.save_data(dataset.data.loc[indicies][['FlowGeneratorSessionInterventionId',
                                                               'FlowGeneratorSessionInterventionMessageId',
                                                               'IsTutor',
                                                               'Message',
                                                               'Sequence',
                                                               'label']], name)
        return Subset(dataset, indicies)
        
    def create_dataset_split(self, 
                             full_dataset: Dataset, 
                             subset_indicies: List[int], 
                             monirity_labels: pd.Series, 
                             train_size: float, 
                             val_split: bool = False):
    
        train_idx, test_idx = train_test_split(np.arange(len(subset_indicies)),
                                                    train_size=train_size,
                                                    random_state=global_immutable.SEED,
                                                    shuffle=True,
                                                    stratify=monirity_labels)
        
        aug_test_idx = full_dataset.expand_group_indices(subset_indicies[test_idx])
        test_idx = subset_indicies[test_idx]
        
        train_minority_labels = monirity_labels.iloc[train_idx]

        aug_train_idx = full_dataset.expand_group_indices(subset_indicies[train_idx])
        train_idx = subset_indicies[train_idx]

        if val_split:
            synth_train_idx, synth_test_idx = full_dataset.get_synthetic_split_indices(train_size)
            aug_train_idx.extend(synth_train_idx)
            aug_test_idx.extend(synth_test_idx)
        
        return train_idx, aug_train_idx, train_minority_labels, test_idx, aug_test_idx

    def __load_data__(self):
        console.rule("Step 2: Loading the data.")

        self.tokenizer = create_tokenizer(self.config.experiment.model.params.name, 
                                          self.config.experiment.model.params.from_pretrained, 
                                          self.config.experiment.model.pretrained_params.pretrained_model_name_or_path)
        
        # TODO add logic for testing

        full_dataset = create_dataset(self.config.input_data.dataset.params.name,
                                      self.config.experiment.model.params.max_len, 
                                      self.tokenizer,
                                      self.config.input_data.dataset)
        
        self.config.experiment.model.pretrained_params.num_labels = len(full_dataset.labels)
        
        # map non-augmented indidices back to full_dataset 
        full_idx = np.array(full_dataset.get_non_augmented_indices())
        full_minority_labels = full_dataset.get_non_augmented_df()['minority_label']

        train_valid_idx, aug_train_valid_idx, train_valid_minority_labels, test_idx, _ = self.create_dataset_split(
            full_dataset, 
            full_idx, 
            full_minority_labels, 
            self.config.input_data.train_split,
            val_split= ~self.config.input_data.split)

        # self.test_dataloader = DataLoader(
        #     self.create_and_save_subset(
        #         full_dataset, 
        #         test_idx, 
        #         "test"), 
        #     **self.config.input_data.valid_params.model_dump())
        self.test_dataset = self.create_and_save_subset(
                full_dataset, 
                test_idx, 
                "test")
        
        if self.config.input_data.split:
            _, aug_train_idx, _, val_idx, aug_val_idx = self.create_dataset_split(
                full_dataset, 
                train_valid_idx, 
                train_valid_minority_labels, 
                1 - self.config.input_data.valid_split,
                val_split = True)
            
            # self.train_dataloader = DataLoader(
            #     self.create_and_save_subset(
            #         full_dataset, 
            #         aug_train_idx, 
            #         "train"), 
            #     **self.config.input_data.train_params.model_dump())
            self.train_dataset = self.create_and_save_subset(
                    full_dataset, 
                    aug_train_idx, 
                    "train")
            
            # self.valid_dataloaders = [
            #     DataLoader(
            #         self.create_and_save_subset(
            #             full_dataset, 
            #             val_idx, 
            #             "val_0"), 
            #         **self.config.input_data.valid_params.model_dump())]
            self.valid_datasets = [self.create_and_save_subset(
                full_dataset, 
                val_idx, 
                "val_0")]
        
            if self.config.input_data.dataset.params.augmented_non_pii or self.config.input_data.dataset.params.augmented_pii or self.config.input_data.dataset.params.add_synthetic:
                # self.valid_dataloaders.append(
                #     DataLoader(
                #         self.create_and_save_subset(
                #             full_dataset, 
                #             aug_val_idx, 
                #             "val_1"), 
                #         **self.config.input_data.valid_params.model_dump()))
                self.valid_datasets.append(self.create_and_save_subset(
                    full_dataset, 
                    aug_val_idx, 
                    "val_1"))
        else:
            self.train_dataset = self.create_and_save_subset(
                    full_dataset,
                    aug_train_valid_idx, 
                    "train")
            
            self.valid_datasets = []

        # train_valid_idx, test_idx = train_test_split(np.arange(len(full_indices)),
        #                                             train_size=self.config.input_data.train_split,
        #                                             random_state=global_immutable.SEED,
        #                                             shuffle=True,
        #                                             stratify=full_minority_labels)
        
        # test_idx = full_indices[test_idx]

        # train_valid_minority_labels = full_minority_labels.iloc[train_valid_idx]
        # train_valid_indices = full_indices[train_valid_idx]

        # test_dataset = self.create_and_save_subset(full_dataset, test_idx, "test")

        # train_idx, val_idx = train_test_split(np.arange(len(train_valid_indices)),
        #                                             test_size=self.config.input_data.valid_split,
        #                                             random_state=global_immutable.SEED,
        #                                             shuffle=True,
        #                                             stratify=train_valid_minority_labels)
        
        # aug_train_idx = full_dataset.expand_group_indices(train_valid_indices[train_idx])
        # val_idx = train_valid_indices[val_idx]
        
        # synth_train_idx, synth_val_idx = full_dataset.get_synthetic_split_indices(self.config.input_data.valid_split)
        # aug_train_idx.extend(synth_train_idx)
        
        # self.train_dataloader = DataLoader(self.create_and_save_subset(full_dataset, aug_train_idx, "train"), **self.config.input_data.train_params.model_dump())
        # self.valid_dataloaders = [DataLoader(self.create_and_save_subset(full_dataset, val_idx, "val_0"), **self.config.input_data.valid_params.model_dump())]
        # if self.config.input_data.dataset.params.augmented_non_pii or self.config.input_data.dataset.params.augmented_pii or self.config.input_data.dataset.params.add_synthetic:
        #     aug_val_idx = full_dataset.expand_group_indices(val_idx)
        #     aug_val_idx.extend(synth_val_idx)
        #     self.valid_dataloaders.append(DataLoader(self.create_and_save_subset(full_dataset, aug_val_idx, "val_1"), **self.config.input_data.valid_params.model_dump()))
        
        # self.test_dataloader = DataLoader(test_dataset, **self.config.input_data.valid_params.model_dump())
    def __load_hyperparameters__(self):
        lr = self.config.experiment.trainer.optimizer.params.lr
        epochs = self.config.experiment.trainer.epochs
        batch_size = self.config.input_data.train_params.batch_size

        if not isinstance(lr, list):
            lr = [lr]
        if not isinstance(epochs, list):
            epochs = [epochs]
        if not isinstance(batch_size, list):
            batch_size = [batch_size]

        param_combinations = product(lr, epochs, batch_size)
        self.hyperparameter_configs = []

        for lr_val, epochs_val, batch_size_val in param_combinations:
            new_config = self.config.model_copy(deep=True)
            new_config.experiment.trainer.optimizer.params.lr = lr_val
            new_config.experiment.trainer.epochs = epochs_val
            new_config.input_data.train_params.batch_size = batch_size_val
            
            self.hyperparameter_configs.append(new_config)

    def run_train(self):
        model = None
        if not self.persistence.already_exists or global_immutable.rerun == True:
            for i, hyper_config in enumerate(self.hyperparameter_configs):
                console.print(f"Starting training for hyperparameter config {i}.")

                train_dataloader = DataLoader(self.train_dataset, **hyper_config.input_data.train_params.model_dump())
                # test_dataloader = DataLoader(self.test_dataset, **hyper_config.input_data.valid_params.model_dump())

                if self.config.input_data.split:
                    evaluate_dataloaders = []
                    for valid_dataset in self.valid_datasets:
                        evaluate_dataloaders.append(DataLoader(valid_dataset, **hyper_config.input_data.valid_params.model_dump()))
                    evaluate_against = "val"

                else:
                    evaluate_dataloaders = [DataLoader(self.test_dataset, **hyper_config.input_data.valid_params.model_dump())]
                    evaluate_against = "test"
                
                exp_name = f"{self.persistence.exp_dirname}_{i}"
                resume = False
                config={
                    "learning_rate": hyper_config.experiment.trainer.optimizer.params.lr,
                    "epochs": hyper_config.experiment.trainer.epochs,
                }
                tracker = WandbTracker(exp_name, config, resume)

                trainer = DialogueTrainer(
                        tracker=tracker,
                        exp_name=exp_name,
                        experiment_config=hyper_config.experiment,
                        grad_clipping_max_norm=hyper_config.experiment.trainer.grad_clipping_max_norm, # TODO is this a value we want to hyper parameterize?
                    )
                
                evaluator = DialogueEvaluator(self.tokenizer, hyper_config.input_data.dataset.ids_to_labels)

                self.persistence.save_config(
                    config=hyper_config,
                    hyperparam_combination_id=i
                )

                try:
                    errors = trainer.train(
                            train_dataloader,
                            evaluate_dataloaders,
                            evaluator=evaluator,
                            num_epochs=hyper_config.experiment.trainer.epochs,
                            use_tqdm=hyper_config.experiment.trainer.use_tqdm,
                            val_frequency=hyper_config.experiment.trainer.val_every,
                            device=global_immutable.DEVICE
                        )

                    print("most recent epoch", trainer._epoch)

                    self.persistence.save_checkpoint(
                        trainer.model.state_dict(), 
                        trainer.optimizer.state_dict(), 
                        trainer._epoch, 
                        i,
                        hyper_config.input_data.dataset.ids_to_labels
                    )
                    model = trainer.model

                    print("model saved...")

                    if errors is None:
                        print("no validation used during model training. try increasing val_frequency.")
                    else:
                        self.persistence.save_errors(
                            evaluate_against,
                            i,
                            errors)

                    print("validation errors saved...")

                except KeyboardInterrupt:
                    print("most recent epoch", trainer._epoch)

                    self.persistence.save_checkpoint(
                        trainer.model.state_dict(), 
                        trainer.optimizer.state_dict(), 
                        trainer._epoch, 
                        i,
                        True
                    )

                tracker.end_run()
                
        return model, self.tokenizer if model else None

#     def generate_model_card_details(self):
#         '''Training Details
# Training Data
# [More Information Needed]

# Training Procedure
# Preprocessing [optional]
# [More Information Needed]

# Training Hyperparameters
# Training regime: [More Information Needed]
# Speeds, Sizes, Times [optional]
# [More Information Needed]'''
