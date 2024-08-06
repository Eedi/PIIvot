
from collections import defaultdict
from typing import Dict, Tuple, List
import numpy as np
from pandas import DataFrame
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    balanced_accuracy_score,
)

from .Tracker import Tracker
from piivot.utils.immutable import global_immutable

DEFAULT_DEVICE = "cpu"

def update_history(history, cur_k_fold, cur_epoch, metrics):
    if cur_k_fold not in history:
        history[cur_k_fold] = {}
    
    if cur_epoch not in history[cur_k_fold]:
        history[cur_k_fold][cur_epoch] = {}
    
    history[cur_k_fold][cur_epoch].update(metrics)

def analyze_history(history):
    max_accuracies = {}
    max_accuracy_epochs = {}
    min_losses = {}
    min_loss_epochs = {}

    for k, epoch in history.items():
        accuracies = []
        losses = []
        
        for cur_epoch, metrics in epoch.items():
            accuracies.append(metrics.get('accuracy', 0))
            losses.append(metrics.get('loss', 0))
        
        if accuracies and (~max_accuracies[k] or max(accuracies) > max_accuracies[k]):
            max_accuracies[k] = max(accuracies)
            max_accuracy_epochs[k] = epoch
        if losses and (~min_losses[k] or  min(losses) < min_losses[k]):
            min_losses[k] = min(losses)
            min_loss_epochs[k] = epoch
    
    # Convert max accuracy and loss values to numpy arrays for mean and std calculations
    max_accuracy_values = np.array(list(max_accuracies.values()))
    max_loss_values = np.array(list(min_losses.values()))
    
    # Calculate statistics
    stats = {
        'max_accuracies': max_accuracies,
        'max_accuracy_epochs': max_accuracy_epochs,
        'min_losses': min_losses,
        'min_loss_epochs': min_loss_epochs,
        'mean_max_accuracy': np.mean(max_accuracy_values),
        'std_max_accuracy': np.std(max_accuracy_values),
        'mean_min_loss': np.mean(max_loss_values),
        'std_min_loss': np.std(max_loss_values)
    }
    
    return stats

class DialogueEvaluator:
    def __init__(
        self,
        tokenizer,
        ids_to_labels
    ):
        """Class for evaluating PII predictions
        """
        self.history = {}
        self.tokenizer = tokenizer
        self.ids_to_labels = ids_to_labels

    def classification_report_custom(self, y_true, y_pred, labels=None):
        """
        Build a text report showing the main classification metrics.

        Parameters:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
            labels (array-like, optional): List of labels to include in the report.

        Returns:
            str: Text report showing the main classification metrics.
        """

        tp = defaultdict(int) 
        fp = defaultdict(int)
        fn = defaultdict(int)
        support = defaultdict(int)

        for true, pred in zip(y_true, y_pred):
            if true == pred:
                tp[true] += 1
            else:
                fp[pred] += 1
                fn[true] += 1
            support[true] += 1

        precision = {}
        recall = {}
        f1_score = {}

        for label in set(y_true + y_pred):
            precision[label] = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) != 0 else 0
            recall[label] = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) != 0 else 0
            f1_score[label] = (2 * precision[label] * recall[label]) / (precision[label] + recall[label]) if (precision[label] + recall[label]) != 0 else 0

        macro_precision = np.mean(list(precision.values()))
        macro_recall = np.mean(list(recall.values()))
        macro_f1_score = np.mean(list(f1_score.values()))

        report = "              precision    recall  f1-score   support\n\n"

        for label in sorted(set(y_true + y_pred)):

            report += f"    {self.ids_to_labels[label]:<10}   {precision.get(label, 0):<9.2f}   {recall.get(label, 0):<6.2f}   {f1_score.get(label, 0):<8.2f}   {support.get(label, 0):<7}\n"

        report += "\n"

        report += f"    macro avg   {macro_precision:<9.2f}   {macro_recall:<6.2f}   {macro_f1_score:<8.2f}   {len(y_true):<7}\n"
        report += f" weighted avg   {macro_precision:<9.2f}   {macro_recall:<6.2f}   {macro_f1_score:<8.2f}   {len(y_true):<7}\n"

        return report
    def evaluate(
        self,
        model,
        split_name,
        seq_len,
        eval_dataloader: DataLoader,
        tracker: Tracker = None,
        cur_epoch: int = None,
        cur_k_fold: int = None,
        device=DEFAULT_DEVICE,
    ) -> Tuple[Dict[str, float], DataFrame]: #TODO terrible typing, this needs to be an object or something
        """Method to evaluate the model on the validation set

        :param eval_dataloader: Dataloader to get batches of Shape:(batch_size, 3) -> Misleading name -> evaluation_dataloader, dataloader.
:param tracker: Tracker: a tracker class to log the loss and metrics
        :param cur_epoch: int: current epoch number before training

        returns: dictionary of metrics and a list of mislabled datapoints
        """
        
        print(f"\n\n---- EVALUATION STARTING ({split_name}) ----\n\n")
        y_pred_all, y_true_all, x_tokens_all = [], [], []
        loss_all = 0

        # loss_fn = nn.CrossEntropyLoss(reduction="mean")
        eval_dataloader_len = len(eval_dataloader)

        with torch.no_grad():
            model.eval()

            for batch in eval_dataloader:
                ids = batch['input_ids'].to(device, dtype = torch.long)
                mask = batch['attention_mask'].to(device, dtype = torch.long)
                labels = batch['labels'].to(device, dtype = torch.long)
                
                output = model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = output.loss
                eval_logits = output.logits
                
                loss_all += loss.item()

                flattened_targets = labels.view(-1) # shape (batch_size * seq_len)
                flattened_tokens = ids.view(-1)
                active_logits = eval_logits.view(-1, model.num_labels) 
                flattened_predictions = torch.argmax(active_logits, axis=1)

                # iterate over messages in batch to identify message-level errors
                for idx in range(0, len(flattened_targets), seq_len):

                    active_accuracy = torch.zeros(len(flattened_targets), dtype=torch.bool, device=device)
                    active_accuracy[idx:idx + seq_len] = flattened_targets[idx:idx + seq_len] != -100
                
                    labels = torch.masked_select(flattened_targets, active_accuracy)
                    predictions = torch.masked_select(flattened_predictions, active_accuracy)
                    tokens = torch.masked_select(flattened_tokens, active_accuracy)
                    
                    y_true_all.append(labels.tolist())
                    y_pred_all.append(predictions.tolist())
                    x_tokens_all.append(tokens.tolist())
            
        metrics = self.metrics_for_all_batches(
            eval_dataloader_len,
            split_name,
            loss_all,
            [pred for message_pred in y_pred_all for pred in message_pred],
            [true for message_true in y_true_all for true in message_true],
        )
        self.log_to_tracker(
            tracker,
            metrics,
            cur_epoch,
            verbose=True,
        )  # save metrics to tracker
        errors = self.identify_errors(
            y_pred_all,
            y_true_all,
            x_tokens_all,
        )
        
        update_history(self.history, cur_k_fold, cur_epoch, metrics)
        
        print(F"---- EVALUATION FINISED ({split_name})----\n")
        return metrics, errors

    def metrics_for_all_batches(
        self,
        eval_dataloader_len,
        split_name,
        loss_all,
        y_pred_all,
        y_true_all,
    ):

        print(self.classification_report_custom(y_true_all, y_pred_all))

        epoch_loss = loss_all / eval_dataloader_len
        epoch_accuracy = balanced_accuracy_score(y_true_all, y_pred_all)
        
        metrics = {
                f"Loss/{split_name}_loss": epoch_loss,
                f"Balanced Accuracy/{split_name}_accuracy": epoch_accuracy,
            }
        return metrics

    def identify_errors(
        self,
        y_pred_all,
        y_true_all,
        x_tokens_all,
    ) -> DataFrame:
        
        errors_data = {
            'tokens': [],
            'y_pred': [],
            'y_true': []
        }

        for y_pred, y_true, x_tokens in zip(y_pred_all, y_true_all, x_tokens_all):
            if y_pred != y_true:
                errors_data['tokens'].append(self.tokenizer.convert_ids_to_tokens(x_tokens) )
                errors_data['y_pred'].append([self.ids_to_labels[token_pred] for token_pred in y_pred])
                errors_data['y_true'].append([self.ids_to_labels[token_true] for token_true in y_true])
        
        return pd.DataFrame(errors_data)
        
    def log_to_tracker(
        self,
        tracker: Tracker,
        metrics: Dict[str, float],
        cur_epoch: int = None,
        verbose=False,
    ) -> None:
        """Method to save metrics to the given tracker. Could extend this to have multiple trackers

        :param tracker: tracker class
        :param metrics: Dict[str, float]: dictionary of metrics to log
        :param cur_epoch: int: report the epoch (step) for the given metrics
        :param verbose: Bool: if True emits the metrics to cmd
        Returns:

        """
        print(f"Logging metrics to tracker:\n{metrics}")
        
        for k, v in metrics.items():
            if verbose:
                print(f"{k}: {v:.2f}")
        tracker.log_metrics(metrics, step=cur_epoch)

    