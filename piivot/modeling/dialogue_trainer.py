import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler
from pathlib import Path
from typing import List
from pandas import DataFrame
import wandb
from tqdm.autonotebook import tqdm, trange

from .Tracker import WandbTracker
from .DialogueEvaluator import DialogueEvaluator
from piivot.utils.immutable import global_immutable
from piivot.utils.console import console
from piivot.utils import ExperimentConfig
from .model_factory import create_model
from .optimizer_factory import create_optimizer

repo_path = Path(__file__).resolve().parents[0]
DEFAULT_DEVICE = "cpu"
SAVE_MODEL_EVERY = 30

class DialogueTrainer:

    def __init__(
        self,
        tracker: WandbTracker,
        exp_name: str,
        experiment_config: ExperimentConfig,
        lr_scheduler: lr_scheduler = None,
        grad_clipping_max_norm: int = 10,
    ) -> None:
        """
        Class to train a BERT model
        :param model: The model to train, drived from interface BaseModel
        :param optimizer: The optimizer to use, should be already instantiated
        :param tracker: The tracker to use for logging
        :param lr_scheduler: The learning rate scheduler to use, default None
        :param epoch: The epoch to start training from, default 0 unless resume logic is initiated.
        """
        self.model = None
        self.experiment_config = experiment_config
        self.lr_scheduler = lr_scheduler
        self._epoch = 0
        self.grad_clipping_max_norm = grad_clipping_max_norm
        self.losses_per_n_steps: List[int] = []
        self.losses_per_epochs: List[int] = []
        self.acc_per_n_steps: List[int] = []
        self.acc_per_epochs: List[int] = []
        self.tracker = tracker
        self.exp_name = exp_name

    def initialize_model(self):
        console.rule(
            f"Initializing the {self.experiment_config.model.params.name} model."
        )
        
        self.model= create_model(self.experiment_config.model.params.name, 
                                 self.experiment_config.model.params.from_pretrained, 
                                 **self.experiment_config.model.pretrained_params.model_dump())
        
        self.model.to(global_immutable.DEVICE)

        self.optimizer = create_optimizer(self.experiment_config.trainer.optimizer.name,
                                          self.model.parameters(),
                                          **self.experiment_config.trainer.optimizer.params.model_dump())

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloaders: List[DataLoader],
        evaluator: DialogueEvaluator,
        num_epochs: int = 1,
        use_tqdm: bool = True,
        val_frequency: int = 1,
        device=DEFAULT_DEVICE,
    ) -> DataFrame:
        """
        Train the model for a specified number of epochs

        :param dataset: The dataset to train on
        :param negative_sampler: The negative sampler to use
        :param evaluator: The evaluator to use
        :param num_epochs: The number of epochs to train for
        :param use_tqdm: Whether to use tqdm for progress bar
        :param val_frequency: The frequency of evaluation
        :param device: The device to use for computation
        """

        # self.initialize_model()
        self.initialize_model()

        errors = None

        if use_tqdm:
            epochs = trange(
                self._epoch + 1,
                1 + num_epochs,
                desc="Epochs",
                initial=self._epoch,
                total=num_epochs,
            )
        else:
            epochs = range(self._epoch + 1, 1 + num_epochs)

        for epoch in epochs:
            
            if use_tqdm:
                # progress bar for batches
                batches = tqdm(train_dataloader, desc="Batches")
            else:
                batches = train_dataloader  # no progress bar
                
            current_epoch_loss, tr_balanced_accuracy = 0, 0
            current_batch_loss = []
            nb_tr_examples, nb_tr_steps = 0, 0
            tr_preds, tr_labels = [], []

            self.model.train()
            for idx, batch in enumerate(batches):
                ids = batch['input_ids'].to(device, dtype = torch.long)
                mask = batch['attention_mask'].to(device, dtype = torch.long)
                labels = batch['labels'].to(device, dtype = torch.long)

                outputs = self.model(input_ids=ids, attention_mask=mask, labels=labels)
                loss, tr_logits = outputs.loss, outputs.logits
                
                current_epoch_loss += loss.item()

                nb_tr_steps += 1
                nb_tr_examples += labels.size(0)
                
                if idx % 100==0:
                    loss_step = current_epoch_loss/nb_tr_steps

                    self.losses_per_n_steps.append(loss_step)
                    current_batch_loss.append(loss_step)

                    if use_tqdm:
                        # sets the value of "prev_loss" to the second-to-last loss value if there are at least two losses recorded, otherwise set to None.
                        batches.set_postfix(
                            {
                                "Train loss": self.losses_per_n_steps[-1],
                                "prev_loss": (
                                    self.losses_per_n_steps[-2]
                                    if len(self.losses_per_n_steps) > 1
                                    else None
                                ),
                            }
                        )
                
                flattened_targets = labels.view(-1) # shape (batch_size * seq_len)
                active_logits = tr_logits.view(-1, self.model.num_labels) 
                flattened_predictions = torch.argmax(active_logits, axis=1)
                
                # only compute accuracy at active labels
                active_accuracy = labels.view(-1) != -100 
                
                labels = torch.masked_select(flattened_targets, active_accuracy)
                predictions = torch.masked_select(flattened_predictions, active_accuracy)
                
                tr_labels.extend(labels)
                tr_preds.extend(predictions)

                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(), max_norm=self.grad_clipping_max_norm
                )
                
                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # clear batch data
            del batches

            epoch_loss = current_epoch_loss / len(train_dataloader)
            
            print(f"Logging step and epoch-level training loss to tracker for epoch {epoch}.")
            temp_table = [[loss, step] for (loss, step) in zip(current_batch_loss, list(range(0, len(current_batch_loss) * 100, 100)))]
            table = wandb.Table(data=temp_table, columns=["loss", "step"])
            self.tracker.log_metrics(
                {
                    f"Step Loss/{epoch}_train_loss": wandb.plot.line(
                        table, "step", "loss", title=f"Epoch {epoch} loss by step"
                    )
                }
            )

            self.losses_per_epochs.append(epoch_loss)
            self.tracker.log_metrics({"Loss/train_loss": epoch_loss}, step=epoch)

            if use_tqdm:
                # sets the value of "prev_loss" to the second-to-last loss value if there are at least two losses recorded, otherwise set to None.
                epochs.set_postfix(
                    {
                        "Train loss": self.losses_per_epochs[-1],
                        "prev_loss": (
                            self.losses_per_epochs[-2]
                            if len(self.losses_per_epochs) > 1
                            else None
                        ),
                    }
                )

            # the last successful finished epoch
            self._epoch = epoch

            data_set_errors = []

            # --- validataion
            for i, val_dataloader in enumerate(val_dataloaders):
                if (epoch + 1) % val_frequency == 0:
                    (metrics, errors) = evaluator.evaluate(
                            self.model,
                            f"val_{i}",
                            self.experiment_config.model.params.max_len,
                            val_dataloader,
                            tracker=self.tracker,
                            cur_epoch=epoch,
                            device=device,
                        )
                    data_set_errors.append(errors)
                
        # return the last set of errors from each valid set
        return data_set_errors