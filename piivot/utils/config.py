"""A module for getting and setting experiment configs."""

from datetime import datetime, timezone
from typing import Literal, Optional, Union

from pydantic import BaseModel, PositiveFloat, PositiveInt, confloat, Field

def datetime_now() -> datetime:
    return datetime.now(timezone.utc)

class BatchParamsConfig(BaseModel):
    batch_size: Union[PositiveInt, list[PositiveInt]]
    shuffle: bool
    num_workers: int
    pin_memory: bool

class DatasetParamsConfig(BaseModel):
    name:  Literal["BERTDialogue", "MultiSentenceBERTDialogue"]
    augmented_non_pii: bool
    augmented_pii: bool
    add_synthetic: bool
    exclude_labels: list[str]
    label_scheme: Literal["IOB2", "IO"]

class DatasetConfig(BaseModel):
    params: DatasetParamsConfig
    ids_to_labels: Optional[list[str]]

class InputDataConfig(BaseModel):
    processed_date: datetime = Field(default_factory=datetime_now)
    split: bool
    train_split: confloat(ge=0.0, le=1.0)  # type: ignore
    train_params: BatchParamsConfig
    valid_split: confloat(ge=0.0, le=1.0)  # type: ignore
    valid_params: BatchParamsConfig
    dataset: DatasetConfig


class ModelParamsConfig(BaseModel):
    name: Literal["BERT", "DeBERTa"]
    from_pretrained: bool
    max_len: PositiveInt


class PretrainedModelParamsConfig(BaseModel):
    pretrained_model_name_or_path: Literal["bert-base-cased", "microsoft/deberta-v3-base"]
    num_labels: Optional[PositiveInt] = None

class ModelConfig(BaseModel):
    params: ModelParamsConfig
    pretrained_params: PretrainedModelParamsConfig

class OptimizerParamsConfig(BaseModel):
    lr: Union[PositiveFloat, list[PositiveFloat]]

class OptimizerConfig(BaseModel):
    name: Literal["Adam", "AdamW"]
    params: OptimizerParamsConfig

class TrainerConfig(BaseModel):
    name: Literal["DialogueTrainer"]
    val_every: PositiveInt
    epochs: Union[int, list[int]]
    use_tqdm: bool
    grad_clipping_max_norm: PositiveInt
    optimizer: OptimizerConfig
    resume_checkpoint_path: Optional[str]

class ExperimentConfig(BaseModel):
    model: ModelConfig
    trainer: TrainerConfig
    seed: PositiveInt

class Config(BaseModel):
    input_data: InputDataConfig
    experiment: ExperimentConfig
    
class AnalyzerConfig(BaseModel):
    optimizer: OptimizerConfig
    model: ModelConfig
    checkpoint_path: str

class AnonymizerConfig(BaseModel):
    open_ai_api_key: str