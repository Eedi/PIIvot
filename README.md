# PIIvot
PIIvot utilizes fine-tuned named entity recognition to identify and anonymize entities that commonly contain personally identifiable information with contextually accurate surrogates


## Table of Contents

1. [Overview] 📖
2. [Setup]🧑‍🔬
    - [Prerequisites] 📋
    - [Installation] ⏬
        - [Windows]
        - [MacOS]
        - [Linux]
3. [Run] 🏃
4. [Backend Model Training] 🏋️‍♂️

# Overview <a id="overview"></a> 📖

PIIvot is a library for the detection of potential PII and its anonymization in data workflows. It utilizes realistic surrogates to obfuscate names, schools, phone numbers, and locations.

For a closer look, you can explore the core module's primary code located at `piivot/engine/analyzer.py` and `piivot/engine/anonymizer.py` where you'll find the implementation of the main functional classes: `Analyzer`and `Anonymizer`.

- `Analyzer`
    
    TBD
    
- `Anonymizer`
    
    TBD
    

# 2. Setup <a id="setup"></a> 🧑‍🔬

## 2.1 Prerequisites <a id="prereq"></a> 📋

- Python 🐍
    
    `Python version 3.12^` is required
    
- OpenAI API Key
    
    To anonymize data with the Anonymizer you’ll need an active OpenAI API key.

- (Optional) Huggingface Account
  
    To use Eedi's finetuned models, you may need to request access for your Huggingface account. Once you've been granted access to the hub, use `huggingface-cli login` with a User Access Token that has 'Read access to contents of all public gated repos you can access'.
## 2.2 Installation <a id="installation"></a> ⏬

### Poetry

`piivot` uses `poetry` **(do not use `pip` or `conda`)**.
To create the environment:

- Windows <a id="windows"></a>
    
    ```
    poetry env use 3.12
    poetry config virtualenvs.in-project true
    poetry install
    
    # to activate the env
    poetry shell
    
    ```
    
- MacOS <a id="mac"></a>
    
    ```bash
    poetry env use 3.12
    poetry config virtualenvs.in-project true
    
    poetry config --local installer.no-binary pyodbc
    
    poetry install
    
    # to activate the env
    poetry shell
    
    ```
    
- Linux/ Eedi VM <a id="linux"></a>
    
    ```bash
    export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring
    
    poetry env use 3.12
    poetry config virtualenvs.in-project true
    
    poetry install
    
    # to activate the env
    poetry shell
    
    ```
    
    ❗ **NOTE**:
    if you get the following error
    
    ```
    This error originates from the build backend, and is likely not a problem with poetry but with multidict (6.0.4) not supporting PEP 517 builds. You can verify this by running 'pip wheel --use-pep517 "multidict (==6.0.4)"'.
    
    ```
    
    Run:
    
    ```
    poetry shell
    pip install --upgrade pip
    MULTIDICT_NO_EXTENSIONS=1 pip install multidict
    poetry add inflect
    poetry add pyodbc
    
    # if package are not reinstalled then run:
    poetry update
    
    ```
    

# 3. Run 🏃<a id="run"></a>

Example run for analyze and anonymize functions:

```python
from piivot.engine import Analyzer, Anonymizer, LabelAnonymizationManager

from openai import OpenAI
import pandas as pd

data = [
    {"message": "Hi, I'm John and I live in New York."},
    {"message": "Hello, my name is Jane and I live in Los Angeles."},
    {"message": "Hey, I'm Alice from Chicago."},
    {"message": "Greetings, I'm Bob and I'm based in Seattle."},
    {"message": "Hi, I'm Carol and I reside in Miami."},
]
df = pd.DataFrame(data, columns=["message"])

analyzer = Analyzer("Eedi/DeBERTa-PIIvot-NER-IO")
df = analyzer.analyze(df, data_columns=['message'])


# gpt_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
label_anon_manager = LabelAnonymizationManager()
anonymizer = Anonymizer(label_anon_manager)

anonymized_df = anonymizer.anonymize(df, data_columns=['message'], label_columns=['message_labels'])
print(anonymized_df.head())
```

If running locally in a Jupyter Notebook, you can import the PIIvot Repo with the following code.

```python
import os
import sys

module_path = os.path.abspath(os.path.join([[Path to PIIvot Repo]]))
if module_path not in sys.path:
    sys.path.append(module_path)
```

## Using `PIIvot` in other repositories <a id=otherRepo></a>

Previously we would have installed the package globally using `pip install -e .`, using `poetry` you simply add a dependency to the local package.

1. Clone the repository:
    
    ```bash
    git clone git@github.com:zentavious/PIIvot.git
    
    ```
    
2. In your other repository, add the following to the `pyproject.toml`:
    
    ```python
    piivot = {path = <path-to-piivot>, develop=true}
    
    ```
    
    Example:
    `piivot` was cloned in the parent directory of the current project.
    
    ```bash
    piivot = {path = "../piivot", develop = true}
    
    ```
    
    The develop flag should mean that your installation will be automatically updated when `piivot` is editted.
    
3. You can now import this package:
    
    ```python
    from piivot.engine import Analyzer, Anonmizer
    
    ```
    
5. If you then update this package it should update automatically (if `develop = true`). If this does not happen you should be able to just run `poetry update piivot` but you may need to reinstall your poetry environment.

# 4. Backend Model Training 🏋️‍♂️<a id="experimentation"></a>

PIIvot has built in support for a variety of model fine-tuning and experimentation use cases. This code is highly tailored to the Eedi Tutor/Student Dialogue dataset and further work is need to generalize the experimentation pipeline to any dataset.

## Prerequisites <a id="prereq"></a> 📋

- A labeled .csv

    Use orchex to generate a ground truth tutor student dialogues extract.

## ./learn.py

Example experiment run using deberta_base_experiment

```bash
poetry run python ./learn.py --exp_folder ./experiment_configs/deberta_base_experiment --data_filepath [[data_filepath]]

```


