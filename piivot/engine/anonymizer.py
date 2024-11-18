import ast
import datetime
import itertools
import re
import warnings
from typing import Dict, List, Tuple

import pandas as pd
import tiktoken
from tqdm.autonotebook import tqdm


def match_casing(reference_string, target_string):
    """Matches the casing (capitalization) of reference_string to target_string by words. For uneven length words, an average casing is applied.

    Args:
        reference_string (str): The string whose casing will be matched.
        target_string (str): The string whose casing will be applied to reference_string.

    Returns:
        str: The reference_string with casing matched to target_string.
    """
    ref_words = reference_string.split()
    target_words = target_string.split()
    max_length = max(len(word) for word in target_words)
    matched_words = []

    # Mantain a list of tuples in the form (num_examples, num_capital) for every character index for when len(ref_words) > len(target_words)
    avg_cassing = [(0, 0)] * max_length

    # Iterate over pairs of words, using the length of the shorter list
    for ref_word, target_word in zip(
        ref_words, target_words
    ):  # TODO could do this differently categorize as first_caps, no_caps, or random
        matched_chars = []

        # Iterate over characters in both words
        for i, (ref_char, target_char) in enumerate(
            itertools.zip_longest(ref_word, target_word, fillvalue=None)
        ):
            if target_char and target_char.isupper():
                avg_cassing[i] = (avg_cassing[i][0] + 1, avg_cassing[i][1] + 1)
                if ref_char:
                    matched_chars.append(ref_char.upper())
            elif target_char:
                avg_cassing[i] = (avg_cassing[i][0] + 1, avg_cassing[i][1])
                if ref_char:
                    matched_chars.append(ref_char.lower())
            else:
                matched_chars.append(ref_char.lower())

        matched_word = "".join(matched_chars)
        matched_words.append(matched_word)

    if len(ref_words) > len(target_words):
        for ref_word in ref_words[len(target_words) :]:
            matched_chars = []

            for ref_char, target_char in zip(ref_word, avg_cassing):
                if (
                    target_char[1] / target_char[0] >= 0.5
                ):  # half or more letters in this position were uppercase
                    matched_chars.append(ref_char.upper())
                else:
                    matched_chars.append(ref_char.lower())

            if len(ref_word) > len(avg_cassing):
                matched_chars.append(ref_word[len(avg_cassing) :].lower())

            matched_word = "".join(matched_chars)
            matched_words.append(matched_word)

    matched_string = " ".join(matched_words)

    return matched_string


def is_tutor(isTutor):
    match isTutor:
        case 1:
            return "TUTOR"
        case 0:
            return "STUDENT"


gpt_general_prompt = "Given a multiple labeled lists strings in the form [[LABEL_TYPE]]:[list of strings], find surrogate replacements for each that anonymize the original string but are not obviously anonymized. It should be difficult to guess what the original string was based on the anonymized surrogate. Favor using words not present in the data. When two strings to be anonymized have similar spellings or contain similarly spelled words, ensure both replacements have similar spellings to each other. Use chat history under [[CHAT HISTORY]] to ensure each replacement makes logical sense in the context of all messages in the chat history. Your response should be a dictionary in the form {[original text]: [anonymized text]}. The dictionary should be parsable using ast.literal_eval() meaning all nested single and double quotes should be escaped with \\. [original text] should always be lowercase, even if the text is cased differently in the chat history"
gpt_missing_reprompt = "Your prior response is missing replacements for some of the required text. Make sure to create anonymized replacements for the exact spellings of all strings. Use prior responses to generate similarly spelled replacements for strings that were originally similarly spelled. Generate a dictionary in the form {[original_string]: [anonymized_string]} for the following list of strings. The dictionary should be parsable using ast.literal_eval() meaning all nested single and double quotes should be escaped with \\."
gpt_feedback_reprompt = "Your prior response doesn't meet all replacement criteria. Generate a dictionary in the form {[original_string]: [new_anonymized_string]} where the newly anonymized string follows the feedback provided after [[FEEDBACK]] for each of the following strings. The dictionary should be parsable using ast.literal_eval() meaning all nested single and double quotes should be escaped with \\."


def extract_dictionary(response):
    """Extracts a dictionary from the given response string.

    Parameters:
    - response (str): The response string potentially containing a dictionary.

    Returns:
    - dict: The extracted dictionary.
    """
    match = re.search(r"({.*?})", response, re.DOTALL)
    if match:
        dict_str = match.group(1)
        try:
            extracted_dict = ast.literal_eval(dict_str)
            if isinstance(extracted_dict, dict):
                return extracted_dict
        except (SyntaxError, ValueError) as e:
            return {"[[Error]]": f"{e}"}
    return dict()


class Anonymizer:
    def __init__(
        self,
        label_manager,
        client=None,
        assistant_general_prompt=gpt_general_prompt,
        temperature=0.2,
        max_tokens=3000,  # Not include dialogues above the limit ->
        frequency_penalty=0.0,
        gpt_model="gpt-3.5-turbo",
        missing_reprompt=gpt_missing_reprompt,
        feedback_reprompt=gpt_feedback_reprompt,
        reprompt_additional_tokens=896,
    ) -> None:
        """Initialize the Anonymizer class with specified parameters for anonymizing future data. If no client is specified,
        the class will not anonymize data, but will count tokens used for anonymization calls.

        Args:
            label_manager: An object responsible for managing labels.
            client (Optional): The OpenAI client interface for interacting with the Chat-GPT. Default is None.
            assistant_general_prompt (str): The general prompt used for the GPT assistant. Default behavior is provided in class.
            temperature (float): The sampling temperature for the GPT model, affecting the randomness of the output. Default is 0.2.
            max_tokens (int): The maximum number of tokens to include in the model's output. Default is 3000.
            frequency_penalty (float): The penalty for token frequency in the GPT model's output, controlling the likelihood of repeating tokens. Default is 0.0.
            gpt_model (str): The GPT model name. Default is "gpt-3.5-turbo".
            missing_reprompt (str): The prompt to use when the model's response is missing information. Default behavior is provided in class.
            feedback_reprompt (str): The prompt to use when additional feedback is needed from the model. Default behavior is provided in class.
            reprompt_additional_tokens (int): The additional tokens to allocate for reprompting per call. Default is 896.
        """
        self.label_manager = label_manager
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.assistant_general_prompt = assistant_general_prompt
        self.gpt_model = gpt_model
        self.client = client
        self.missing_reprompt = missing_reprompt
        self.feedback_reprompt = feedback_reprompt
        self.reprompt_additional_tokens = reprompt_additional_tokens

        self.logs = []
        self.anonymize_call_count = 0
        self.debug_logs = True

        if not self.client:
            self.dev_tokenizer = enc = tiktoken.encoding_for_model(
                self.gpt_model
            )  # GPT2Tokenizer.from_pretrained("gpt2") # TODO calcualte how "off" this estimate is. Try tiktoken
            self.token_count = []
        else:
            print(
                "GPTAnonymizer is Live. Subsequent calls to anonymize will incure a cost on this client."
            )

    def log(self, message):
        if self.debug_logs:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            self.logs[-1] += f"[{current_time}]\n{message}\n\n"

    def initialize_new_log(self, identifier):
        if self.debug_logs:
            self.logs.append("")
            self.log(identifier)

    def print_logs(self):
        print("\n\n\n".join(self.logs))

    def write_logs_to_file(self, filename):
        with open(filename, "w") as file:
            file.write("\n\n\n".join(self.logs))

    def count_tokens(self, prompt):
        tokens = self.dev_tokenizer.encode(prompt)
        num_tokens = len(tokens)

        return num_tokens

    def extract_labeled_substrings(self, data, labels):
        labeled_substrings = {}
        for start, end, label in labels:
            substring = data[start:end].lower()

            if label.upper() in labeled_substrings:
                labeled_substrings[label.upper()].add(substring)
            else:
                labeled_substrings[label.upper()] = {substring}

        return labeled_substrings

    def extract_group_labeled_substrings(self, group, data_column, label_column):
        group_labeled_substrings = {}
        labeled_substrings_list = group.apply(
            lambda row: self.extract_labeled_substrings(
                row[data_column], row[label_column]
            ),
            axis=1,
        ).tolist()
        for labeled_substrings in labeled_substrings_list:
            for key, value in labeled_substrings.items():
                if key in group_labeled_substrings:
                    group_labeled_substrings[key].update(value)
                else:
                    group_labeled_substrings[key] = value.copy()

        return group_labeled_substrings

    def anonymize_data(
        self,
        data: str,
        labels: List[Tuple[int, int, str]],
        new_mappings: Dict[str, str],
    ) -> Tuple[str, List[Tuple[int, int, str]], bool]:
        """Anonymize the given data by applying the specified labels and new mappings.

        Args:
            data (str): The data to be anonymized.
            labels (List[Tuple[int, int, str]]): A list of tuples where each tuple contains the start index,
                end index, and label name for the portions of data to be anonymized.
            new_mappings (dict[str, str]): A dictionary mapping original values to their anonymized counterparts.

        Returns:
        tuple (Tuple): A tuple containing:
            - anonymized_data (str): The anonymized data with the specified labels applied and mappings implemented.
            - new_labels (List[Tuple[int, int, str]]): A list of the new labeled spans of anonymized surrogates in the form (start_index, end_index, label_name).
        """
        anonymized_data = data
        new_labels = sorted(labels, key=lambda x: x[1])
        if self.client:
            for i in range(len(new_labels)):
                label = new_labels[i]
                label_name = label[2].upper()

                original_text = anonymized_data[label[0] : label[1]]

                if label_name in self.label_manager.label_names:
                    new_text = match_casing(
                        new_mappings[original_text.lower()], original_text
                    )
                else:
                    # update down index label indicies by the new length of the
                    new_text = f"[[{label_name}]]"

                anonymized_data = (
                    anonymized_data[: label[0]] + new_text + anonymized_data[label[1] :]
                )
                offset = len(new_text) - len(original_text)
                new_labels[i] = (label[0], label[1] + offset, label[2])
                for j in range(i + 1, len(new_labels)):
                    new_labels[j] = (
                        new_labels[j][0] + offset,
                        new_labels[j][1] + offset,
                        new_labels[j][2],
                    )

        return anonymized_data, new_labels, False

    def generate_missing_reprompt(self, new_mappings, all_gpt_targets):
        reprompt = ""
        missing_mappings = [
            element for element in all_gpt_targets if element not in new_mappings
        ]
        if missing_mappings:
            reprompt = f"{self.missing_reprompt} {missing_mappings}"

        return reprompt, missing_mappings

    def generate_feedback_reprompt(self, new_mappings, labeled_substrings, gpt_targets):
        reprompt = ""
        gpt_target_list = []
        for label in labeled_substrings:
            if label in self.label_manager.label_names:
                substrings = labeled_substrings[label]
                for substring in substrings:
                    if substring in gpt_targets:
                        feedback = self.label_manager.get_anoymization_feedback(
                            label, substring, new_mappings[substring]
                        )
                        if feedback:
                            reprompt = f"{reprompt} {feedback}"
                            gpt_target_list.append(substring)

        if gpt_target_list:
            reprompt = (
                f"{self.feedback_reprompt} {gpt_target_list}\n[[FEEDBACK]]{reprompt}"
            )

        return reprompt, gpt_target_list

    def generate_new_mappings(
        self,
        prompt_messages: List[Dict[str, str]],
        labeled_substrings: Dict[str, str],
        assert_targets: List[str] = [],
        additional_max_tokens: int = 0,
        debug_content: str = "",
    ) -> Tuple[Dict[str, str], bool]:
        """Generate new mappings for labeled substrings based on provided prompt messages.

        Args:
            prompt_messages (List[dict[str, str]]): A list of messages to be used as prompts for gpt_client.
            labeled_substrings (dict[str, str]): A list of labeled substrings that need to be mapped.
            assert_targets (List[str], optional): A list of target assertions to validate are present in the generated mappings. Default is an empty list.
            additional_max_tokens (int, optional): Additional tokens to allocate for the generation process. Default is 0.
            debug_content (str, optional): A debug prompt response. If provided, no prompt will be sent to gpt_client. Default is an empty string.

        Returns:
            Dict[str, str]: A dictionary containing the new mappings generated from the prompt to gpt_client.
            Bool: A flag denoting if there was an error generating new mappings.
        """
        if not debug_content:
            chat_completion = self.client.chat.completions.create(
                messages=prompt_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens + additional_max_tokens,
                frequency_penalty=self.frequency_penalty,
                model=self.gpt_model,
            )
            self.log(f"GPT Response:\n{chat_completion.choices[0].message.content}")

            new_mappings = extract_dictionary(
                chat_completion.choices[0].message.content
            )
        else:
            self.log(debug_content)
            new_mappings = extract_dictionary(debug_content)
        new_mappings = {key.lower(): value for key, value in new_mappings.items()}

        if assert_targets:
            try:
                assert all(
                    target in new_mappings for target in assert_targets
                ), f"Elements still missing from {assert_targets} in response mapping {new_mappings}"
            except AssertionError as e:
                return {"[[Error]]": e}, True

        labeled_mappings = {}
        for label in labeled_substrings:
            labeled_mappings[label] = {}
            for labeled_substing in labeled_substrings[label]:
                if labeled_substing in new_mappings:
                    labeled_mappings[label][labeled_substing] = new_mappings[
                        labeled_substing
                    ]
            pass
        self.label_manager.update_prior_label_mappings(labeled_mappings)

        return new_mappings, False

    def anonymize_group(
        self,
        group,
        data_column: str,
        label_column: str,
        group_name: str = "CHAT HISTORY",
        dialogue_identifier_column: str = "IsTutor",
        dialogue_identifier_func=is_tutor,
    ):
        """Anonymize the specified data column within a given group by prompting and reprompting gpt_client to generate surrogate replacements.

        Args:
            group (pd.DataFrame): The DataFrame group to be anonymized.
            data_column (str): The name of the column containing the data to be anonymized.
            label_column (str): The name of the column where the labels are be stored.
            group_name (str, optional): The name to append before the grouped data prompt. Default is "CHAT HISTORY".
            dialogue_identifier_column (str, optional): The name of the column used to identify the "who" of this data_column. Default is 'IsTutor'.
            dialogue_identifier_func (function, optional): A function to apply to the values in dialogue_identifier_column to apply string values to rows of this group data prompt. Default is is_tutor.

        Returns:
            pd.DataFrame: A DataFrame containing the anonymized data and new updated label spans.
        """
        labeled_substrings = self.extract_group_labeled_substrings(
            group, data_column, label_column
        )

        assistant_group_prompt = self.assistant_general_prompt
        group_prompt = ""
        all_gpt_targets = []
        for label in labeled_substrings:
            gpt_target_list = []

            if label in self.label_manager.label_names:
                gpt_target_list = labeled_substrings[label]

            if gpt_target_list:
                all_gpt_targets.extend(gpt_target_list)
                assistant_group_prompt = f"{assistant_group_prompt}\n{self.label_manager.assistant_label_prompts[label]}"
                group_prompt = f"{group_prompt} [[{label}]]:{gpt_target_list}"  # -> "Hi Tom, how are you? I'm good cindy! Just raining here in new york" -> [[NAME]] [tom, cindy] [[LOCATION_ADDRESS]] [new york]

        new_mappings = {}
        self.label_manager.clear_prior_label_mappings()

        if all_gpt_targets:
            # append [[TUTOR]] or [[STUDENT]] to each message based on the speaker
            if dialogue_identifier_column:
                grouped_data = "\n".join(
                    [
                        f"[[{dialogue_identifier_func(row[dialogue_identifier_column])}]] {row[data_column]}"
                        for _, row in group.iterrows()
                    ]
                )
            else:
                grouped_data = "\n".join(
                    [f"{row[data_column]}" for _, row in group.iterrows()]
                )

            group_prompt = f"{group_prompt}\n[[{group_name}]]\n{grouped_data}"

            messages = [
                {"role": "system", "content": assistant_group_prompt},
                {"role": "user", "content": group_prompt},
            ]

            self.log(f"Sending prompt:\n{messages}")

            if self.client:
                new_mappings, _ = self.generate_new_mappings(
                    messages, labeled_substrings
                )

                missing_reprompt, missing_targets = self.generate_missing_reprompt(
                    new_mappings, all_gpt_targets
                )

                if missing_reprompt:
                    messages = [
                        {"role": "system", "content": assistant_group_prompt},
                        {"role": "user", "content": group_prompt},
                        {"role": "assistant", "content": str(new_mappings)},
                        {"role": "user", "content": missing_reprompt},
                    ]
                    self.log(f"Sending prompt:\n{messages}")

                    added_mappings, error = self.generate_new_mappings(
                        messages,
                        labeled_substrings,
                        assert_targets=missing_targets,
                        additional_max_tokens=self.reprompt_additional_tokens,
                    )

                    if error:
                        group[f"anonymized_{data_column}"] = group[data_column]
                        group[f"new_{label_column}"] = group.apply(
                            lambda _: new_mappings, axis=1
                        )
                        group["has_error"] = True

                        return group

                    new_mappings = new_mappings | added_mappings

                feedback_reprompt, reprompt_targets = self.generate_feedback_reprompt(
                    new_mappings, labeled_substrings, all_gpt_targets
                )

                if feedback_reprompt:
                    messages = [
                        {"role": "system", "content": assistant_group_prompt},
                        {"role": "user", "content": group_prompt},
                        {"role": "assistant", "content": str(new_mappings)},
                        {"role": "user", "content": feedback_reprompt},
                    ]
                    self.log(f"Sending prompt:\n{messages}")

                    added_mappings, error = self.generate_new_mappings(
                        messages,
                        labeled_substrings,
                        assert_targets=reprompt_targets,
                        additional_max_tokens=self.reprompt_additional_tokens,
                    )

                    if error:
                        group[f"anonymized_{data_column}"] = group[data_column]
                        group[f"new_{label_column}"] = group.apply(
                            lambda _: new_mappings, axis=1
                        )
                        group["has_error"] = True

                        return group

                    new_mappings = new_mappings | added_mappings

            else:
                self.token_count.append(
                    self.count_tokens(assistant_group_prompt)
                    + self.count_tokens(group_prompt)
                )

        # group[[f'anonymized_{data_column}', f'new_{label_column}']] = group.apply(lambda row: self.anonymize_data(row[data_column], row[label_column], self.label_manager.get_prior_label_mappings("NAME") | new_mappings), axis=1, result_type ='expand')
        group[
            [f"anonymized_{data_column}", f"new_{label_column}", "has_error"]
        ] = group.apply(
            lambda row: self.anonymize_data(
                row[data_column], row[label_column], new_mappings
            ),
            axis=1,
            result_type="expand",
        )

        return group

    def anonymize(
        self,
        df: pd.DataFrame,
        data_columns: List[str],
        label_columns: List[str],
        context_groups: List[str] = None,
        identifier_column: str = None,
        debug_logs: bool = True,
        use_tqdm: bool = True,
    ) -> pd.DataFrame:
        """Anonymize specified data columns in the given DataFrame by processing labeled sensitive information.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data to be anonymized.
            data_columns (List[str]): A list of column names in the DataFrame that contain the data to be anonymized.
            label_columns (List[str]): A list of column names where the labels are stored.
            context_groups (List[str], optional): A list of column names that define the context groups for analysis. Grouped
                data will be batched in GPT calls. This behavior is recommended for cost and performance optimization. Default is None.
            identifier_column (str, optional): The name of the column containing personal identifiers for who the text data
                originates from if in a dialogue format. Default is None.
            debug_logs (bool, optional): Whether to store debug logs. Use print_logs or write_logs_to_file to retrieve. Default is True.

        Returns:
            pd.DataFrame: A DataFrame containing the anonymized data and new lists of label spans in the form (start_index, end_index, label_name).
        """
        if len(data_columns) != len(label_columns):
            raise Exception(
                "Length of data_columns must equal the length of label_columns"
            )

        input_length = len(data_columns)
        for i, (data_column, label_column) in enumerate(
            zip(reversed(data_columns), reversed(label_columns))
        ):
            original_i = input_length - 1 - i
            if data_column not in df.columns or label_column not in df.columns:
                warnings.warn(
                    f"Either column '{data_column}' and/or '{label_column}' do not exist in the input dataframe. Skipping..."
                )
                data_columns.pop(original_i)
                label_columns.pop(original_i)

        self.debug_logs = debug_logs
        self.anonymize_call_count += 1

        if use_tqdm:
            zipped_data_columns = tqdm(zip(data_columns, label_columns), desc="Columns")
        else:
            zipped_data_columns = zip(data_columns, label_columns)

        for data_column, label_column in zipped_data_columns:
            self.initialize_new_log(f"{self.anonymize_call_count}_{data_column}")
            # if context_groups:
            #     df = df.groupby(context_groups).apply(lambda group: self.anonymize_group(group, data_column, label_column, dialogue_identifier_column=identifier_column)).reset_index(drop=True)
            # else:
            #     df = df.groupby(level=0).apply(lambda group: self.anonymize_group(group, data_column, label_column, dialogue_identifier_column=identifier_column)).reset_index(drop=True)

            if context_groups:
                data_grouped = df.groupby(context_groups)
            else:
                data_grouped = df.groupby(level=0)
            result = []
            if use_tqdm:
                data_groups = tqdm(
                    data_grouped, total=len(data_grouped), desc="Anonymizing groups"
                )

            for name, group in data_groups:
                result.append(
                    self.anonymize_group(
                        group,
                        data_column,
                        label_column,
                        dialogue_identifier_column=identifier_column,
                    )
                )

            df = pd.concat(result).reset_index(drop=True)
        if not self.client:
            # Issue a simple warning
            warnings.warn(
                f"GPT client not set. Columns {data_columns} were not anonymized."
            )
            print(
                f"Using a live gpt client would cost a minimum of {sum(self.token_count)} with up to ~3x tokens for subsequent reprompts."
            )

        return df.drop(columns=data_columns + label_columns)
