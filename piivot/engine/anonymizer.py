from transformers import GPT2Tokenizer #TODO look into better token estimator
import Levenshtein
import re
import itertools
import ast

def match_casing(reference_string, target_string):
    """
    Matches the casing (capitalization) of reference_string to target_string by words.

    Parameters:
    - reference_string (str): The string whose casing will be matched.
    - target_string (str): The string whose casing will be applied to reference_string.

    Returns:
    - str: The reference_string with casing matched to target_string.
    """
    
    ref_words = reference_string.split()
    target_words = target_string.split()
    max_length = max(len(word) for word in target_words)
    matched_words = []

    # Mantain a list of tuples in the form (num_examples, num_capital) for every character index for when len(ref_words) > len(target_words)
    avg_cassing = [(0, 0)] * max_length
    
    # Iterate over pairs of words, using the length of the shorter list
    for ref_word, target_word in zip(ref_words, target_words): # TODO could do this differently categorize as first_caps, no_caps, or random
        matched_chars = []
        
        # Iterate over characters in both words
        for i, (ref_char, target_char) in enumerate(itertools.zip_longest(ref_word, target_word, fillvalue=None)):
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
            
        matched_word = ''.join(matched_chars)
        matched_words.append(matched_word)
        
    if (len(ref_words) > len(target_words)):
        for ref_word in ref_words[len(target_words):]:
            matched_chars = []
            
            for ref_char, target_char in zip(ref_word, avg_cassing):
                if target_char[1] / target_char[0] >= 0.5:  # half or more letters in this position were uppercase
                    matched_chars.append(ref_char.upper())
                else: 
                    matched_chars.append(ref_char.lower()) 
            
            if len(ref_word) > len(avg_cassing):
                matched_chars.append(ref_word[len(avg_cassing):].lower())
                
            matched_word = ''.join(matched_chars)
            matched_words.append(matched_word)
    
    matched_string = ' '.join(matched_words)
    
    return matched_string

def is_tutor(isTutor):
    match isTutor:
        case 1:
            return "TUTOR"
        case 0:
            return "STUDENT"

gpt_general_prompt = "Given a multiple labeled lists strings in the form [[LABEL_TYPE]]:[list of strings], find replacements for each that anonymize the original string but are not obviously anonymized. Favor using words not present in the data. When two strings to be anonymized are similar or contain similar words, ensure their replacements are also similar. Use chat history under [[CHAT HISTORY]] for added context for each replacement. Your response should be a dictionary in the form {[original text]: [anonymized text]}. [original text] should always be lowercase, even if the text is cased differently in the chat history"
gpt_missing_reprompt = "Your prior response is missing replacements for some of the required text. Make sure to create anonymized replacements for the exact spellings of all strings. Use prior responses to generate similarly spelled replacements for strings that were originally similarly spelled. Generate a dictionary in the form {[original_string]: [anonymized_string]} for the following list of strings."
gpt_feedback_reprompt = "Your prior response doesn't meet all replacement criteria. Generate a dictionary in the form {[original_string]: [new_anonymized_string]} where the newly anonymized string follows the feedback provided after [[FEEDBACK]] for each of the following strings."


def contains_counting_sequence(input_string):
    """
    Checks if the input_string contains any sequences of 3 or more digits
    that count up or down (ignoring spaces).

    Parameters:
    - input_string (str): The string to check.

    Returns:
    - bool: True if the string contains such sequences, False otherwise.
    """
    sanitized_string = input_string.replace(' ', '')

    digit_sequences = re.findall(r'\d+', sanitized_string)

    for seq in digit_sequences:
        if len(seq) >= 3:
            for i in range(len(seq) - 2):
                if (seq[i+1] == str(int(seq[i]) + 1) and seq[i+2] == str(int(seq[i+1]) + 1)) or \
                   (seq[i+1] == str(int(seq[i]) - 1) and seq[i+2] == str(int(seq[i+1]) - 1)):
                    return True
    return False

def extract_dictionary(response):
    """
    Extracts a dictionary from the given response string.

    Parameters:
    - response (str): The response string potentially containing a dictionary.

    Returns:
    - dict: The extracted dictionary.
    """
    
    match = re.search(r'({.*?})', response, re.DOTALL)
    if match:
        dict_str = match.group(1)
        try:
            extracted_dict = ast.literal_eval(dict_str)
            if isinstance(extracted_dict, dict):
                return extracted_dict
        except (SyntaxError, ValueError):
            pass
    return None

class LabelAnonymizationManager():
    def name_feedback_func(self, original_text, anonymized_text):
        feedback = ""
        if Levenshtein.distance(original_text, anonymized_text) <= self.distance_threshold:
            feedback = self.append_feedback(feedback, f"{anonymized_text} is too similar to {original_text}. Ensure the anonymized replacement is a different word.")
            
        # for prior_name in self.prior_mappings["NAME"]:
        #     # Check if the Levenshtein distance between key and original_text is less or equal to 1
        #     if Levenshtein.distance(prior_name, original_text) <= self.distance_threshold and prior_name != original_text:
        #         # Get the corresponding value from prior_mappings
        #         prior_mapped_name = self.prior_mappings["NAME"][prior_name]
        #         # Check if the Levenshtein distance between anonymized_text and mapped_text is larger
        #         if Levenshtein.distance(anonymized_text, prior_mapped_name) > self.distance_threshold:
        #            feedback = self.append_feedback(feedback, f"The similarly spelled name {prior_name} was anonymized to {prior_mapped_name}. Ensure the anonymized replacement for {original_text} is the same name as {prior_mapped_name}, but matches the character differences between {original_text} and {prior_name}.")

        return feedback
    
    def phone_number_feedback_func(self, original_text, anonymized_text):
        feedback = ""
        if any(char.isalpha() for char in anonymized_text):
            feedback = self.append_feedback(feedback, f"{anonymized_text} contains alphabetic characters. Ensure the anonymized phone number doesn't have any placeholder characters.")

        if contains_counting_sequence(anonymized_text):
            feedback = self.append_feedback(feedback, f"{anonymized_text} contains ascending or descending strings of digits. Ensure the anonymized phone number doesn't look fake.")
                
        return feedback
    
    def location_address_feedback_func(self, original_text, anonymized_text):
        feedback = ""

        for orig_word in original_text.split():
            for anon_word in anonymized_text.split():
                if Levenshtein.distance(orig_word, anon_word) <= self.distance_threshold:
                    feedback = self.append_feedback(feedback, f"{anon_word} in {anonymized_text} is too similar to {orig_word} in {original_text}. Ensure the anonymized location/address doesn't contain any similar words.")

        return feedback
    
    def school_name_feedback_func(self, original_text, anonymized_text):
        feedback = ""
        school_stage_identifiers = [
            'elementary', 'primary', 'middle', 'secondary', 'academy', 'middle school',
            'middleschool', 'college', 'university', 'high school', 'highschool', 'high',
            'kindergarten', 'nursery', 'reception', 'sixth form', 'junior', 'infant'
        ]
        for school_stage_identifier in school_stage_identifiers:
            if school_stage_identifier in original_text.lower() and school_stage_identifier not in anonymized_text.lower():
                feedback = self.append_feedback(feedback, f"{original_text} refers to a {school_stage_identifier} stage school. Ensure the anonymized school name also uses the word {school_stage_identifier}.")

        return feedback
    
    def date_of_birth_feedback_func(self, original_text, anonymized_text):
        feedback = ""
        if Levenshtein.distance(original_text, anonymized_text) <= self.distance_threshold:
            feedback = self.append_feedback(feedback, f"{anonymized_text} is too similar to {original_text}. Ensure the anonymized replacement is a different day, month, and/or year.")
        return feedback
    
    def default_feedback_func(self, original_text, anonymized_text):
        return ""
    
    def append_feedback(self, feedback, new_feedback):
        if feedback:
            return f"{feedback} {new_feedback}"
        else:
            return new_feedback
    
    def __init__(self,
                #  prior_mappings = {}, #TODO remove these, and add documentation about how we could do this and why we aren't
                 distance_threshold = 1):
        
        # self.prior_mappings = prior_mappings
        self.distance_threshold = distance_threshold
        self.label_names = ['NAME','PHONE_NUMBER','LOCATION_ADDRESS','SCHOOL_NAME','DATE_OF_BIRTH']

        # if not prior_mappings:
        #     self.prior_mappings = {key: {} for key in self.label_names}

        self.feedback_func_dictionary = {
            "NAME": self.name_feedback_func,
            "PHONE_NUMBER": self.phone_number_feedback_func,
            "LOCATION_ADDRESS": self.location_address_feedback_func,
            "SCHOOL_NAME": self.school_name_feedback_func,
            "DATE_OF_BIRTH": self.date_of_birth_feedback_func,
        }
        self.assistant_label_prompts = {
            "NAME": "When anonymizing [[NAME]], preserve their gender and ethnic background.",
            "PHONE_NUMBER": "When anonymizing [[PHONE_NUMBER]], if there are any other references the digits or format of the phone number in the chat history ensure they still make sense.",
            "LOCATION_ADDRESS": "When anonymizing multiple [[LOCATION_ADDRESS]], ensure the distances between them stay consistent.",
            "SCHOOL_NAME": "When anonymizing [[SCHOOL_NAME]], ensure the resulting school has the same grade or year group.",
            "DATE_OF_BIRTH": "When anonymizing [[DATE_OF_BIRTH]], ensure its replacement has the same specificity as the original and makes sense relative to other dates in the chat history.",
        }
        # TODO add enforcement between label_names and feedback and label prompts
        # Add a class for each label -> has a name, feedback, and prompt for label -> require each peice.
        # Register each of these classes to the default behavior of the anonymization manager

    def get_anoymization_feedback(self, label_name, original_text, anonymized_text):
        
        feedback = self.feedback_func_dictionary.get(label_name, self.default_feedback_func)(original_text, anonymized_text)

        # if feedback:
        #     self.prior_mappings[label_name].pop(original_text, None)

        return feedback


    # def update_prior_label_mapping(self, label_name, additional_mappings):
    #     label_prior_mappings = self.prior_mappings.get(label_name)
    #     label_prior_mappings.update(additional_mappings)

    #     return label_prior_mappings.copy()
    
    
    # def get_prior_label_mappings(self, label_name):
    #     label_prior_mappings = self.prior_mappings.get(label_name)

    #     return label_prior_mappings.copy()

class Anonymizer():
    
    def __init__(self, 
                 label_manager,
                 client=None,
                 assistant_general_prompt=gpt_general_prompt,
                 temperature=0.2,
                 max_tokens=3000, # Not include dialogues above the limit ->
                 frequency_penalty=0.0,
                 gpt_model="gpt-3.5-turbo",
                 missing_reprompt=gpt_missing_reprompt,
                 feedback_reprompt=gpt_feedback_reprompt,
                 reprompt_additional_tokens=896) -> None:
        
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

        if not self.client:
            self.dev_tokenizer = GPT2Tokenizer.from_pretrained("gpt2") # TODO calcualte how "off" this estimate is. Try tiktoken
            self.token_count = []
        else:
            print("GPTAnonymizer is Live. Subsequent calls to anonymize will incure a cost on this client.")


    def count_tokens(self, prompt):
        tokens = self.dev_tokenizer.encode(prompt, add_special_tokens=False)
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
        labeled_substrings_list = group.apply(lambda row: self.extract_labeled_substrings(row[data_column], row[label_column]), axis=1).tolist()
        for labeled_substrings in labeled_substrings_list:
            for key, value in labeled_substrings.items():
                if key in group_labeled_substrings:
                    group_labeled_substrings[key].update(value)
                else:
                    group_labeled_substrings[key] = value.copy()

        return group_labeled_substrings
        
    def anonymize_data(self, data, labels, new_mappings):
        anonymized_data = data
        new_labels = sorted(labels, key=lambda x: x[1])
        if self.client:
            for i in range(len(new_labels)):
                label = new_labels[i]
                label_name = label[2].upper()
                
                original_text = anonymized_data[label[0]:label[1]]
                
                if label_name in self.label_manager.label_names:
                    new_text = match_casing(new_mappings[original_text.lower()], original_text)
                else:
                    # update down index label indicies by the new length of the 
                    new_text = f"[[{label_name}]]"

                anonymized_data = anonymized_data[:label[0]] + new_text + anonymized_data[label[1]:]
                offset = len(new_text) - len(original_text)
                new_labels[i] = (label[0], label[1] + offset, label[2])
                for j in range(i + 1, len(new_labels)):
                    new_labels[j] = (new_labels[j][0] + offset, new_labels[j][1] + offset, new_labels[j][2])
        
        return anonymized_data, new_labels
    
    def generate_missing_reprompt(self, new_mappings, all_gpt_targets):
        reprompt = ""
        missing_mappings = [element for element in all_gpt_targets if element not in new_mappings]
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
                        feedback = self.label_manager.get_anoymization_feedback(label, 
                                                                                substring, 
                                                                                new_mappings[substring])
                        if feedback:
                            reprompt = f"{reprompt} {feedback}"
                            gpt_target_list.append(substring)
        
        if gpt_target_list:
            reprompt = f"{self.feedback_reprompt} {gpt_target_list}\n[[FEEDBACK]]{reprompt}"

        return reprompt, gpt_target_list
    
    def generate_new_mappings(self, prompt_messages, labeled_substrings, assert_targets=[], mappings={}, additional_max_tokens=0, debug_content=""):
        if not debug_content:
            chat_completion = self.client.chat.completions.create(
                messages = prompt_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens + additional_max_tokens,
                frequency_penalty=self.frequency_penalty,
                model=self.gpt_model,
            )
            print(chat_completion.choices[0].message.content)

            new_mappings = extract_dictionary(chat_completion.choices[0].message.content)
        else:
            print(debug_content)
            new_mappings = extract_dictionary(debug_content)
        new_mappings = {key.lower(): value for key, value in new_mappings.items()}

        if assert_targets:
            assert all(target in new_mappings for target in assert_targets), f"Elements still missing from {assert_targets} in response mapping {new_mappings}"

        new_mappings = mappings | new_mappings

        # if labeled_substrings.get('NAME'):
        #     new_names = {key: new_mappings[key] for key in new_mappings if key in labeled_substrings['NAME']}
        #     self.label_manager.update_prior_label_mapping("NAME", new_names)
        
        return new_mappings

    def anonymize_group(self, group, data_column, label_column, group_name="CHAT HISTORY", dialogue_identifier_column='IsTutor', dialogue_identifier_func=is_tutor):
        labeled_substrings = self.extract_group_labeled_substrings(group, data_column, label_column)
        
        assistant_group_prompt = self.assistant_general_prompt
        group_prompt = ""
        all_gpt_targets = []
        for label in labeled_substrings:
            gpt_target_list = []

            if label in self.label_manager.label_names:
                gpt_target_list = labeled_substrings[label]
            
            # if label == "NAME":
            #     for name in labeled_substrings[label]:
            #         # if self.label_manager.get_prior_label_mappings("NAME").get(name) is None:
            #         gpt_target_list.append(name)
            # elif label in self.label_manager.label_names:
            #     gpt_target_list = labeled_substrings[label]

            if gpt_target_list:
                all_gpt_targets.extend(gpt_target_list)
                assistant_group_prompt = f"{assistant_group_prompt}\n{self.label_manager.assistant_label_prompts[label]}"
                group_prompt = f"{group_prompt} [[{label}]]:{gpt_target_list}" # -> "Hi Tom, how are you? I'm good cindy! Just raining here in new york" -> [[NAME]] [tom, cindy] [[LOCATION_ADDRESS]] [new york]
        
        new_mappings = {}
        
        if all_gpt_targets:
            # TODO create functions for these string concat variables
            grouped_data = '\n'.join([f"[[{dialogue_identifier_func(row[dialogue_identifier_column])}]] {row[data_column]}" for _, row in group.iterrows()])
            group_prompt = f"{group_prompt}\n[[{group_name}]]\n{grouped_data}"

            messages=[{"role": "system", "content": assistant_group_prompt},
                      {"role": "user", "content": group_prompt}
                    ]
            if self.client:
                print("----")
                print(messages)
                print(all_gpt_targets)
                new_mappings = self.generate_new_mappings(messages, labeled_substrings)

                missing_reprompt, missing_targets = self.generate_missing_reprompt(new_mappings, all_gpt_targets)
                print(missing_reprompt)
                if missing_reprompt:
                    messages=[{"role": "system", "content": assistant_group_prompt},
                              {"role": "user", "content": group_prompt},
                              {"role": "assistant", "content": str(new_mappings)},
                              {"role": "user", "content": missing_reprompt}
                             ]
                    print(messages)

                    new_mappings = self.generate_new_mappings(messages, labeled_substrings, assert_targets=missing_targets, mappings=new_mappings, additional_max_tokens=self.reprompt_additional_tokens)

                feedback_reprompt, reprompt_targets = self.generate_feedback_reprompt(new_mappings, labeled_substrings, all_gpt_targets)
                if feedback_reprompt:
                    messages=[{"role": "system", "content": assistant_group_prompt},
                              {"role": "user", "content": group_prompt},
                              {"role": "assistant", "content": str(new_mappings)},
                              {"role": "user", "content": feedback_reprompt}
                             ]
                    print(messages)
                    new_mappings = self.generate_new_mappings(messages, labeled_substrings, assert_targets=reprompt_targets, mappings=new_mappings, additional_max_tokens=self.reprompt_additional_tokens)
 
            else:
                self.token_count.append(self.count_tokens(assistant_group_prompt) + self.count_tokens(group_prompt))
                
        # group[[f'anonymized_{data_column}', f'new_{label_column}']] = group.apply(lambda row: self.anonymize_data(row[data_column], row[label_column], self.label_manager.get_prior_label_mappings("NAME") | new_mappings), axis=1, result_type ='expand')
        group[[f'anonymized_{data_column}', f'new_{label_column}']] = group.apply(lambda row: self.anonymize_data(row[data_column], row[label_column], new_mappings), axis=1, result_type ='expand')

        return group
    
    def anonymize(self, 
                  df, 
                  data_columns=['Message'], 
                  label_columns=['Message_labels'], 
                  context_groups=['FlowGeneratorSessionInterventionId']):
        
        for data_column, label_column in zip(data_columns, label_columns):
            if context_groups:
                df = df.groupby(context_groups).apply(lambda group: self.anonymize_group(group, data_column, label_column)).reset_index(drop=True)
            else:
                df = df.groupby(level=0).apply(lambda group: self.anonymize_group(group, data_column, label_column)).reset_index(drop=True)
                
        return df