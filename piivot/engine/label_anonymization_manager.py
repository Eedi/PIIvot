import re
import Levenshtein


def contains_counting_sequence(input_string):
    """
    Checks if the input_string contains any sequences of 3 or more digits
    that count up or down (ignoring spaces).

    Args:
        input_string (str): The string to check.

    Returns:
        bool: True if the string contains such sequences, False otherwise.
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

class LabelAnonymizationManager():

    def get_similar_words(self, str1, str2):
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        similar_words = set()

        for word1 in words1:
            for word2 in words2:
                if Levenshtein.distance(word1, word2) <= self.distance_threshold:
                    similar_words.add((word1, word2))

        return similar_words

    def name_feedback_func(self, original_text, anonymized_text):
        feedback = ""
        if Levenshtein.distance(original_text, anonymized_text) <= self.distance_threshold:
            feedback = self.append_feedback(feedback, f"{anonymized_text} is too similar to {original_text}. Ensure the anonymized replacement is a different name.")
        
        similar_words_feedback = self.get_similar_words_feedback("NAME", original_text, anonymized_text)
        if similar_words_feedback:
            feedback = self.append_feedback(feedback, similar_words_feedback)
        
        return feedback
    
    def phone_number_feedback_func(self, original_text, anonymized_text):
        feedback = ""
        if Levenshtein.distance(original_text, anonymized_text) <= self.distance_threshold:
            feedback = self.append_feedback(feedback, f"{anonymized_text} is too similar to {original_text}. Ensure the anonymized replacement is a different phone number.")
            
        if any(char.isalpha() for char in anonymized_text):
            feedback = self.append_feedback(feedback, f"{anonymized_text} contains alphabetic characters. Ensure the anonymized phone number doesn't have any placeholder characters.")

        if contains_counting_sequence(anonymized_text):
            feedback = self.append_feedback(feedback, f"{anonymized_text} contains ascending or descending strings of digits. Ensure the anonymized phone number doesn't look fake.")
        
        similar_words_feedback = self.get_similar_words_feedback("PHONE_NUMBER", original_text, anonymized_text)
        if similar_words_feedback:
            feedback = self.append_feedback(feedback, similar_words_feedback)
        
        return feedback
    
    def location_address_feedback_func(self, original_text, anonymized_text):
        feedback = ""
        if Levenshtein.distance(original_text, anonymized_text) <= self.distance_threshold:
            feedback = self.append_feedback(feedback, f"{anonymized_text} is too similar to {original_text}. Ensure the anonymized replacement is a different word.")
        
        similar_words_feedback = self.get_similar_words_feedback("LOCATION_ADDRESS", original_text, anonymized_text)
        if similar_words_feedback:
            feedback = self.append_feedback(feedback, similar_words_feedback)
            
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
        
        similar_words_feedback = self.get_similar_words_feedback("SCHOOL_NAME", original_text, anonymized_text)
        if similar_words_feedback:
            feedback = self.append_feedback(feedback, similar_words_feedback)

        return feedback
    
    def date_of_birth_feedback_func(self, original_text, anonymized_text):
        feedback = ""
        if Levenshtein.distance(original_text, anonymized_text) <= self.distance_threshold:
            feedback = self.append_feedback(feedback, f"{anonymized_text} is too similar to {original_text}. Ensure the anonymized replacement is a different day, month, and/or year.")
        
        similar_words_feedback = self.get_similar_words_feedback("DATE_OF_BIRTH", original_text, anonymized_text)
        if similar_words_feedback:
            feedback = self.append_feedback(feedback, similar_words_feedback)
            
        return feedback
    
    def default_feedback_func(self, original_text, anonymized_text):
        return ""
    
    def get_similar_words_feedback(self, label_name, original_text, anonymized_text):
        prior_mappings = self.get_prior_label_mappings(label_name)
        for prior_text in prior_mappings:
            if prior_text != original_text:
                similar_words = self.get_similar_words(prior_text,
                                                       original_text)
                prior_mapped_text = prior_mappings[prior_text]

                anonymized_similar_words = self.get_similar_words(prior_mapped_text,
                                                                  anonymized_text)
                if len(similar_words) > len(anonymized_similar_words):
                    return f'The previously anonymized {label_name} {prior_text} was anonymized to {prior_mapped_text}. {prior_text} and {original_text} share the following similarly spelled words: {similar_words}. Ensure the anonymized replacement for {original_text} follows the same pattern as {prior_text} is to {prior_mapped_text}.'
        return None
    
    def append_feedback(self, feedback, new_feedback):
        if feedback:
            return f"{feedback} {new_feedback}"
        else:
            return new_feedback
    
    def __init__(self,
                 distance_threshold = 1):
        
        self.prior_mappings = {}
        self.distance_threshold = distance_threshold
        self.label_names = ['NAME','PHONE_NUMBER','LOCATION_ADDRESS','SCHOOL_NAME','DATE_OF_BIRTH']

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

        if feedback:
            self.prior_mappings[label_name].pop(original_text, None)

        return feedback


    def update_prior_label_mappings(self, additional_mappings):
        for key, value in additional_mappings.items():
            if key in self.prior_mappings:
                self.prior_mappings[key].update(value)
            else:
                self.prior_mappings[key] = value
    
    def clear_prior_label_mappings(self):
        self.prior_mappings = {}
    
    def get_prior_label_mappings(self, label_name):
        if label_name in self.prior_mappings:
            label_prior_mappings = self.prior_mappings.get(label_name)

            return label_prior_mappings.copy()
        else:
            return []