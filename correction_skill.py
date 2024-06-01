from email import message
import logging
import Levenshtein
import jellyfish
import pandas as pd
from opsdroid.skill import Skill
from opsdroid.matchers import match_parse
from opsdroid.matchers import match_event, match_parse, match_regex
from opsdroid.events import JoinRoom, UserInvite
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import re


class DistanceModel:
    def __init__(self, words_path):
        self.words = self.load_words(words_path)
        self.words_set = set(self.words)
        self.soundex_dict = self.build_soundex_dict(self.words)

    def load_words(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            words = file.read().splitlines()
        return words

    def build_soundex_dict(self, words):
        soundex_dict = {}
        for word in words:
            soundex_code = jellyfish.soundex(word)
            if soundex_code not in soundex_dict:
                soundex_dict[soundex_code] = []
            soundex_dict[soundex_code].append(word)
        return soundex_dict

    def calculate_distance(self, word1, word2):
        return Levenshtein.distance(word1, word2)

    def find_closest_word(self, word):
        soundex_code = jellyfish.soundex(word)
        candidate_words = self.soundex_dict.get(soundex_code, self.words)
        min_distance = float('inf')
        closest_word = None
        for persian_word in candidate_words:
            distance = self.calculate_distance(word, persian_word)
            if distance < min_distance:
                min_distance = distance
                closest_word = persian_word
        return closest_word

    def inference(self, sentence):
        words = sentence.split()
        cleaned_sentence = ' '.join(word if word in self.words_set else self.find_closest_word(word) for word in words)
        return cleaned_sentence

class RegexModel:
    def __init__(self, words_df_path):
        self.words_df = pd.read_csv(words_df_path, encoding='utf-8', sep='\t')
        self.wrong_to_correct = dict(zip(self.words_df['wrong'], self.words_df['correct']))
        self.regex = self.compile_regex()

    def compile_regex(self):
        pattern = '|'.join(r'\b' + re.escape(wrong) + r'\b' for wrong in self.wrong_to_correct.keys())
        return re.compile(pattern)

    def correct_text(self, text):
        def replace(match):
            return self.wrong_to_correct[match.group(0)]
        return self.regex.sub(replace, text)

class CombinedModel:
    def __init__(self, model_path, words_path, words_df_path, use_gpt2=False):
        self.use_gpt2 = use_gpt2
        if self.use_gpt2:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.distance_model = DistanceModel(words_path)
        self.regex_model = RegexModel(words_df_path)

    def inference(self, sentence):

        cleaned_sentence = self.distance_model.inference(sentence)
        corrected_sentence = self.regex_model.correct_text(cleaned_sentence)

        if self.use_gpt2:
            input_ids = self.tokenizer.encode(f"<start> {corrected_sentence} <end>", return_tensors='pt')
            with torch.no_grad():
                outputs = self.model.generate(input_ids, max_length=64, num_beams=5, early_stopping=True)
            final_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            final_output_words = final_output.split()
        else:
            final_output_words = corrected_sentence.split()

        input_words = sentence.split()
        if len(final_output_words) != len(input_words):
            final_output_words = final_output_words[:len(input_words)]
            if len(final_output_words) < len(input_words):
                final_output_words += input_words[len(final_output_words):]

        return ' '.join(final_output_words)

words_path = "resources/distinct_words.txt"
words_df_path = "resources/created_words_dataset.txt"
model = CombinedModel(model_path=None, words_path=words_path, words_df_path=words_df_path, use_gpt2=False)


class AcceptDirectMessageSkill(Skill):
    @match_event(UserInvite)
    async def accept_direct_message(self, event):
        logging.info(f"accept_direct_message -> {event}")
        await message.respond("You can enter any persian text. We will correct typo and different mistakes.\nEnter your input:")

    @match_event(JoinRoom)
    async def join_room(self, event):
        logging.info(f"new-joining -> {event}")
        await message.respond("Wellcome to the best bot ever.\nEnter your input to get the correct one:")

class SpellingSkill(Skill):
    @match_parse(r'{speech}')
    async def correction(self, message):
        sample_input = message.entities['speech']['value']
        await message.respond(model.inference(sample_input))