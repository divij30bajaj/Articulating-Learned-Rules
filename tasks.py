import abc
import random
import re
from datetime import time
from typing import Dict, Type


class Task:
    def __init__(self, num_examples_per_prompt):
        self.template = "Input: {}; Label: {}"
        self.num_examples_per_prompt = num_examples_per_prompt

    @abc.abstractmethod
    def generate_prompt(self):
        pass


TASK_CLASSES: Dict[str, Type[Task]] = {}


def register_task(task_id: str):
    def decorator(cls):
        TASK_CLASSES[task_id] = cls
        return cls
    return decorator


def generate_random_time(format_24=False):
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    random_time = time(hour, minute)
    if format_24:
        return random_time.strftime("%H%M")
    else:
        return random_time.strftime("%I:%M %p")


# Task 1: Input contains a 12-hour time format
@register_task('task1')
class TimeFormat(Task):
    def __init__(self, num_examples_per_prompt):
        super().__init__(num_examples_per_prompt)
        self.prompt_styles = ["The time is {TIME}", "{TIME} is the current time",
                              "The clock tells {TIME} as the current time"]

    def generate_prompt(self):
        in_context_examples = []

        for _ in range(self.num_examples_per_prompt + 1):
            prompt_style = random.sample(self.prompt_styles, 1)[0]
            if random.random() < 0.5:
                random_time = generate_random_time(format_24=False)
                label = True
            else:
                random_time = generate_random_time(format_24=True) + " hours"
                label = False
            example_input = prompt_style.replace("{TIME}", random_time)
            example = self.template.format(example_input, label)
            in_context_examples.append(example)

        return in_context_examples


# Input ends with an exclamation mark
@register_task('task2')
class ExclamationMark(Task):
    def __init__(self, num_examples_per_prompt):
        super().__init__(num_examples_per_prompt)
        f = open('inputs/simple.txt', 'r')
        sentences = f.readlines()
        self.sentences = [s.replace("\n", "") for s in sentences]

    def generate_prompt(self):
        in_context_examples = []
        sampled_sentences = random.sample(self.sentences, self.num_examples_per_prompt + 1)
        for sentence in sampled_sentences:
            if random.random() < 0.5:
                sentence = "$$" + sentence[:-1] + "$$"
                label = True
            else:
                label = False
            example = self.template.format(sentence, label)
            in_context_examples.append(example)
        return in_context_examples


# Task 3: First word is in all caps in the input
@register_task('task3')
class FirstWordCaps(Task):
    def __init__(self, num_examples_per_prompt):
        super().__init__(num_examples_per_prompt)
        f = open('inputs/simple.txt', 'r')
        sentences = f.readlines()
        self.sentences = [s.replace("\n", "") for s in sentences]

    def generate_prompt(self):
        in_context_examples = []
        sampled_sentences = random.sample(self.sentences, self.num_examples_per_prompt + 1)
        for i, sentence in enumerate(sampled_sentences):
            if random.random() < 0.5:
                sentence = " ".join([sentence.split(" ")[0].upper()] + sentence.split(" ")[1:])
                label = True
            else:
                sentence = sentence[0].lower() + sentence[1:]
                label = False
            example = self.template.format(sentence, label)
            in_context_examples.append(example)
        return in_context_examples


# Task 4: Input mentions apples
@register_task('task4')
class Apples(Task):
    def __init__(self, num_examples_per_prompt):
        super().__init__(num_examples_per_prompt)
        self.fruits = ["orange", "banana", "pear", "melon", "kiwi"]
        f = open('inputs/apples.txt', 'r')
        sentences = f.readlines()
        self.sentences = [s.replace("\n", "") for s in sentences]

    def generate_prompt(self):
        in_context_examples = []
        sampled_sentences = random.sample(self.sentences, self.num_examples_per_prompt + 1)
        for i, sentence in enumerate(sampled_sentences):
            if random.random() < 0.5:
                label = True
            else:
                fruit = random.sample(self.fruits, 1)[0]
                sentence = sentence.replace("apple", fruit).replace("Apple", fruit[0].upper() + fruit[1:])
                label = False
            example = self.template.format(sentence, label)
            in_context_examples.append(example)
        return in_context_examples


# Task 5: Input contains numbers spelled out as words
@register_task('task5')
class SpelledNumbers(Task):
    def __init__(self, num_examples_per_prompt):
        super().__init__(num_examples_per_prompt)
        self.number_names = {
            '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
            '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten'
        }
        f = open('inputs/numbers.txt', 'r')
        sentences = f.readlines()
        self.sentences = [s.replace("\n", "") for s in sentences]

    def generate_prompt(self):
        in_context_examples = []
        sampled_sentences = random.sample(self.sentences, self.num_examples_per_prompt + 1)
        for i, sentence in enumerate(sampled_sentences):
            if random.random() < 0.5:
                label = False
            else:
                sentence = re.sub(r'\b([1-9]|10)\b', lambda x: self.number_names[x.group()], sentence)
                label = True
            example = self.template.format(sentence, label)
            in_context_examples.append(example)
        return in_context_examples


# Task 6: Input contains two sentences
@register_task('task6')
class TwoSentences(Task):
    def __init__(self, num_examples_per_prompt):
        super().__init__(num_examples_per_prompt)
        f = open('inputs/simple.txt', 'r')
        sentences = f.readlines()
        self.sentences = [s.replace("\n", "") for s in sentences]

    def generate_prompt(self):
        in_context_examples = []
        sampled_sentences = random.sample(self.sentences, self.num_examples_per_prompt + 1)
        other_sentences = list(set(self.sentences).difference(set(sampled_sentences)))
        for i, sentence in enumerate(sampled_sentences):
            if random.random() < 0.5:
                random_sentence = random.sample(other_sentences, 1)[0]
                sentence = "{} {}".format(sentence, random_sentence)
                label = True
            else:
                label = False
            example = self.template.format(sentence, label)
            in_context_examples.append(example)
        return in_context_examples
