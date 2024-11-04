import json
import tqdm

from models import Model

num_tasks = 6
for task_id in range(1, num_tasks + 1):
    f = open('data/task{}.json'.format(task_id), 'r')
    prompts = json.load(f)
    f.close()

    old_instruction = "Given the example inputs and their labels, what label should be given to the last input?" \
                      " Only output the label and nothing else.\n"
    new_instruction = "Given few sentences and their labels, what label should be given to the last input and what " \
                      "rule would you use to classify the last input? Only fill the blanks and output nothing else:\n" \
                      "Rule: Label as True if and only if <BLANK>, otherwise label as False.\nLabel: <BLANK>\n"

    supported_models = ["gemini-1.5-pro", "claude-sonnet", "gpt-4", "gpt-4o", "llama3-70b", "llama3.1-70b"]

    ground_truth_keywords = [["AM/PM", "AM or PM", "12-hour format"], ["ends with an exclamation mark"],
                             ["starts with 'THE' in all caps", 'starts with "THE" in all caps'],
                             ["mentions apples"], ["spelled", "word"], ["two or more", "two sentences"]]

    for model_name in supported_models:
        model = Model(model_name)

        print("Evaluating {}".format(model_name))

        accuracy = 0
        f = open('{}_task{}.txt'.format(model_name, task_id), 'w')
        for k, sample in enumerate(tqdm.tqdm(prompts)):
            prompt = sample['input'].replace(old_instruction, new_instruction)
            response = model.query(prompt=prompt)
            f.write("{},{}".format(str(k), response))

            output = True if any(keyword in response for keyword in ground_truth_keywords[task_id - 1]) else False
            if output:
                accuracy += 1
        print("Model: {} Accuracy: {}%".format(model_name, accuracy))
        f.close()
