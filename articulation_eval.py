import json
import tqdm

from models import Model

num_tasks = 6
old_instruction = "Given the example inputs and their labels, what label should be given to the last input?" \
                  " Only output the label and nothing else.\n"
new_instruction = "Given few sentences and their labels, what label should be given to the last input and what " \
                  "rule would you use to classify the last input? Only fill the blanks and output nothing else:\n" \
                  "Rule: Label as True if and only if <BLANK>, otherwise label as False.\nLabel: <BLANK>\n"

supported_models = ["claude-sonnet", "gpt-4", "gpt-4o"]

ground_truth_keywords = [["AM/PM", "AM or PM", "12-hour format"], ["enclosed", "$$", "surrounded"],
                         ["starts with 'THE'", "in all caps", "in all capital", 'starts with "THE"'],
                         ["apples"], ["spelled", "word"], ["two"]]

for task_id in range(1, num_tasks + 1):
    f = open('data/task{}.json'.format(task_id), 'r')
    prompts = json.load(f)
    f.close()

    for model_name in supported_models:
        model = Model(model_name)

        print("Evaluating {}".format(model_name))

        accuracy = 0
        fl = open('results/{}_task{}_labels.txt'.format(model_name, task_id), 'w')
        fr = open('results/{}_task{}_articulation.txt'.format(model_name, task_id), 'w')
        for k, sample in enumerate(tqdm.tqdm(prompts)):
            prompt = sample['input'].replace(old_instruction, new_instruction)
            response = model.query(prompt=prompt)

            # Handling different possibilities of model response formatting (especially for GPT-4o)
            components = response.split("\n")
            if len(components) == 1:
                components = response.split("Label:")
            if len(components) == 1:
                articulation = ""
                label = components[0]
            else:
                articulation = components[0]
                label = components[1]

            fl.write(label + "\n")
            fr.write(articulation + "\n")

            # Comparing articulation with pre-defined keywords for each task
            output = True if any(keyword in articulation for keyword in ground_truth_keywords[task_id - 1]) else False
            if output:
                accuracy += 1
        print("Model: {} Accuracy: {}%".format(model_name, accuracy*100/len(prompts)))
        f.close()
        fl.close()
        fr.close()
