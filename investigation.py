import json
import tqdm

from models import Model

num_tasks = 6
old_instruction = "Given the example inputs and their labels, what label should be given to the last input?" \
                  " Only output the label and nothing else.\n"
new_instruction = "Given few sentences and their labels, what label should be given to the last input and what rule " \
                  "would you use to classify the last input given the reasoning: {}? Only fill the blanks and output " \
                  "nothing else:\n" \
                  "Rule: Label as True if and only if <BLANK>, otherwise label as False.\nLabel: <BLANK>\n"

supported_models = ["claude-sonnet", "gpt-4", "gpt-4o"]

ground_truth_keywords = [["AM/PM", "AM or PM", "12-hour format"], ["ends with an exclamation mark"],
                         ["starts with 'THE' in all caps", 'starts with "THE" in all caps'],
                         ["mentions apples"], ["spelled", "word"], ["two or more", "two sentences"]]


def modify_reasoning(articulation, type="corrupt"):
    if type == "corrupt":
        new_reasoning = articulation.replace("False", "True").replace("Label as True", "Label as False")
    else:
        new_reasoning = "." * 50
    return new_reasoning


for task_id in range(num_tasks):
    f = open('data/task{}.json'.format(task_id), 'r')
    prompts = json.load(f)
    f.close()

    for model_name in supported_models:
        model = Model(model_name)

        print("Evaluating {}".format(model_name))

        # Loading articulations saved in Step 2
        fr = open('results/{}_task{}_articulation.txt'.format(model_name, task_id), 'r')
        articulations = fr.readlines()
        fr.close()

        accuracy = 0
        f = open('{}_task{}.txt'.format(model_name, task_id), 'w')

        # Evaluating on first 20 prompts
        for k, sample in enumerate(tqdm.tqdm(prompts[:20])):
            prompt = sample['input'].replace(old_instruction, new_instruction)
            articulation = articulations[k].replace("Rule: ", "").replace("\n", "")

            # Modifying the articulation by either corrupting it (type="corrupt") or replacing with filler periods
            new_reasoning = modify_reasoning(articulation, type="filler")  # Replace with "corrupt" for corrupting instead

            prompt = prompt.format(new_reasoning)
            response = model.query(prompt=prompt)
            f.write("{},{}".format(str(k), response))

            output = True if any(keyword in response for keyword in ground_truth_keywords[task_id - 1]) else False
            if output:
                accuracy += 1
        print("Model: {} Accuracy: {}%".format(model_name, accuracy*5))
        f.close()
