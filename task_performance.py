import json
import tqdm

from models import Model

num_tasks = 6
supported_models = ["llama3-70b", "llama3.1-70b",  "claude-sonnet", "gpt-4", "gpt-4o", "gemini-1.5-pro"]
for i in range(num_tasks):
    print(f"==== TASK {i+1} ===")

    f = open('data/task{}.json'.format(i+1), 'r')
    prompts = json.load(f)
    f.close()

    for model_name in supported_models:
        model = Model(model_name)

        print("Evaluating {}".format(model_name))

        accuracy = 0
        for sample in tqdm.tqdm(prompts):
            response = model.query(prompt=sample['input'])
            output = "True" if "true" in response.lower() else "False"
            ground_truth = sample['ground_truth']
            if ground_truth == output:
                accuracy += 1

        print("Model: {} Accuracy: {}%".format(model_name, accuracy*100/len(prompts)))
