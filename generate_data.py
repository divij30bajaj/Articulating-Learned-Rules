import argparse
import json
import random
from tasks import TASK_CLASSES


instruction = "Given the example inputs and their labels, what label should be given to the last input?" \
           " Only output the label and nothing else.\n"
num_tasks = 6


def main(num_examples_per_prompt, num_prompts):
    for i in range(num_tasks):
        task_class = TASK_CLASSES[f"task{i+1}"]
        task = task_class(num_examples_per_prompt=num_examples_per_prompt)

        prompts = []
        for _ in range(num_prompts):
            in_context_examples = task.generate_prompt()
            random.shuffle(in_context_examples)
            prompt = " ".join("\n".join(in_context_examples).split(" ")[:-1])
            prompt = instruction + prompt
            data_sample = {"input": prompt, "ground_truth": in_context_examples[-1].split(" ")[-1]}
            prompts.append(data_sample)

        f = open('data/task{}.json'.format(i+1), 'w')
        json.dump(prompts, f)
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples_per_prompt", default=32, help="Number of in-context examples in each prompt "
                                                                      "before the query")
    parser.add_argument("--num_prompts", default=100, help="Number of prompts to evaluate on")
    args = parser.parse_args()

    main(args.num_examples_per_prompt, args.num_prompts)