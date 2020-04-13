import json
import numpy as np
from os import listdir
import os
import numpy as np
import _pickle as pickle

def reconstruct_data(split):
    task_data = listdir(split)
    max_len = 0
    entries = []
    for task_file in task_data:
        task_file = os.path.join(split, task_file)
        task = json.load(open(task_file))
        input = []
        for tr in task['train']:
            resized_inp = np.zeros((90, 90))
            resized_out = np.zeros((90, 90))
            inp = np.array(tr['input'])
            out = np.array(tr['output'])
            h_inp, w_inp = np.shape(inp)
            h_out, w_out = np.shape(out)
            resized_inp[:h_inp, :w_inp] = inp
            resized_out[:h_out, :w_out] = out
            input.append(resized_inp)
            input.append(resized_out)

        # get the input and target of test-task
        test_task = task['test']
        resized_inp = np.zeros((90, 90))
        resized_out = np.zeros((90, 90))
        inp = np.array(test_task[0]['input'])
        # out = np.array(test_task[0]['output'])
        h_inp, w_inp = np.shape(inp)
        h_out, w_out = np.shape(out)
        resized_inp[:h_inp, :w_inp] = inp
        # resized_out[:h_out, :w_out] = out

        input.append(resized_inp)
        if len(input) > max_len:
            max_len = len(input)
        entries.append({'input': np.array(input), 'target': resized_out})

    print(max_len)
    with open('test.pkl', 'wb') as f:
        pickle.dump(entries, f)

reconstruct_data('test')