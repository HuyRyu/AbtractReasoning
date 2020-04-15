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
    count = 0
    total_samples = 0
    for task_file in task_data:
        task_file = os.path.join(split, task_file)
        task = json.load(open(task_file))
        input = []
        target = []
        for tr in task['train']:
            resized_inp = np.zeros((30, 30))
            resized_out = np.zeros((30, 30))
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
        resized_inp = np.zeros((30, 30))
        resized_out = np.zeros((30, 30))

        inp = np.array(test_task[0]['input'])
        out = np.array(test_task[0]['output'])

        h_inp, w_inp = np.shape(inp)
        h_out, w_out = np.shape(out)

        resized_inp[:h_inp, :w_inp] = inp
        resized_out[:h_out, :w_out] = out

        # add padding for input set
        if len(input) < 12:
            idx_padding = len(input)
            input.append(resized_inp)
            input += [np.zeros((30, 30))] * (12 - len(input))
            count += 1
        else:
            input = input[:11] + [np.zeros((30, 30))]
            idx_padding = 12

        total_samples += 1
        entries.append({'input': np.array(input), 'target': resized_out, 'idx_padding': idx_padding})

    print(count * 1.0 / total_samples)
    with open('%s.pkl' % split, 'wb') as f:
        pickle.dump(entries, f)


reconstruct_data('training')