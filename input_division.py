import random


INPUT_FILE_NAME = './inputs/original.txt'
OUTPUT_TRAINING = './inputs/training.txt'
OUTPUT_TESTING = './inputs/testing.txt'

with open(INPUT_FILE_NAME, 'r', encoding='utf-8') as input_file:
    lines = input_file.readlines()
    random.shuffle(lines)
    training_lines = lines[:int(len(lines)*0.8)]
    testing_lines = lines[int(len(lines)*0.8):]

    with open(OUTPUT_TRAINING, 'w') as output_training:
        for line in training_lines:
            # parse line to utf-8
            output_training.write(line)

    with open(OUTPUT_TESTING, 'w') as output_testing:
        for line in testing_lines:
            # parse line to utf-8
            output_testing.write(line)
