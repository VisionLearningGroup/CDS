
import os
import glob



files = glob.glob('*.txt')


for file in files:

    lines = open(file, 'r').readlines()
    new_lines = []

    new_file = open(file, 'w')

    for line in lines:

        line_new = line.replace('./', '../')
        new_file.write(line_new)

    new_file.close()

