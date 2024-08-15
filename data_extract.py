import lzma
import os
import tarfile
from tqdm import tqdm
# pip install datasets

# https://www.youtube.com/watch?v=UU1WVnMk4E8 @ 4:43:36

number_of_tar_data_files = 21
number_of_xz_files = 1000
tar_data_files = ['data/urlsf_subset{:02d}.tar'.format(i) for i in range(number_of_tar_data_files)]

def extract_tar_files():
    for file in tar_data_files:
        print(file)
        with tarfile.open(file, 'r') as tar:
            tar.extractall('data/')

def get_xz_files(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

folder_path = "C:/Users/1279919/OneDrive - Chipotle Mexican Grill/Desktop/osibot/data/openwebtext"
output_training_file = "output_training.txt"
output_validation_file = "output_validation.txt"
files = get_xz_files(folder_path)
total_files = len(files)
split_index = int(total_files * 0.9)
training_files = files[:split_index]
validation_files = files[split_index:]
vocab_file = "vocab.txt"
vocab = set()

with open(output_training_file, "w", encoding="utf-8") as output_file:
    for filename in tqdm(training_files, total=len(training_files)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as input_file:
            text = input_file.read()
            output_file.write(text)
            characters = set(text)
            vocab.update(characters)

with open(output_validation_file, "w", encoding="utf-8") as output_file:
    for filename in tqdm(validation_files, total=len(validation_files)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as input_file:
            text = input_file.read()
            output_file.write(text)
            characters = set(text)
            vocab.update(characters)

with open(vocab_file, "w", encoding="utf-8") as vocab_file:
    for char in vocab:
        vocab_file.write(char + "\n")
