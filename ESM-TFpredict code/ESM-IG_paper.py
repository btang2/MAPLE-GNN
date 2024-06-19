import os
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import torch
import numpy as np
import torch
import glob
import tensorflow as tf
from keras import backend as K

matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)


def f1_score_new(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
def sliding_window_avg(sequence, window_size, overlap):
    step = window_size - overlap
    return [
        sum(sequence[i:i+window_size]) / window_size
        for i in range(0, len(sequence) - window_size + 1, step)
    ]

window_size_sum = []
def pad_or_truncate_sequences(X, target_length):
    # Initialize an empty list to store the processed sequences
    processed_X = []
    counter = 0
    # Iterate over each sequence in X
    for sequence in X:
        if sequence.shape[0] > target_length:
            # If the sequence is longer than the target length, truncate it, 95%
            #truncated_sequence = sequence[:target_length]
            #processed_X.append(truncated_sequence)

            # If the sequence is longer than the target length, drop it
            #Y.pop(counter)
            #counter= counter-1

            # If the sequence is longer than the target length, use slide window to average it
            #overlap = 5

            #window_size = 2#len(sequence) // (target_length // overlap)
            window_size = len(sequence) // target_length + 1
            overlap = 0 #window_size//2
            """
            while True:
                if overlap >= window_size:
                    overlap = window_size // 2
                temp = sliding_window_avg(sequence, window_size, overlap)
                if len(temp) <= target_length:
                    avg_seq = temp
                    break
                else:
                    window_size += 1  # Adjust window size to try to achieve desired length
            """
            avg_seq = sliding_window_avg(sequence, window_size, overlap)
            #print("window size: ", window_size)
            window_size_sum.append(window_size)
            avg_seq = np.array(avg_seq)
            #print(avg_seq.shape[0])
            padding = np.zeros((target_length - avg_seq.shape[0], avg_seq.shape[1]))
            padded_sequence = np.concatenate((avg_seq, padding), axis=0)
            processed_X.append(padded_sequence)

        # If the sequence is shorter than the target length, pad it
        elif sequence.shape[0] < target_length:
            padding = np.zeros((target_length - sequence.shape[0], sequence.shape[1]))
            padded_sequence = np.concatenate((sequence, padding), axis=0)
            processed_X.append(padded_sequence)
        # If the sequence is exactly the target length, leave it as is
        else:
            processed_X.append(sequence)
        counter += 1
    # Convert the list of processed sequences to a 3D numpy array
    processed_X = np.stack(processed_X, axis=0)
    return np.array(processed_X)

# load deep learning model
model = tf.keras.models.load_model('best_model/best_model3.h5', custom_objects={'f1_score_new': f1_score_new}, compile=False) #, custom_objects={'balanced_binary_crossentropy': balanced_binary_crossentropy}
# Evaluate the restored model
#loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
#print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))


# load test data
path_0620 = '/home/liyuan/Project/DNA sequence/CCM/Fasta/zinc_finger/*.*'  #zinc_finger, max_length=1022, 320D, trainingmodellarge_0620_full_L6
files_0620 = glob.glob(path_0620)
# get file names
file_names = [os.path.basename(file) for file in files_0620]
X_0620 = []
Y_0620 = []
sequence = []
protein_ID = []
# Read the CSV file
df = pd.read_csv('test_data/zinc_finger.csv')
#df.rename(columns={"ID": "description", "FATSA ": "seq"}, inplace=True)
root_path = "/home/liyuan/Project/DNA sequence/CCM/Fasta/zinc_finger/"
for name in file_names:

    file_path = root_path + name#'./NTF->sp|A6X8Z5|RHG31_MOUSE.pt'

    #with open(file_path) as f:
    embedding_0620 = torch.load(file_path)
    embedding_0620 = embedding_0620['representations'][6].numpy()
    X_0620.append(embedding_0620)
    # find rows that contain the string
    pos = df.loc[df['description'].str.contains(name[3:-3])]
    seq = pos['seq'].values
    sequence.append(seq)
    protein_ID.append(name[0:-3])
    #print("seq:", seq)

print("sequence:", len(sequence))


X = X_0620
Y = Y_0620

emb_dim = len(X[0][0])
X = np.array(X)
len_criteria = 300

start = 17
view = 1
original_input = X[start:start+view]
sequence = sequence[start:start+view]
X = pad_or_truncate_sequences(original_input, len_criteria)
protein_ID = protein_ID[start:start+view]
print("protein_ID: ", protein_ID)
# provided zinc finger positions
zinc_finger_positions = [ (373,395), (401,423), (428,450)]

#Y = np.array(Y)

preprocessed_input = X
print("tf.shape(preprocessed_input): ", tf.shape(preprocessed_input))


baseline_input = tf.zeros(shape=(len(original_input), len_criteria, 320))

input_tensor = tf.convert_to_tensor(preprocessed_input, dtype=tf.float32)
baseline_tensor = tf.convert_to_tensor(baseline_input, dtype=tf.float32)

def integrated_gradients(inputs):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs)
        scaled_inputs = [baseline_tensor + (inputs - baseline_tensor) * alpha for alpha in np.linspace(0, 1, num=100)]
        outputs = [model(inputs) for inputs in scaled_inputs] #tf.expand_dims(inputs, axis=0)
    grads = [tape.gradient(output, inputs) for output in outputs]
    avg_grads = tf.reduce_mean(grads, axis=0)
    attributions = (inputs - baseline_tensor) * avg_grads
    return attributions

attributions = integrated_gradients(input_tensor)  #[input_length, 320]
attributions = np.average(attributions, axis=2) #[input_length, 1]
print("np.shape(attributions): ", np.shape(attributions))

predict_results = ((model.predict(input_tensor)[:] >= 0.5).astype(bool))*1
print("predict_results: ", predict_results)

# normalize the attributions for better visualization
#attributions = (attributions - np.min(attributions)) / (np.max(attributions) - np.min(attributions))
"""
plt.figure(figsize=(10, 4))
plt.bar(range(len(attributions[1])), attributions[1])
plt.xlabel('Feature Index')
plt.ylabel('Attribution')
plt.title('Integrated Gradients - TF Feature Attributions')
plt.show()
"""
#
def map_new_pos_to_original(i, window_size, overlap):
    step = window_size - overlap
    start = i * step
    end = start + window_size
    return [start, end]
print("original_input.shape[0]:", original_input.shape[0])
original_attributions = [] #np.zeros(original_input.shape[0])
original_attributions_sum = []

def remap(atr, w):
    remapped = []
    for i in atr:
        for j in range(w):
            #print("i: ", i)
            remapped.append(i)
    return remapped

for i in range(0, len(attributions)):
    length = len(original_input[i])
    window = length // len(attributions[i]) + 1
    attr = attributions[i][0:(length // window)]
    attr = remap(attr, window)
    original_attributions_sum.append(attr)
    #print(len(attr))

"""
for i in range (0, len(attributions)):
    if original_input[i].shape[1] > 512:
        for pos in range(0,len(attributions[i])):
            tem_window_size = window_size_sum[i]
            start, end = map_new_pos_to_original(pos, tem_window_size, 0)
            for j in range(start, end+1):
                original_attributions.append(attributions[pos])
        original_attributions_sum.append(original_attributions)
    else:
        original_attributions_sum.append(attributions[i])

print("original_attributions_sum: ", len(original_attributions_sum))
"""

"""
for i in range(0,len(original_attributions_sum)):
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(original_attributions_sum[i])), original_attributions_sum[i])
    plt.xlabel('original Feature Index')
    plt.ylabel('original_attributions')
    plt.title('Integrated Gradients - TF Feature Attributions')
    plt.show()
"""
#
for i in range(0,len(original_attributions_sum)):
    plt.figure(figsize=(10, 4))
    bars = plt.bar(range(len(original_attributions_sum[i])), original_attributions_sum[i])
    plt.xlabel('Amino acid position in original protein sequence', fontsize=20, fontweight = 'bold' )
    plt.ylabel('Attribution score', fontsize=15)
    #plt.title('IG Attributions-' + protein_ID[i] + "-" +str(predict_results[i]))
    seq_str = str(sequence[i]) + '-'*(len(original_attributions_sum[i]) - len(str(sequence[i])))
    print("seq_str: ", seq_str)
    # Add labels on top of each bar
    for idx, bar in enumerate(bars):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, seq_str[idx], va='bottom')  # va stands for vertical alignment

    plt.show()



# Convert to 0-indexed positions
zinc_finger_positions = [(start-1, end-1) for start, end in zinc_finger_positions]
# initialize lists to hold zinc finger and other attribute scores
zinc_finger_attributions = []
other_attributions = []

attributions_list = original_attributions_sum[0]
# iterate over each amino acid in the protein sequence
for i in range(len(attributions_list)):
    # check if the current position belongs to a zinc finger region
    if any(start <= i < end for start, end in zinc_finger_positions):
        zinc_finger_attributions.append(attributions_list[i])
    else:
        other_attributions.append(attributions_list[i])

# calculate average attribute scores
avg_zinc_finger_attr_score = sum(zinc_finger_attributions) / len(zinc_finger_attributions)
avg_other_attr_score = sum(other_attributions) / len(other_attributions)

print("Sequence name: ", protein_ID[0])
print('Average attribute score for zinc finger amino acids:', avg_zinc_finger_attr_score)
print('Average attribute score for other amino acids:', avg_other_attr_score)
print('Normalize for zinc finger amino acids:', avg_zinc_finger_attr_score/(avg_zinc_finger_attr_score + avg_other_attr_score))
print('Normalize for other amino acids:', avg_other_attr_score/(avg_zinc_finger_attr_score + avg_other_attr_score))

import scipy.stats
# Perform the two-sided t-test
t_statistic, p_value = scipy.stats.ttest_ind(zinc_finger_attributions, other_attributions)
# If the t-statistic is positive (indicating that the zinc finger scores are higher),
# divide the p-value by 2 to get the one-sided p-value.
if t_statistic > 0:
    p_value /= 2

print('t-statistic:', t_statistic)
print('one-sided p-value:', p_value)

import seaborn as sns
# Boxplots
# Create the boxplot data (list of data)
boxplot_data = [zinc_finger_attributions, other_attributions]
# Create the boxplot
# Create the boxplot
sns.boxplot(data=boxplot_data)
sns.set_style("whitegrid")

# Add labels for clarity
plt.xticks([0, 1], ['Zinc finger attributions', 'Non-TF-related attributions'])
plt.ylabel('Attribution Score')
#plt.title('Boxplot of zinc finger and other domain attributions')

plt.show()