import os
import pandas as pd
import mne
import numpy as np
from sklearn.model_selection import train_test_split

from utils.eeg_utils import map_categories_to_numbers, map_numbers_to_categories

class Patient:
    def __init__(self, participant_id, gender, age, group, MMSE, eeg=None):
        self.participant_id = participant_id
        self.gender = gender
        self.age = age
        self.group = group
        self.MMSE = MMSE
        self.eeg = eeg # EEG are of shape (2,19) Why 2 i don't know, 19 because we have 19 channels
        self.epochs=[]

    def __repr__(self):
        return f"Patient(ID={self.participant_id}, Age={self.age}, Group={self.group}, MMSE={self.MMSE})"

def load_participants(file_path):
    # Read the tsv file into a pandas DataFrame
    participants_df = pd.read_csv(file_path, sep='\t')
    return participants_df


def load_eeg(root_path, participant_id):
    # Define the EEG file path
    eeg_file = os.path.join(root_path, 'derivatives', participant_id, 'eeg', f"{participant_id}_task-eyesclosed_eeg.set")
    
    if os.path.exists(eeg_file):
        # Load the EEG data using mne library
        try:
            eeg_data = mne.io.read_raw_eeglab(eeg_file, preload=True)
            return eeg_data
        except Exception as e:
            print(f"Could not load EEG file for participant {participant_id}: {e}")
            return None
    else:
        print(f"EEG file for participant {participant_id} not found.")
        return None
    
def create_patients_list(root_path, participants_df):
    patients_list = []
    
    for index, row in participants_df.iterrows():
        participant_id = row['participant_id']
        gender = row['Gender']
        age = row['Age']
        group = row['Group']
        MMSE = row['MMSE']
        
        # Load the EEG file for this participant
        eeg = load_eeg(root_path, participant_id)
        
        # Create a Patient object
        patient = Patient(participant_id, gender, age, group, MMSE, eeg)
        
        
        # Append the Patient object to the list
        patients_list.append(patient)
        
    return patients_list

def get_epochs(subject, 
               duration=10, #in seconds
               overlap_ratio=0.5
               ):
    eeg = subject.eeg
    # Rename boundary annotations to bad_boundary to drop epochs overlapping with boundary annotations
    rename_dict = {'boundary': 'bad_boundary'}
    if 'boundary' in eeg.annotations.description:
        eeg.annotations.rename(rename_dict)
    # Create epochs
    overlap = int(overlap_ratio * duration)
    epochs = mne.make_fixed_length_epochs(eeg, duration=duration, overlap=overlap, reject_by_annotation=True, preload=False, verbose=0) 
    epochs.drop_bad(verbose=0)
    return epochs

def get_train_and_test_data(patients_list, subset_channel_names, duration, sample_rate, label_filter):
    data=[]
    labels=[]
    wanted_shape=(len(subset_channel_names),1000)
    total_channel_names=patients_list[85].eeg.info['ch_names']
    index = [total_channel_names.index(ch) for ch in subset_channel_names if ch in total_channel_names]
    
    for subject in patients_list:
        # Loop through each epoch in the subject's epochs
        if subject.group not in label_filter:
            for epoch in subject.epochs:
                # Select only the required channels using the indices
                selected_channels_epoch = epoch[index, :]
                
                # Check if the selected data matches the wanted shape
                if selected_channels_epoch.shape == (len(subset_channel_names), int(duration*sample_rate)):
                    data.append(selected_channels_epoch)
                    labels.append(map_categories_to_numbers(subject.group))
            # else:
                # print("Problem")

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)
    return( np.array(X_train), np.array(X_test), np.array(y_train),np.array(y_test))

def EEG(root_path=os.getcwd(), duration=10, sample_rate=100, overlap_ratio=0.5, subset_channel_names=['Cz', 'Pz', 'Fz'], label_filter=[]):
    print(f'Current root path (path to EEG dataset): {root_path}')
    participants_file = os.path.join(root_path, 'participants.tsv')
    
    # Load the participants data
    participants_df = load_participants(participants_file) #Create the panda dataframe of the file.

    # Create the patients list
    patients_list = create_patients_list(root_path, participants_df)
    # Downsample the EEG data to 100 Hz and create epochs of 10 seconds
    for subject in patients_list:
        subject.eeg.resample(sample_rate) # downsample to 100hz; which method?
        subject.epochs = get_epochs(subject, duration=duration, overlap_ratio=overlap_ratio)
    
    X_train, X_test, y_train, y_test = get_train_and_test_data(patients_list, subset_channel_names, duration, sample_rate, label_filter)
    Data = {}
    Data['train_data'] = X_train
    Data['train_label'] = y_train
    Data['test_data'] = X_test
    Data['test_label'] = y_test

    # if not os.path.exists(root_path):
    #     os.makedirs(root_path)
        
    np.save(os.path.join(root_path, 'EEG.npy'), Data, allow_pickle=True)

if __name__ == '__main__':
    root_path = './Dataset/EEG/EEG/'
    EEG(root_path,
        duration=10, 
        sample_rate=100, 
        overlap_ratio=0, 
        subset_channel_names=['Fp1', 'Fp2'], # 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz'
        label_filter=['A']) # ['C','A','F'] -> [0, 1, 2]; Use list of letter(s) here;