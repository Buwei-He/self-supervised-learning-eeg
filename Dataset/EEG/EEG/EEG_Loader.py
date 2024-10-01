import os
import pandas as pd
import mne
import numpy as np
from sklearn.model_selection import train_test_split
# from utils.eeg_utils import map_categories_to_numbers, map_numbers_to_categories

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
    print(epochs.get_data().shape)
    return epochs

def filter_patients(patients_list, MMSE_max_A, MMSE_max_F,wanted_class=['A','C','F']):
    # Filter patients based on MMSE for group A and C
    filtered_patients = []
    for patient in patients_list:
        if patient.group in wanted_class:
            if patient.group == 'A' and patient.MMSE <= MMSE_max_A:
                filtered_patients.append(patient)
            elif patient.group == 'F' and patient.MMSE <= MMSE_max_F:
                filtered_patients.append(patient)
            elif patient.group == 'C':
                filtered_patients.append(patient)
    return filtered_patients

def z_score(x):
    # Return the z-score normalisation of x
    return (x-x.mean())/x.std()

def get_train_and_test_data(patients_list, subset_channel_names, duration, sample_rate,test_size):
    data_train = []
    labels_train = []
    data_test = []
    labels_test = []
    wanted_shape = (len(subset_channel_names), int(duration * sample_rate))
    
    # Identify total channel names from a sample subject (assuming uniform channels across subjects)
    total_channel_names = patients_list[0].eeg.info['ch_names']
    index = [total_channel_names.index(ch) for ch in subset_channel_names if ch in total_channel_names]
    
    # Separate patients by groups
    group_dict = {}
    for subject in patients_list:
        if subject.group not in group_dict:
            group_dict[subject.group] = []
        group_dict[subject.group].append(subject)
    
    # Split subjects into training and testing, maintaining balance for each group
    for group, subjects in group_dict.items():
        # Split subjects into training and testing sets
        train_subjects, test_subjects = train_test_split(subjects, test_size=0.25)
        
        # Extract epochs data and labels for training subjects
        for subject in train_subjects:
            for epoch in subject.epochs:
                selected_channels_epoch = epoch[index, :]
                if selected_channels_epoch.shape == wanted_shape:
                    data_train.append(selected_channels_epoch)
                    labels_train.append(subject.group)
                else:
                    print("PROBLEM WITH SHAPE")

        # Extract epochs data and labels for testing subjects
        for subject in test_subjects:
            for epoch in subject.epochs:
                selected_channels_epoch = epoch[index, :]
                if selected_channels_epoch.shape == wanted_shape:
                    data_test.append(selected_channels_epoch)
                    labels_test.append(subject.group)
    
    # Convert lists to numpy arrays
    X_train = np.array(data_train)
    X_test = np.array(data_test)
    y_train = np.array(labels_train)
    y_test = np.array(labels_test)
    
    return X_train, X_test, y_train, y_test


def map_categories_to_numbers(categories):
    category_mapping = {'C': 0, 'A': 1, 'F': 2}
    if isinstance(categories, np.ndarray):
        return np.array([category_mapping[cat] for cat in categories])
    else:
        return category_mapping[categories]
    

def EEG(root_path=os.getcwd(), duration=10, sample_rate=100, overlap_ratio=0.5,test_size=0.25, 
        subset_channel_names=['Cz', 'Pz', 'Fz'], MMSE_max_A=25, MMSE_max_F=30,wanted_class=['A','C','F'],
        normalisation_fun=None #If None then no normalisation, if not None applies this function to eeg data
        ):
    print(f'Current root path (path to EEG dataset): {root_path}')
    participants_file = os.path.join(root_path, 'participants.tsv')
    
    # Load the participants data
    participants_df = load_participants(participants_file) #Create the panda dataframe of the file.

    # Create the patients list
    patients_list = create_patients_list(root_path, participants_df)
    # Downsample the EEG data to 100 Hz and create epochs of 10 seconds
    patients_list_filtered = filter_patients(patients_list, MMSE_max_A, MMSE_max_F,wanted_class)

    if not normalisation_fun:
        # If no normalisation then the function is just the identity function
        normalisation_fun = lambda x: x

    for subject in patients_list_filtered:
        # Apply normalisation subject wise, across all channels
        subject.eeg.apply_function(normalisation_fun, picks='all', channel_wise=False)
        subject.eeg.resample(sample_rate)
        subject.epochs = get_epochs(subject, duration=duration, overlap_ratio=overlap_ratio)
    
    # Get train and test data
    X_train, X_test, y_train, y_test = get_train_and_test_data(patients_list_filtered, subset_channel_names, duration, sample_rate,test_size)

    Data = {}
    Data['train_data'] = X_train
    Data['train_label'] = map_categories_to_numbers(y_train) # ['C','A','F'] -> [0, 1, 2]
    Data['test_data'] = X_test
    Data['test_label'] = map_categories_to_numbers(y_test)
    print(X_train.shape,X_test.shape)
    # if not os.path.exists(root_path):
    #     os.makedirs(root_path)
        
    np.save(os.path.join(root_path, 'EEG.npy'), Data, allow_pickle=True)

if __name__ == '__main__':
    root_path = './Dataset/EEG/EEG'
    EEG(root_path, duration=10, sample_rate=100, overlap_ratio=0, subset_channel_names=['Cz', 'Pz'],test_size=0.25, MMSE_max_A=30, MMSE_max_F=30,wanted_class=['C','F','A'],
        normalisation_fun=z_score) # 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz'
    