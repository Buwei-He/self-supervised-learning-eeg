import os
import warnings
import itertools
import pandas as pd
import mne
import numpy as np
from collections import Counter
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
               sample_rate,
               duration=10, #in seconds
               overlap_ratio=0.5,
               subset_channel_names = 'all',
               crop = 60, # in seconds
               ):
    eeg = subject.eeg
    # Rename boundary annotations to bad_boundary to drop epochs overlapping with boundary annotations
    rename_dict = {'boundary': 'bad_boundary'}
    if 'boundary' in eeg.annotations.description:
        eeg.annotations.rename(rename_dict)
    # Crop and resample raw data
    tmin = crop
    tmax = eeg.times[-1] - crop
    eeg.crop(tmin=tmin,tmax=tmax).resample(sample_rate)
    # Create epochs
    overlap = int(overlap_ratio * duration)
    epochs = mne.make_fixed_length_epochs(eeg, duration=duration, overlap=overlap, reject_by_annotation=True, preload=True, verbose=False) 
    # Keep only wanted channels
    epochs = epochs.pick(subset_channel_names,verbose=False)
    epochs.drop_bad(verbose=False)
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

def get_train_and_test_data(patients_list, subset_channel_names, duration, sample_rate, val_ratio, test_ratio, seed, max_train_samples, is_analysis):
    wanted_shape = (len(subset_channel_names), int(duration * sample_rate))
    
    # Get list of subjects' group
    groups = [subject.group for subject in patients_list]

    Data = {}
    
    # Split subjects into training+validation and testing sets
    train_val_subjects, test_subjects, groups_train_val, _ = train_test_split(patients_list, groups, test_size=test_ratio, stratify=groups, random_state=seed)
    if val_ratio == 0:
        train_subjects = train_val_subjects
        split_subjects = {'train':train_subjects, 'test':test_subjects}
        Data['val_data'] = np.empty(shape=(0,0))
        Data['val_label'] = np.empty(shape=(0,0))
    else:
        # Split subjects into training and valisation sets
        train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=val_ratio, stratify=groups_train_val, random_state=seed)
        split_subjects = {'train':train_subjects, 'val':val_subjects, 'test':test_subjects}
    
    for split in split_subjects.keys():
        max_samples = max_train_samples if split == 'train' else None
        Data[f"{split}_data"], Data[f"{split}_label"], Data[f"{split}_ids"] = get_data_labels(split_subjects[split], wanted_shape, split_name=split, max_samples=max_samples)
        if not is_analysis:
            Data.pop(f'{split}_ids')
    
    return Data

def get_data_labels(subjects, wanted_shape, split_name='', max_samples=None):
    """ Get for each split (train, val, test) the data and labels as numpy arrays and print the statistics"""
    # Print stats of subjects
    count_groups = Counter([sub.group for sub in subjects])
    count_epochs = {}
    for group in count_groups.keys():
        count_epochs[group] = sum([len(sub.epochs) for sub in subjects if sub.group==group])
    if max_samples is None: max_samples = float('inf')
    # Get minimal number of epochs for one group or max number of samples per class
    min_n_epochs = min(min(count_epochs.values()), max_samples)
    # Select min_n_epochs for each group
    data, labels, ids = [], [], []
    new_count_epochs = {}
    print(f"{split_name}: {len(subjects)} sub =", end='') 
    for group in count_groups.keys():
        data_group = [sub.epochs.get_data(copy=False, verbose=False) for sub in subjects if sub.group==group]
        data_group = np.concatenate(data_group, axis=0)
        ids_group = [[sub.participant_id]*len(sub.epochs) for sub in subjects if sub.group==group]
        ids_group = np.array(list(itertools.chain.from_iterable(ids_group)))
        if count_epochs[group] > min_n_epochs:
            # Select data
            filtered_data_idx = np.random.choice(range(len(data_group)), size=min_n_epochs, replace=False)
        else:
            filtered_data_idx = range(len(data_group))
        data_group = data_group[filtered_data_idx]
        ids_group = ids_group[filtered_data_idx]
        if len(data) == 0:
            data = data_group
            ids = ids_group
        else:
            data = np.concatenate((data, data_group))
            ids = np.concatenate((ids, ids_group))
        labels.extend([[group]*len(filtered_data_idx)])
        new_count_epochs[group] = len(filtered_data_idx)
        print(f" {count_groups[group]} {group} {count_epochs[group],new_count_epochs[group]}", end='')
    print("\n")
    # Get epochs and labels for each subject
    labels = map_categories_to_numbers(np.concatenate(labels, axis=0))
    # Check shape
    if (data.shape[0] != labels.shape[0]) or (data.shape[1] != wanted_shape[0]) or (data.shape[2] != wanted_shape[1]) :
        print(data.shape, labels.shape, wanted_shape)
        raise ValueError("Problem in input shape, check code.")
    return data, labels, ids

def map_categories_to_numbers(categories):
    category_mapping = {'C': 0, 'A': 1, 'F': 2}
    if isinstance(categories, np.ndarray):
        return np.array([category_mapping[cat] for cat in categories])
    else:
        return category_mapping[categories]

def EEG(root_path=os.getcwd(), duration=10, sample_rate=100, overlap_ratio=0.5, val_ratio=0.1, test_ratio=0.1, 
        subset_channel_names=['Cz', 'Pz', 'Fz'], MMSE_max_A=25, MMSE_max_F=30,wanted_class=['A','C','F'],
        max_train_samples=None, # Max number of samples to use for each class in training
        normalisation_fun=None, #If None then no normalisation, if not None applies this function to eeg data
        seed=1234, return_data=False,
        is_analysis=False, #If True then returns other demographics info
        crop=60, #Duration (in s) to crop at start and end of recording
        reject_threshold=6.14, #Drop segments with peak-to-peak amplitude higher than this threshold
        flat_threshold=1, #Drop segments with peak-to-peak amplitude lower than this threshold
        ):

    print(f'Current root path (path to EEG dataset): {root_path}')
    participants_file = os.path.join(root_path, 'participants.tsv')
    
    # Load the participants data
    participants_df = load_participants(participants_file) #Create the panda dataframe of the file.

    # Create the patients list
    patients_list = create_patients_list(root_path, participants_df)
    patients_list_filtered = filter_patients(patients_list, MMSE_max_A, MMSE_max_F,wanted_class)

    if not normalisation_fun:
        # If no normalisation then the function is just the identity function
        normalisation_fun = lambda x: x

    # Normalise and downsample the EEG data to 100 Hz and create epochs of 10 seconds
    for subject in patients_list_filtered:
        # Apply normalisation for each segment and channel
        subject.eeg.apply_function(normalisation_fun, picks='all', channel_wise=True)
        subject.epochs = get_epochs(subject, sample_rate=sample_rate, duration=duration, overlap_ratio=overlap_ratio, subset_channel_names=subset_channel_names, crop=crop)
        # Drop outliers
        subject.epochs.drop_bad(reject={'eeg':reject_threshold}, flat={'eeg':flat_threshold}, verbose=0)
        
    # Get train and test data
    Data = get_train_and_test_data(patients_list_filtered, subset_channel_names, duration, 
                                 sample_rate, val_ratio, test_ratio, seed, max_train_samples, is_analysis)

    print(Data['train_data'].shape, Data['val_data'].shape, Data['test_data'].shape)
    # if not os.path.exists(root_path):
    #     os.makedirs(root_path)
        
    np.save(os.path.join(root_path, 'EEG.npy'), Data, allow_pickle=True)

    if return_data:
        return Data

if __name__ == '__main__':
    root_path = './Dataset/EEG/EEG'
    EEG(root_path, duration=10, sample_rate=100, overlap_ratio=0, subset_channel_names=['Cz', 'Pz'],
        val_ratio=0.1, test_ratio=0.1, MMSE_max_A=30, MMSE_max_F=30, wanted_class=['C','F','A'],
        normalisation_fun=z_score, seed=2024, return_data=False) # 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz'
    