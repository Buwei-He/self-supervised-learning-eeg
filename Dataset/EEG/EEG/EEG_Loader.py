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
        Data['val_info'] = np.empty(shape=(0,0))
    else:
        # Split subjects into training and valisation sets
        train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=val_ratio, stratify=groups_train_val, random_state=seed)
        split_subjects = {'train':train_subjects, 'val':val_subjects, 'test':test_subjects}
    
    for split in split_subjects.keys():
        max_samples = max_train_samples if split == 'train' else None
        Data[f"{split}_data"], Data[f"{split}_label"], Data[f"{split}_info"] = get_data_labels(split_subjects[split], wanted_shape, split_name=split, max_samples=max_samples)
    
    return Data

def k_fold_split(patients_list, nb_k_fold, seed):

    # Get list of subjects' group
    groups = [subject.group for subject in patients_list]

    # TODO: 
    # 1. save the _temp_data (contains all the data) for quick loading
    # 2. create method to split dataset by index

    _data = {}
    _data['split'] = []
    _data['patient'] = np.array(patients_list)
    _data['groups'] = np.array(groups)

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=int(nb_k_fold), random_state=seed, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(patients_list, groups)):
        # TODO: add index to dict, and save to data['split']
        _data['split'].append({'train': train_index, 'test': test_index})
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
    
    return _data

def get_k_fold_train_and_test_data(_data, wanted_shape, val_ratio, k_fold_cnt, seed, max_train_samples):

    # TODO:
    # 1. load data from .npy file
    # 2. split full set to train/test using index
    # 3. split train set to train/val using index
    # 4. return 

    # Split subjects into training+validation and testing sets
    Data = {}
    train_val_index, test_index = _data['split'][k_fold_cnt-1]['train'], _data['split'][k_fold_cnt-1]['test']
    train_val_subjects, test_subjects = _data['patient'][train_val_index], _data['patient'][test_index]
    groups_train_val = _data['groups'][train_val_index]
    if val_ratio == 0:
        train_subjects = train_val_subjects
        split_subjects = {'train':train_subjects, 'test':test_subjects}
        Data['val_data'] = np.empty(shape=(0,0))
        Data['val_label'] = np.empty(shape=(0,0))
        Data['val_info'] = np.empty(shape=(0,0))
    else:
        # Split subjects into training and valisation sets
        train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=val_ratio, stratify=groups_train_val, random_state=seed)
        split_subjects = {'train':train_subjects, 'val':val_subjects, 'test':test_subjects}
    
    for split in split_subjects.keys():
        max_samples = max_train_samples if split == 'train' else None
        Data[f"{split}_data"], Data[f"{split}_label"], Data[f"{split}_info"] = get_data_labels(split_subjects[split], wanted_shape, split_name=split, max_samples=max_samples)
    
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
    data, labels, info = [], [], []
    new_count_epochs = {}

    for group in count_groups.keys():
        data_group, info_group = [], []
        for sub in subjects:
            if sub.group == group:
                data_group.append(sub.epochs.get_data(copy=False, verbose=False))
                info_group.append(np.array([np.array([int(sub.participant_id.split('-')[1])] * sub.epochs.events.shape[0]), 
                                 sub.epochs.events[:, 0]]).T)
        # data_group = [sub.epochs.get_data(copy=False, verbose=False) for sub in subjects if sub.group==group]
        data_group = np.concatenate(data_group, axis=0)
        info_group = np.concatenate(info_group, axis=0)

        if count_epochs[group] > min_n_epochs:
            # Select data
            filtered_data_idx = np.random.choice(range(len(data_group)), size=min_n_epochs, replace=False)
        else:
            filtered_data_idx = range(len(data_group))

        data_group = data_group[filtered_data_idx]
        info_group = info_group[filtered_data_idx]

        if len(data) == 0:
            data = data_group
            info = info_group
        else:
            data = np.concatenate((data, data_group))
            info = np.concatenate((info, info_group))
        
        labels.extend([[group]*len(filtered_data_idx)])
        new_count_epochs[group] = len(filtered_data_idx)

    # print(f'''{split_name}: {len(subjects)} sub = {count_groups['A']} AD ({count_epochs['A'],new_count_epochs['A']}), {count_groups['F']} FTD ({count_epochs['F'],new_count_epochs['F']}), {count_groups['C']} HC ({count_epochs['C'], new_count_epochs['C']})''')
    
    class_labels = {'A': 'AD', 'F': 'FTD', 'C': 'HC'}
    class_count_str = []

    for class_key, class_name in class_labels.items():
        if class_key in count_groups:
            class_count_str.append(f"{count_groups[class_key]} {class_name} ({count_epochs[class_key]}, {new_count_epochs[class_key]})")

    print(f"{split_name}: {len(subjects)} sub = {', '.join(class_count_str)}")

    # Get epochs and labels for each subject
    labels = map_categories_to_numbers(np.concatenate(labels, axis=0))
    # Check shape
    if (data.shape[0] != labels.shape[0]) or (data.shape[1] != wanted_shape[0]) or (data.shape[2] != wanted_shape[1]) :
        print(data.shape, labels.shape, wanted_shape)
        raise ValueError("Problem in input shape, check code.")
    return data, labels, info

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
        nb_k_fold=0, k_fold_cnt=1, create_data=False,
        seed=1234, return_data=False,
        is_analysis=False, #If True then returns other demographics info
        crop=60, #Duration (in s) to crop at start and end of recording
        reject_threshold=6.14, #Drop segments with peak-to-peak amplitude higher than this threshold
        flat_threshold=1, #Drop segments with peak-to-peak amplitude lower than this threshold
        ):
    
    print(f'Current root path (path to EEG dataset): {root_path}')

    if create_data: # create new dataset, instead of load from existing file
        participants_file = os.path.join(root_path, 'participants.tsv')
        
        # Load the participants data
        participants_df = load_participants(participants_file) #Create the panda dataframe of the file.

        # Create the patients list
        patients_list = create_patients_list(root_path, participants_df)
        patients_list_filtered = filter_patients(patients_list, MMSE_max_A, MMSE_max_F, wanted_class)

        if not normalisation_fun:
            # If no normalisation then the function is just the identity function
            normalisation_fun = lambda x: x

        # Normalise and downsample the EEG data to 100 Hz and create epochs of 10 seconds
        for subject in patients_list_filtered:
            # Apply normalisation subject wise, across all channels
            subject.eeg.apply_function(normalisation_fun, picks='all', channel_wise=False)
            subject.epochs = get_epochs(subject, sample_rate=sample_rate, duration=duration, overlap_ratio=overlap_ratio, subset_channel_names=subset_channel_names)
            # Drop outliers
            if reject_threshold != -1:
                if flat_threshold != -1:
                    subject.epochs.drop_bad(reject={'eeg':reject_threshold}, flat={'eeg':flat_threshold}, verbose=0)
                else:
                    subject.epochs.drop_bad(reject={'eeg':reject_threshold}, verbose=0)
            if flat_threshold != -1:
                subject.epochs.drop_bad(flat={'eeg':flat_threshold}, verbose=0)
        
        # Get train and test data
        if nb_k_fold < 1: # normal mode
            Data = get_train_and_test_data(patients_list_filtered, subset_channel_names, duration, 
                                        sample_rate, val_ratio, test_ratio, seed, max_train_samples)
            np.save(os.path.join(root_path, 'EEG.npy'), Data, allow_pickle=True)
        else: # using k-fold cross validation, create new dataset
            k_fold_data = k_fold_split(patients_list_filtered, nb_k_fold, seed)
            wanted_shape = (len(subset_channel_names), int(duration * sample_rate))
            Data = get_k_fold_train_and_test_data(k_fold_data, wanted_shape, val_ratio, k_fold_cnt, seed, max_train_samples)
            np.save(os.path.join(root_path, 'EEG_k_fold.npy'), np.array(k_fold_data), allow_pickle=True)
    else: # using k-fold cross validation, and load from existing file
        k_fold_data = np.load(os.path.join(root_path, 'EEG_k_fold.npy'), allow_pickle=True).item()
        wanted_shape = (len(subset_channel_names), int(duration * sample_rate))
        Data = get_k_fold_train_and_test_data(k_fold_data, wanted_shape, val_ratio, k_fold_cnt, seed, max_train_samples)

    print(Data['train_data'].shape, Data['val_data'].shape, Data['test_data'].shape)
    
    if return_data:
        return Data

if __name__ == '__main__':
    root_path = './Dataset/EEG/EEG'
    EEG(root_path, duration=10, sample_rate=100, overlap_ratio=0, subset_channel_names=['Cz', 'Pz'],
        val_ratio=0.1, test_ratio=0.1, MMSE_max_A=30, MMSE_max_F=30, wanted_class=['C','F','A'],
        normalisation_fun=z_score, 
        nb_k_fold=0, k_fold_cnt=1, create_data=False,
        seed=2024, return_data=False) # 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz'
    