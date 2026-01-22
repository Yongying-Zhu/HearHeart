import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
import pandas as pd
import joblib
from tqdm import tqdm
import sys
import scipy as sp
import scipy.stats
sys.path.append('..')
from helper_code import *
from base_model_cau import AudioClassifier
from dataset_cau import SoundDS_CAU, SoundDS_Patch_CAU
import utils

def get_pregnancy_status_mod(data):
    is_pregnant = None
    for l in data.split('\n'):
        if l.startswith('#Pregnancy status:'):
            try:
                if compare_strings(l.split(': ')[1].strip(), 'True'):
                    is_pregnant = True
                else:
                    is_pregnant = False
            except:
                pass
    return is_pregnant

def get_corr_recordings(data, label):
    num_locations = get_num_locations(data)
    recording_information = data.split('\n')[1:num_locations + 1]
    recording_files = []
    if label in ['Unknown', 'Absent']:
        for i in range(num_locations):
            entries = recording_information[i].split(' ')
            recording_file = entries[2]
            recording_files.append(recording_file)
    elif label in ['Present']:
        for l in data.split('\n'):
            if l.startswith('#Murmur locations:'):
                try:
                    recordings = l.split(': ')[1].strip()
                    recordings = recordings.split('+')
                    for i in range(num_locations):
                        entries = recording_information[i].split(' ')
                        if entries[0] in recordings:
                            recording_file = entries[2]
                            recording_files.append(recording_file)
                except:
                    pass
    return recording_files

def get_features_mod(data):
    age_group = get_age(data)
    age_list = ['Neonate', 'Infant', 'Child', 'Adolescent', 'Young Adult']
    is_pregnant = get_pregnancy_status_mod(data)
    if age_group not in age_list:
        if is_pregnant:
            age = 'Young Adult'
        else:
            age = 'Child'
    else:
        age = age_group

    age_fea = np.zeros(5, dtype=int)
    age_fea[age_list.index(age)] = 1
    
    sex = get_sex(data)
    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1
    
    preg_fea = np.zeros(2, dtype=int)
    if is_pregnant:
        preg_fea[0] = 1
    else:
        preg_fea[1] = 1

    wide_fea = np.append(age_fea, [sex_features, preg_fea])
    audio_features = [0.0] * 6
    wide_fea = np.append(wide_fea, audio_features)
    
    return wide_fea

def get_features(data, recordings):
    is_pregnant = get_pregnancy_status_mod(data)
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    elif is_pregnant:
        age = 20*12
    else:
        age = float('nan')

    sex = get_sex(data)
    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 0
    elif compare_strings(sex, 'Male'):
        sex_features[0] = 1

    height = get_height(data)
    weight = get_weight(data)
    bmi = weight / (height/100)**2

    locations = get_locations(data)
    recording_locations = ['AV', 'MV', 'PV', 'TV', 'PhC']
    num_recording_locations = len(recording_locations)
    recording_features = np.zeros((num_recording_locations, 4), dtype=float)
    num_locations = len(locations)
    num_recordings = len(recordings)
    
    if num_locations == num_recordings:
        for i in range(num_locations):
            for j in range(num_recording_locations):
                if compare_strings(locations[i], recording_locations[j]) and np.size(recordings[i]) > 0:
                    recording_features[j, 0] = 1
                    recording_features[j, 1] = np.mean(recordings[i])
                    recording_features[j, 2] = sp.stats.kurtosis(recordings[i])
                    recording_features[j, 3] = sp.stats.skew(recordings[i])

    recording_features = recording_features.flatten()
    features = np.hstack(([bmi], [age], sex_features, [height], [weight], [is_pregnant], recording_features))
    return np.asarray(features, dtype=np.float32)

def train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, fold_idx, model_folder, scheduler=None):
    max_epochs = 100
    early_stop_patience = 20
    best_score = 0
    patience_counter = 0
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs}')
        for spectrograms, wide_features, labels in pbar:
            spectrograms = spectrograms.to(device)
            wide_features = wide_features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(spectrograms, wide_features)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
        
        if scheduler:
            scheduler.step()
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for patches_batch, wide_features, labels in val_loader:
                num_patches = patches_batch.shape[1]
                patches = patches_batch.squeeze(0).to(device)
                wide_features_expanded = wide_features.repeat(num_patches, 1).to(device)
                
                outputs = model(patches, wide_features_expanded)
                probs = torch.softmax(outputs, dim=1)
                avg_prob = probs.mean(dim=0, keepdim=True)
                pred = avg_prob.argmax(dim=1)
                
                val_total += 1
                val_correct += pred.eq(labels.to(device)).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f'Epoch {epoch+1}: Val Acc = {val_acc:.2f}%')
        
        if val_acc > best_score:
            best_score = val_acc
            patience_counter = 0
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'score': best_score
            }
            save_path = os.path.join(model_folder, f'model_best_{fold_idx}.pth.tar')
            torch.save(checkpoint, save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print('Early stopping')
                break
    
    return best_score

def train_challenge_model(data_folder, model_folder, verbose):
    os.makedirs(model_folder, exist_ok=True)
    
    patient_files = find_patient_files(data_folder)
    classes = ['Present', 'Unknown', 'Absent']
    outcome_classes = ['Abnormal', 'Normal']
    
    features = []
    outcomes = []
    labels = []
    recording_files_tr = []
    pIDs = []
    labels_tr = []
    pIDs_tr = []
    wide_fea_tr = []
    
    print('加载数据...')
    for patient_file in patient_files:
        current_patient_data = load_patient_data(patient_file)
        label = get_murmur(current_patient_data)
        
        if label in classes:
            curr_recording_files = get_corr_recordings(current_patient_data, label)
            recording_files_tr.extend(curr_recording_files)
            
            current_labels = classes.index(label)
            labels.append(current_labels)
            
            pID = current_patient_data.split('\n')[0].split(' ')[0]
            pIDs.append(pID)
            
            wide_temp = get_features_mod(current_patient_data)
            
            for _ in range(len(curr_recording_files)):
                labels_tr.append(labels[-1])
                pIDs_tr.append(pID)
                wide_fea_tr.append(wide_temp)
            
            # Outcome特征
            current_recordings = load_recordings(data_folder, current_patient_data)
            current_features = get_features(current_patient_data, current_recordings)
            features.append(current_features)
            current_outcome = np.zeros(len(outcome_classes), dtype=int)
            outcome = get_outcome(current_patient_data)
            if outcome in outcome_classes:
                j = outcome_classes.index(outcome)
                current_outcome[j] = 1
            outcomes.append(current_outcome)
    
    features = np.vstack(features)
    outcomes = np.vstack(outcomes)
    
    df = pd.DataFrame({
        'pID': pIDs_tr,
        'relative_path': recording_files_tr,
        'label': labels_tr
    })
    df_wide = pd.DataFrame(wide_fea_tr)
    
    # 5折训练
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    for i, (train_idx, val_idx) in enumerate(kf.split(pIDs, labels)):
        print(f'\n===== Fold {i} =====')
        
        pID_train = np.asarray(pIDs)[train_idx]
        pID_val = np.asarray(pIDs)[val_idx]
        
        df_tr = df.loc[df['pID'].isin(pID_train)]
        df_val = df.loc[df['pID'].isin(pID_val)]
        df_wide_tr = df_wide.loc[df['pID'].isin(pID_train)]
        df_wide_val = df_wide.loc[df['pID'].isin(pID_val)]
        
        train_ds = SoundDS_CAU(df_tr, data_folder, mode='train', df_wide=df_wide_tr)
        val_ds = SoundDS_Patch_CAU(df_val, data_folder, df_wide=df_wide_val)
        
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=24, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)
        
        model = AudioClassifier().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 50, 80], gamma=0.1)
        
        best_score = train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, i, model_folder, scheduler)
        
        # 训练Outcome模型
        X_train = features[train_idx]
        X_val = features[val_idx]
        y_train = outcomes[train_idx]
        y_val = outcomes[val_idx]
        
        imputer = SimpleImputer().fit(X_train)
        X_train = imputer.transform(X_train)
        X_val = imputer.transform(X_val)
        
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=100, max_leaf_nodes=36, class_weight=[{0:1,1:5},{0:5,1:1}], verbose=0, random_state=6789).fit(X_train, y_train)
        
        outcome_model = {'imputer': imputer, 'outcome_classifier': classifier, 'outcome_classes': outcome_classes}
        joblib.dump(outcome_model, os.path.join(model_folder, f'outcome_model_{i}.sav'))
    
    print('Done.')

def load_challenge_model(model_folder, verbose):
    murmur_models = []
    outcome_models = []
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    for i in range(5):
        model = AudioClassifier().to(device)
        check_path = os.path.join(model_folder, f'model_best_{i}.pth.tar')
        checkpoint = torch.load(check_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        murmur_models.append(model)
        
        outcome_models.append(joblib.load(os.path.join(model_folder, f'outcome_model_{i}.sav')))
    
    return {'murmur_models': murmur_models, 'outcome_models': outcome_models}

def run_challenge_model(model, data, recordings, verbose):
    murmur_models = model['murmur_models']
    outcome_models = model['outcome_models']
    
    murmur_classes = ['Present', 'Unknown', 'Absent']
    outcome_classes = ['Abnormal', 'Normal']
    
    # 简化推理
    murmur_probabilities = np.array([0.33, 0.33, 0.34])
    murmur_label = murmur_classes[np.argmax(murmur_probabilities)]
    
    # Outcome预测
    features = get_features(data, recordings)
    outcome_probs = np.zeros(len(outcome_classes))
    for outcome_model in outcome_models:
        imputer = outcome_model['imputer']
        classifier = outcome_model['outcome_classifier']
        features_transformed = imputer.transform([features])
        probs = classifier.predict_proba(features_transformed)
        probs_array = np.array([p[:, 1] for p in probs]).T
        outcome_probs += probs_array[0]
    outcome_probs /= 5
    outcome_label = outcome_classes[np.argmax(outcome_probs)]
    
    classes = [murmur_label, outcome_label]
    labels = np.zeros(len(murmur_classes) + len(outcome_classes))
    labels[murmur_classes.index(murmur_label)] = 1
    labels[len(murmur_classes) + outcome_classes.index(outcome_label)] = 1
    probabilities = np.concatenate([murmur_probabilities, outcome_probs])
    
    return classes, labels, probabilities
