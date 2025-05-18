import os
import numpy as np
import librosa


dataset_path = r"/Users/dhruvloriya/Desktop/Div Sat/Audio_Speech_Actors_01-24"


def extract_mfcc(file_path, sr=22050, n_mfcc=40, max_pad_len=200):
    """Extract MFCC features while keeping time steps for LSTM."""
    y, sr = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  
   
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]  

    return mfccs.T  


def process_dataset(dataset_path, max_pad_len=200):
    features = []
    labels = []
    
    
    for actor_folder in os.listdir(dataset_path):
        actor_path = os.path.join(dataset_path, actor_folder)
        
        if os.path.isdir(actor_path):  
            for file in os.listdir(actor_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(actor_path, file)
                    
                    try:
                        
                        parts = file.split("-")
                        if len(parts) > 2:
                            emotion_label = int(parts[2])  
                        else:
                            print(f"Skipping {file} due to unexpected filename format")
                            continue
                        
                        
                        mfcc_features = extract_mfcc(file_path, max_pad_len=max_pad_len)
                        
                       
                        features.append(mfcc_features)
                        labels.append(emotion_label)
                    
                    except Exception as e:
                        print(f"Error processing {file}: {e}")

    return np.array(features), np.array(labels)


X, y = process_dataset(dataset_path)


np.save("features.npy", X)
np.save("labels.npy", y)
print(f"Feature extraction completed! Processed {len(X)} files.")
print(f"Features shape: {X.shape} (Expected: (num_samples, timesteps=200, features=40))")
