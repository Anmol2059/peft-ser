import json
import yaml
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == '__main__':

    # Read data path
    with open("../config/config.yml", "r") as stream:
        config = yaml.safe_load(stream)
    data_path   = Path(config["data_dir"]["crema_d"])
    split_path = Path(config["project_dir"]).joinpath("train_split")
    audio_path = Path(config["project_dir"]).joinpath("audio")

    # Iterate over different splits
    for fold_idx in range(1, 6):  # For each fold in the K-fold cross-validation
        # Read split file
        with open(str(split_path.joinpath(f'crema_d_fold{fold_idx}.json')), "r") as f: 
            split_dict = json.load(f)

        for split in ['train', 'dev', 'test']:
            Path.mkdir(audio_path.joinpath(f'crema_d_fold{fold_idx}'), parents=True, exist_ok=True)
            for idx in tqdm(range(len(split_dict[split]))):
                # Read data: speaker_id, path
                data = split_dict[split][idx]
                speaker_id, file_path = data[1], data[2]

                # Read wavforms
                waveform, sample_rate = torchaudio.load(str(file_path))

                # If the waveform has multiple channels, compute the mean across channels to create a single-channel waveform.
                if waveform.shape[0] != 1:
                    waveform = torch.mean(waveform, dim=0).unsqueeze(0)

                # If the sample rate is not 16000 Hz, resample the waveform to 16000 Hz.
                if sample_rate != 16000:
                    transform_model = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = transform_model(waveform)

                # Set the output path for the processed audio file based on the dataset and other information.
                output_path = audio_path.joinpath(f'crema_d_fold{fold_idx}', file_path.split('/')[-1])
                
                # Save the audio file with desired sampling frequency
                torchaudio.save(str(output_path), waveform, 16000)
                split_dict[split][idx][2] = str(output_path)

            # Logging the stats for train/dev/test
            logging.info(f'-------------------------------------------------------')
            logging.info(f'Preprocess audio for crema_d_fold{fold_idx} dataset')
            for split in ['train', 'dev', 'test']: logging.info(f'Split {split}: Number of files {len(split_dict[split])}')
            logging.info(f'-------------------------------------------------------')

        # Save the updated split_dict with new file paths
        with open(str(split_path.joinpath(f'crema_d_fold{fold_idx}_processed.json')), "w") as f:
            json.dump(split_dict, f, indent=4)