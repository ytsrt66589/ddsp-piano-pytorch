import os
import librosa 
import note_seq
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from ddsp_piano.utils.midi_encoders import MIDIRoll2Conditioning

seq_lib = note_seq.sequences_lib

def load_midi_as_note_sequence(mid_path):
    # Read MIDI file
    note_sequence = note_seq.midi_io.midi_file_to_note_sequence(mid_path)
    # Extend offset with sustain pedal
    note_sequence = note_seq.apply_sustain_control_changes(note_sequence)
    return note_sequence

def check_files_exist(path):
    suffix = ['audio.npy', 'conditioning.npy', 'pedal.npy', 'polyphony.npy', 'piano_model.npy']
    for f in suffix:
        if os.path.exists(os.path.join(path, f)) == False:
            return False
    return True

def save_files(path, audio, conditioning, pedal, polyphony, piano_model):
    np.save(os.path.join(path, 'audio.npy'), audio)
    np.save(os.path.join(path, 'conditioning.npy'), conditioning)
    np.save(os.path.join(path, 'pedal.npy'), pedal)
    np.save(os.path.join(path, 'polyphony.npy'), polyphony)
    np.save(os.path.join(path, 'piano_model.npy'), piano_model)

def read_data_from_cache(path):
    return np.load(os.path.join(path, 'audio.npy')), np.load(os.path.join(path, 'conditioning.npy')), np.load(os.path.join(path, 'pedal.npy')), np.load(os.path.join(path, 'polyphony.npy')), np.load(os.path.join(path, 'piano_model.npy'))

class MaestroDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        dataframe,
        piano_models,
        max_polyphony=16,
        split='train',
        data_path=None,
        cache_data_path=None,
        device='cpu'):
        """
        dataframe: metadata read from csv file
        piano_models: different years of piano ex. 2011
        split: train, valid, test
        """
        self.split = split
        self.data_path = data_path
        self.dataframe = dataframe
        self.piano_models = torch.tensor(piano_models, requires_grad=False)#.to(device)

        # Pre-preparing data
        self.files_path = self.extract_data_path()
        self.data = []
        for audio_path, midi_path, piano_model in tqdm(self.files_path, desc='Loading group %s' % self.split):
            piano_model = np.array(piano_model)
            if self.split == 'train':
                cache_data_prefix = os.path.join( cache_data_path, audio_path.split('/')[2], audio_path.split('/')[3] )
            elif self.split == 'validation':
                cache_data_path = 'data_cache_val'
                cache_data_prefix = os.path.join( cache_data_path, audio_path.split('/')[2], audio_path.split('/')[3] )
            os.makedirs( cache_data_prefix, exist_ok=True)
            if check_files_exist(cache_data_prefix):
                audio, conditioning, pedal, polyphony, piano_model = read_data_from_cache(cache_data_prefix)
                num_segments = len(conditioning)
                for i in range(num_segments):
                    data = dict(
                        audio=audio[i, :],
                        conditioning=conditioning[i, :, :, :],
                        pedal=pedal[i, :, :],
                        piano_model=piano_model
                    )
                    current_maximum_polyphony = np.max(polyphony[i, :])
                    if current_maximum_polyphony > max_polyphony:
                        continue
                    self.data.append(data)
                continue
            audio, conditioning, pedal, polyphony, num_segments = self.load_data(audio_path, midi_path, max_polyphony=max_polyphony)
            for i in range(num_segments):
                data = dict(
                    audio=audio[i, :],
                    conditioning=conditioning[i, :, :, :],
                    pedal=pedal[i, :, :],
                    piano_model=piano_model
                )
                current_maximum_polyphony = np.max(polyphony[i, :])
                if current_maximum_polyphony > max_polyphony:
                    continue
                self.data.append(data)
            save_files(cache_data_prefix, audio, conditioning, pedal, polyphony, piano_model)
            

    def extract_data_path(self):
        """
            Load all data 
            Return: 
                [(audio_file_path, midi_file_path, piano_model(year)), ....]
        """
        
        files = sorted([(os.path.join(self.data_path, self.dataframe[row]['audio_filename']),
                             os.path.join(self.data_path, self.dataframe[row]['midi_filename']), self.dataframe[row]['year']) for row in self.dataframe if self.dataframe[row]['split'] == self.split])
        files = [(audio, midi, piano_model) for audio, midi, piano_model in files]
        return files

    def load_data(
        self,
        audio_path,
        midi_path,
        segment_duration=3.,
        max_polyphony=None,
        overlap=0.5,
        sample_rate=16000,
        frame_rate=250): 
        # 會返回 某首歌的全部 segments
        """
        Load aligned audio and MIDI data (as conditioning sequence), then split
        into segments.
            Args:
                - audio_path (path): path to audio file.
                - midi_path (path): path to midi file.
                - segment_duration (float): length of segment chunks (in s).
                - max_polyphony (int): number of monophonic channels for the conditio-
                ning vector (return the piano rolls if None).
                - overlap (float): overlapping ratio between two consecutive segemnts.
                - sample_rate (int): number of audio samples per second.
                - frame_rate (int): number of conditioning vectors per second.
            Returns:
                - segment_audio (list [n_samples,]): list of audio segments.
                - segment_rolls (list [n_frames, max_polyphony, 2]): list of segments
                conditioning vectors.
                - segment_pedals (list [n_frames, 4]): list of segments pedals condi-
                tioning.
                - polyphony (list [n_frames, 1]): list of polyphony information in the
                original piano roll.
        """
        n_samples = int(segment_duration * sample_rate)
        n_frames = int(segment_duration * frame_rate)
        audio_hop_size = int(n_samples * (1 - overlap))
        midi_hop_size = int(n_frames * (1 - overlap))

        # Read audio file
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        
        # Read MIDI file
        note_sequence = load_midi_as_note_sequence(
            midi_path
        )
        # Convert to pianoroll
        roll = seq_lib.sequence_to_pianoroll(note_sequence,
                                            frames_per_second=frame_rate,
                                            min_pitch=21,
                                            max_pitch=108)
        # Retrieve activity and onset velocities
        midi_roll = np.stack((roll.active, roll.onset_velocities), axis=-1)

        # Pedals are CC64, 66 and 67
        pedals = roll.control_changes[:, 64: 68] / 128.0
        if max_polyphony is not None:
            polyphony_manager = MIDIRoll2Conditioning(max_polyphony)
            midi_roll, polyphony = polyphony_manager(midi_roll)
        
        # Split into segments
        audio_t = 0
        midi_t = 0
        segment_audio = []
        segment_rolls = []
        segment_pedals = []
        segment_polyphony = []
        while midi_t + n_frames < np.shape(midi_roll)[0]:
            segment_audio.append(audio[audio_t: audio_t + n_samples])
            segment_rolls.append(midi_roll[midi_t: midi_t + n_frames])
            segment_pedals.append(pedals[midi_t: midi_t + n_frames])    
            if max_polyphony:
                segment_polyphony.append(polyphony[midi_t: midi_t + n_frames])

            audio_t += audio_hop_size
            midi_t += midi_hop_size

        n_segments = len(segment_rolls)
        if max_polyphony is None:
            return np.array(segment_audio), np.array(segment_rolls), np.array(segment_pedals), n_segments
        else:
            return np.array(segment_audio), np.array(segment_rolls), np.array(segment_pedals), np.array(segment_polyphony), n_segments

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        audio, conditioning, pedal, piano_model = data['audio'], data['conditioning'], data['pedal'], data['piano_model']        
        
        # Encode piano model as one-hot
        piano_model_one_hot = torch.where(torch.eq(self.piano_models, torch.from_numpy(piano_model).int()))[0]
        
        return torch.from_numpy(audio).float(), torch.from_numpy(conditioning).float(), torch.from_numpy(pedal).float(), piano_model_one_hot[0]
        #torch.from_numpy(audio).float(), torch.from_numpy(conditioning).float(), torch.from_numpy(pedal).float(), torch.from_numpy(polyphony).int(), piano_model_one_hot[0]

