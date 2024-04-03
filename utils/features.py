import numpy as np
import argparse
import csv
import os
import time
import logging
import h5py
import librosa
import logging
import json
from utilities import (create_folder, float32_to_int16, create_logging, 
    get_filename, read_metadata, read_midi, read_maps_midi,read_guitarset_midi, jams_to_midi, get_first_and_last_element)
import config


def pack_maestro_dataset_to_hdf5(args):
    """Load & resample MAESTRO audio files, then write to hdf5 files.

    Args:
      dataset_dir: str, directory of dataset
      workspace: str, directory of your workspace
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace

    sample_rate = config.sample_rate

    # Paths
    csv_path = os.path.join(dataset_dir, 'maestro-v3.0.0.csv')
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'maestro')

    logs_dir = os.path.join(workspace, 'logs', get_filename(__file__))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    # Read meta dict
    meta_dict = read_metadata(csv_path)

    audios_num = len(meta_dict['canonical_composer'])
    logging.info('Total audios number: {}'.format(audios_num))

    feature_time = time.time()

    # Load & resample each audio file to a hdf5 file
    for n in range(audios_num):
        logging.info('{} {}'.format(n, meta_dict['midi_filename'][n]))

        # Read midi
        midi_path = os.path.join(dataset_dir, meta_dict['midi_filename'][n])
        midi_dict = read_midi(midi_path)

        # Load audio
        audio_path = os.path.join(dataset_dir, meta_dict['audio_filename'][n])
        (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

        packed_hdf5_path = os.path.join(waveform_hdf5s_dir, '{}.h5'.format(
            os.path.splitext(meta_dict['audio_filename'][n])[0]))

        create_folder(os.path.dirname(packed_hdf5_path))

        with h5py.File(packed_hdf5_path, 'w') as hf:
            hf.attrs.create('canonical_composer', data=meta_dict['canonical_composer'][n].encode(), dtype='S100')
            hf.attrs.create('canonical_title', data=meta_dict['canonical_title'][n].encode(), dtype='S100')
            hf.attrs.create('split', data=meta_dict['split'][n].encode(), dtype='S20')
            hf.attrs.create('year', data=meta_dict['year'][n].encode(), dtype='S10')
            hf.attrs.create('midi_filename', data=meta_dict['midi_filename'][n].encode(), dtype='S100')
            hf.attrs.create('audio_filename', data=meta_dict['audio_filename'][n].encode(), dtype='S100')
            hf.attrs.create('duration', data=meta_dict['duration'][n], dtype=np.float32)

            hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100')
            hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32)
            hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
        
    logging.info('Write hdf5 to {}'.format(packed_hdf5_path))
    logging.info('Time: {:.3f} s'.format(time.time() - feature_time))


def pack_maps_dataset_to_hdf5(args):
    """MAPS is a piano dataset only used for evaluating our piano transcription
    system (optional). Ref:

    [1] Emiya, Valentin. "MAPS Database A piano database for multipitch 
    estimation and automatic transcription of music. 2016

    Load & resample MAPS audio files, then write to hdf5 files.

    Args:
      dataset_dir: str, directory of dataset
      workspace: str, directory of your workspace
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace

    sample_rate = config.sample_rate
    pianos = ['ENSTDkCl', 'ENSTDkAm']

    # Paths
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'maps')

    logs_dir = os.path.join(workspace, 'logs', get_filename(__file__))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    feature_time = time.time()
    count = 0

    # Load & resample each audio file to a hdf5 file
    for piano in pianos:
        sub_dir = os.path.join(dataset_dir, piano, 'MUS')

        audio_names = [os.path.splitext(name)[0] for name in os.listdir(sub_dir) 
            if os.path.splitext(name)[-1] == '.mid']
        
        for audio_name in audio_names:
            print('{} {}'.format(count, audio_name))
            audio_path = '{}.wav'.format(os.path.join(sub_dir, audio_name))
            midi_path = '{}.mid'.format(os.path.join(sub_dir, audio_name))

            (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
            midi_dict = read_maps_midi(midi_path)
            
            packed_hdf5_path = os.path.join(waveform_hdf5s_dir, '{}.h5'.format(audio_name))
            create_folder(os.path.dirname(packed_hdf5_path))

            with h5py.File(packed_hdf5_path, 'w') as hf:
                hf.attrs.create('split', data='test'.encode(), dtype='S20')
                hf.attrs.create('midi_filename', data='{}.mid'.format(audio_name).encode(), dtype='S100')
                hf.attrs.create('audio_filename', data='{}.wav'.format(audio_name).encode(), dtype='S100')
                hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100')
                hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32)
                hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
            
            count += 1

    logging.info('Write hdf5 to {}'.format(packed_hdf5_path))
    logging.info('Time: {:.3f} s'.format(time.time() - feature_time))

def pack_GuitarSet_dataset_to_hdf5(args):
    """GuitarSet  Ref:

    [1] Emiya, Valentin. "MAPS Database A piano database for multipitch 
    estimation and automatic transcription of music. 2016

    Load & resample MAPS audio files, then write to hdf5 files.

    Args:
      dataset_dir: str, directory of dataset
      workspace: str, directory of your workspace
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace

    sample_rate = config.sample_rate
    json_path = os.path.join(dataset_dir, "GuitarSet_seed:42_ratio_0.9.json")

    # Paths
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'GuitarSet')

    logs_dir = os.path.join(workspace, 'logs', get_filename(__file__))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    feature_time = time.time()
    count = 0

    # Read meta dict

    metadata = json.load(open(json_path)) 
    train_files = [(wav_path, jams_path, "train") for wav_path, jams_path in zip(metadata["train_data"], metadata["train_annotation"])]
    test_files = [(wav_path, jams_path, "test") for wav_path, jams_path in zip(metadata["test_data"], metadata["test_annotation"])]

    feature_time = time.time()

    # Load & resample each audio file to a hdf5 file
    for wav_path, jams_path, split in train_files + test_files:
        wav_name, jams_name = wav_path.split("/")[-1], jams_path.split("/")[-1]
        
        # Convert jams to midi temporarily 
        tmp_mid_path = jams_path.split("/")[-1][:-4]+"mid"
        _ = jams_to_midi(jams_path, save_path = tmp_mid_path)
        midi_dict = read_guitarset_midi(tmp_mid_path)
        os.system(f"rm {tmp_mid_path}")

        # Load audio
        (audio, _) = librosa.core.load(wav_path, sr=sample_rate, mono=True)
        audio = np.clip(audio, -1, 1)
        packed_hdf5_path = os.path.join(waveform_hdf5s_dir, f'{wav_path.split("/")[-1].split(".")[0]}.h5')

        create_folder(os.path.dirname(packed_hdf5_path))

        with h5py.File(packed_hdf5_path, 'w') as hf:
            hf.attrs.create('split', data=split.encode(), dtype='S20')
            hf.attrs.create('jams_filename', data=jams_name.encode(), dtype='S100')
            hf.attrs.create('audio_filename', data=wav_name.encode(), dtype='S100')
            hf.attrs.create('duration', data=len(audio)/sample_rate, dtype=np.float32)

            hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100') #TODO
            hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32) #TODO
            hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
        
    logging.info('Write hdf5 to {}'.format(packed_hdf5_path))
    logging.info('Time: {:.3f} s'.format(time.time() - feature_time))

def pack_Gaestro_dataset_to_hdf5(args):
    """Gaestro Ref:

    [1]

    Args:
      dataset_dir: str, directory of dataset
      workspace: str, directory of your workspace
    """
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace

    sample_rate = config.sample_rate
    json_path = args.json_file

    # Paths
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'Gaestro')

    logs_dir = os.path.join(workspace, 'logs', get_filename(__file__))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    feature_time = time.time()
    count = 0

    # Read meta dict

    metadata = json.load(open(json_path)) 
    train_files = [(wav_path, mid_path, "train") for wav_path ,mid_path in zip(metadata["train"], metadata["train_annotation"])]
    test_files = [(wav_path, mid_path, "test") for wav_path, mid_path in zip(metadata["test"], metadata["test_annotation"])]

    feature_time = time.time()

    # Load & resample each audio file to a hdf5 file
    for wav_path, mid_path, split in train_files + test_files:
        wav_name, mid_name = wav_path.split("/")[-1], mid_path.split("/")[-1]
        
        # Convert jams to midi temporarily 
        sync_json_path =   "/".join(wav_path.split("/")[:-2]+["syncpoints", wav_path.split("/")[-1][:-4]+"-syncpoints.json"] ) #.../audio/-D1wc.wav --> .../syncpoints/-D1wc-syncpoints.json
        start_time, end_time = get_first_and_last_element(sync_json_path)
        midi_dict = read_guitarset_midi(mid_path)

        # Load audio
        (audio, _) = librosa.core.load(wav_path, sr=sample_rate, mono=True)

        audio = np.clip(audio, -1, 1)
        packed_hdf5_path = os.path.join(waveform_hdf5s_dir, f'{wav_path.split("/")[-1].split(".")[0]}.h5')

        create_folder(os.path.dirname(packed_hdf5_path))

        with h5py.File(packed_hdf5_path, 'w') as hf:
            hf.attrs.create('split', data=split.encode(), dtype='S20')
            hf.attrs.create('midi_filename', data=mid_name.encode(), dtype='S100')
            hf.attrs.create('audio_filename', data=wav_name.encode(), dtype='S100')
            hf.attrs.create('duration', data=len(audio)/sample_rate, dtype=np.float32)
            hf.attrs.create('start_time', data=start_time, dtype=np.float32)
            hf.attrs.create('end_time', data=end_time, dtype=np.float32)
            hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100') #TODO
            hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32) #TODO
            hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
        
    logging.info('Write hdf5 to {}'.format(packed_hdf5_path))
    logging.info('Time: {:.3f} s'.format(time.time() - feature_time))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    parser_pack_maestro = subparsers.add_parser('pack_maestro_dataset_to_hdf5')
    parser_pack_maestro.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_maestro.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')

    parser_pack_maps = subparsers.add_parser('pack_maps_dataset_to_hdf5')
    parser_pack_maps.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_maps.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')

    parser_pack_guitarset = subparsers.add_parser('pack_GuitarSet_dataset_to_hdf5')
    parser_pack_guitarset.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_guitarset.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')

    parser_pack_gaestro = subparsers.add_parser('pack_Gaestro_dataset_to_hdf5')
    parser_pack_gaestro.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_gaestro.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_pack_gaestro.add_argument('--json_file', type=str, required=False, default = '/import/c4dm-datasets-ext/edge_aistpp/Nic_Gastro/Gastro_v1_metadata_train_ratio0.9_seed_42.json',help='Directory of the config file.')
    # Parse arguments
    args = parser.parse_args()
    
    if args.mode == 'pack_maestro_dataset_to_hdf5':
        pack_maestro_dataset_to_hdf5(args)
        
    elif args.mode == 'pack_maps_dataset_to_hdf5':
        pack_maps_dataset_to_hdf5(args)
    elif args.mode == 'pack_GuitarSet_dataset_to_hdf5':
        pack_GuitarSet_dataset_to_hdf5(args)
    elif args.mode == 'pack_Gaestro_dataset_to_hdf5':
        pack_Gaestro_dataset_to_hdf5(args)
    else:
        raise Exception('Incorrect arguments!')