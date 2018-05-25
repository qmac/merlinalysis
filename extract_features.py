''' Extracts features from a video stimulus'''

import numpy as np
import pandas as pd
from pliers.converters import VideoToAudioConverter
from pliers.extractors import (GoogleVisionAPILabelExtractor,
                               GoogleVisionAPIFaceExtractor,
                               WordEmbeddingExtractor,
                               RMSEExtractor,
                               merge_results)
from pliers.filters import FrameSamplingFilter
from pliers.stimuli import VideoStim, TextStim, ComplexTextStim


WORD2VEC_PATH = '/Users/quinnmac/Documents/00-Documents-Archive/'\
                'College Senior Year/Semester 2/NLP/Final Project/'\
                'GoogleNews-vectors-negative300.bin'
GLOVE_PATH = '/Users/quinnmac/Documents/00-Documents-Archive/'\
             'codenames/glove.6B/glove.6B.300d.txt.word2vec'
TR = 1.5


def extract_speech(stims):
    onsets, durations = zip(*[(e.onset, e.duration) for e in stims.elements])
    res = pd.DataFrame({'onset': onsets,
                        'duration': durations,
                        'modulation': [1.0] * len(onsets),
                        'trial_type': 'speech'})
    res.to_csv('events/audio_speech_events.csv')


def extract_audio_semantics(stims, glove=True):
    if glove:
        ext = WordEmbeddingExtractor(GLOVE_PATH, binary=False)
        out = 'events/audio_glove_events.csv'
    else:
        ext = WordEmbeddingExtractor(WORD2VEC_PATH, binary=True)
        out = 'events/audio_semantic_events.csv'
    results = ext.transform(stims)
    res = merge_results(results, metadata=False, flatten_columns=True,
                        format='long')
    res = res.drop(['object_id', 'order'], axis=1)
    res.rename(columns={'value': 'modulation', 'feature': 'trial_type'},
               inplace=True)
    res.to_csv(out)


def extract_audio_energy(video):
    aud = VideoToAudioConverter().transform(video)
    frame_length = int(aud.sampling_rate*TR)
    ext = RMSEExtractor(frame_length=frame_length, hop_length=frame_length,
                        center=False)
    res = ext.transform(aud).to_df(metadata=False, format='long')
    res['onset'] += TR
    res = res.drop(['object_id', 'order'], axis=1)
    res.rename(columns={'value': 'modulation', 'feature': 'trial_type'},
               inplace=True)
    res.to_csv('events/audio_energy_events.csv')


def extract_image_labels(video, save_frames=False):
    frame_sampling_filter = FrameSamplingFilter(hertz=1)
    sampled_video = frame_sampling_filter.transform(video)

    if save_frames:
        # Save frames as images
        for i, f in enumerate(sampled_video):
            if i % 100 == 0:
                f.save('stims/frames/frame_%d.png' % i)

    # Use a Vision API to extract object labels
    ext = GoogleVisionAPILabelExtractor(max_results=10)
    results = ext.transform(sampled_video)
    res = merge_results(results, metadata=False, extractor_names='multi')

    # Clean and write out data
    res = res.fillna(0)
    label_key = 'GoogleVisionAPILabelExtractor'
    res[label_key] = np.round(res[label_key])
    new_cols = []
    for col in res.columns.values:
        if col[0].startswith('Google'):
            new_cols.append(col[1].encode('utf-8'))
        else:
            new_cols.append(col[0])
    res.columns = new_cols
    res.to_csv('events/raw_visual_events.csv')


def extract_visual_objects(visual_events):
    res = pd.DataFrame.from_csv(visual_events)
    res = res.drop(['order', 'object_id'], axis=1)

    # Convert to long format
    ids = ['onset', 'duration']
    values = list(set(res.columns) - set(ids))
    res = pd.melt(res, id_vars=ids, value_vars=values, var_name='trial_type')

    # Slicing here to get rid of b''
    res['trial_type'] = [v[2:-1] for v in res['trial_type']]
    res.rename(columns={'value': 'modulation'}, inplace=True)
    res.to_csv('events/visual_object_events.csv')


def extract_visual_semantics(visual_events, glove=True):
    res = pd.DataFrame.from_csv(visual_events)
    onsets = res['onset']
    durations = res['duration']
    res = res.drop(['onset', 'duration', 'order', 'object_id'], axis=1)
    words = res.apply(lambda x: list(res.columns[x.values.astype('bool')]),
                      axis=1)

    texts = []
    for tags, o, d in zip(words, onsets, durations):
        for w in tags:
            # Slicing here to get rid of b''
            texts.append(TextStim(text=w[2:-1], onset=o, duration=d))

    if glove:
        ext = WordEmbeddingExtractor(GLOVE_PATH, binary=False)
        out = 'events/visual_semantic_events.csv'
    else:
        ext = WordEmbeddingExtractor(WORD2VEC_PATH, binary=True)
        out = 'events/visual_glove_events.csv'
    results = ext.transform(texts)
    res = merge_results(results, metadata=False, flatten_columns=True,
                        format='long')
    res = res.drop(['object_id', 'order', 'duration'], axis=1)
    res = res.groupby('onset').sum().reset_index()
    res['duration'] = durations
    res.rename(columns={'value': 'modulation', 'feature': 'trial_type'},
               inplace=True)
    res.to_csv(out)


def extract_faces(video):
    frame_sampling_filter = FrameSamplingFilter(hertz=1)
    sampled_video = frame_sampling_filter.transform(video)

    ext = GoogleVisionAPIFaceExtractor()
    results = ext.transform(sampled_video)
    res = merge_results(results, metadata=False, format='long',
                        extractor_names=False, object_id=False)

    res = res[res['feature'] == 'face_detectionConfidence']
    res = res.drop(['order'], axis=1)
    res = res.fillna(0)
    res['value'] = np.round(res['value'])
    res.rename(columns={'value': 'modulation', 'feature': 'trial_type'},
               inplace=True)
    res.to_csv('events/visual_face_events.csv')


if __name__ == '__main__':
    video = VideoStim('stims/Merlin.mp4')
    transcript = ComplexTextStim('stims/transcription/aligned_transcript.txt')
    for e in transcript.elements:
        e.onset += 40.5

    extract_speech(transcript)
    print('Done with speech extraction')
    extract_audio_semantics(transcript, glove=True)
    print('Done with audio semantics extraction')
    extract_audio_energy(video)
    print('Done with audio energy extraction')
    extract_image_labels(video, save_frames=False)
    print('Done with label extraction')
    extract_visual_semantics('events/raw_visual_events.csv', glove=True)
    print('Done with visual semantics extraction')
    extract_visual_objects('events/raw_visual_events.csv')
    print('Done with visual object extraction')
    extract_faces(video)
    print('Done with face extraction')
