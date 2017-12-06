''' Extracts features from a video stimulus'''

import sys
import numpy as np
import pandas as pd
from pliers.converters import (IBMSpeechAPIConverter,
                               VideoToAudioConverter)
from pliers.export import to_long_format
from pliers.extractors import (GoogleVisionAPILabelExtractor,
                               GoogleVisionAPIFaceExtractor,
                               WordEmbeddingExtractor,
                               RMSEExtractor,
                               BrightnessExtractor,
                               merge_results)
from pliers.filters import FrameSamplingFilter
from pliers.stimuli import VideoStim, TextStim


WORD2VEC_PATH = '/Users/quinnmac/Documents/00-Documents-Archive/College Senior Year/'\
                'Semester 2/NLP/Final Project/GoogleNews-vectors-negative300.bin'
TR = 1.5


def parse_p2fa(transcript_path):
    with open(transcript_path) as f:
        start_parse = False
        all_lines = f.readlines()
        texts = []
        for i, line in enumerate(all_lines):
            if line == '\titem [2]:\n':
                start_parse = True
            if start_parse and line.startswith('\t\t\ti'):
                onset = float(all_lines[i+1].split()[-1])
                duration = float(all_lines[i+2].split()[-1]) - onset
                text = str(all_lines[i+3].split()[-1])[1:-1]
                if not (text == 'sp'):
                    texts.append(TextStim(text=text, onset=onset, duration=duration))

    return texts


def extract_audio_semantics(stims):
    if isinstance(stims, VideoStim):
        speech_text_converter = IBMSpeechAPIConverter()
        stims = speech_text_converter.transform(stims)
        stims.save('stims/transcription/ibm_transcript.txt')
    ext = WordEmbeddingExtractor(WORD2VEC_PATH, binary=True)
    results = ext.transform(stims)
    res = merge_results(results, metadata=False, flatten_columns=True)
    res = to_long_format(res)
    res.rename(columns={'value': 'modulation', 'feature': 'trial_type'}, inplace=True)
    res.to_csv('events/audio_semantic_events.csv')
    return res


def extract_image_labels(video, save_frames=False):
    # Sample frames at TR
    frame_sampling_filter = FrameSamplingFilter(every=int(TR*video.fps))
    sampled_video = frame_sampling_filter.transform(video)

    if save_frames:
        # Save frames as images
        for i, f in enumerate(sampled_video):
            if i % 100 == 0:
                f.save('stims/frames/frame_%d.png' % i)

    # Use a Vision API to extract object labels
    ext = GoogleVisionAPILabelExtractor(max_results=10)
    results = ext.transform(sampled_video)
    df = merge_results(results, metadata=False)

    # Clean and write out data
    df = df.fillna(0)
    df['GoogleVisionAPILabelExtractor'] = np.round(df['GoogleVisionAPILabelExtractor'])
    new_cols = []
    for col in df.columns.values:
        if col[0].startswith('Google'):
            new_cols.append(col[1].encode('utf-8'))
        else:
            new_cols.append(col[0])
    df.columns = new_cols
    df.to_csv('events/raw_visual_events.csv')
    return df


def extract_visual_objects(visual_events):
    df = pd.DataFrame.from_csv(visual_events)
    df = to_long_format(df)
    df.rename(columns={'value': 'modulation', 'feature': 'trial_type'}, inplace=True)
    df.to_csv('events/visual_object_events.csv')


def extract_visual_semantics(visual_events):
    df = pd.DataFrame.from_csv(visual_events)
    onsets = df['onset']
    durations = df['duration']
    df = df.drop(['onset', 'duration'], axis=1)
    words = df.apply(lambda x: list(df.columns[x.values.astype('bool')]), axis=1)

    texts = []
    for tags, o, d in zip(words, onsets, durations):
        for w in tags:
            texts.append(TextStim(text=w, onset=o, duration=d))
    ext = WordEmbeddingExtractor(WORD2VEC_PATH, binary=True)
    results = ext.transform(texts)
    res = merge_results(results, metadata=False, flatten_columns=True)
    res = res.drop('duration', axis=1)
    res = res.groupby('onset').sum().reset_index()
    res['duration'] = durations
    res = to_long_format(res)
    res.rename(columns={'value': 'modulation', 'feature': 'trial_type'}, inplace=True)
    res.to_csv('events/visual_semantic_events.csv')
    return res


def extract_audio_energy(video):
    aud = VideoToAudioConverter().transform(video)
    frame_length = int(aud.sampling_rate*TR)
    ext = RMSEExtractor(frame_length=frame_length, hop_length=frame_length, center=False)
    res = ext.transform(aud).to_df(metadata=False)
    res = to_long_format(res)
    res['onset'] += TR
    res.rename(columns={'value': 'modulation', 'feature': 'trial_type'}, inplace=True)
    res.to_csv('events/audio_energy_events.csv')


def extract_brightness(video):
    frame_sampling_filter = FrameSamplingFilter(every=5)
    sampled_video = frame_sampling_filter.transform(video)
    ext = BrightnessExtractor()
    res = ext.transform(sampled_video)
    res = merge_results(res, metadata=False, flatten_columns=True)
    res = to_long_format(res)
    res.rename(columns={'value': 'modulation', 'feature': 'trial_type'}, inplace=True)
    res.to_csv('events/visual_brightness_events.csv')


def extract_faces(video):
    # Sample frames at TR
    frame_sampling_filter = FrameSamplingFilter(every=int(TR*video.fps))
    sampled_video = frame_sampling_filter.transform(video)

    # Use a Vision API to extract object labels
    ext = GoogleVisionAPIFaceExtractor()
    results = ext.transform(sampled_video)
    df = merge_results(results, metadata=False, flatten_columns=True)

    # Clean and write out data
    df = df.fillna(0)
    df['face'] = df['GoogleVisionAPIFaceExtractor_face1_face_detectionConfidence']
    cols = []
    for col in df.columns:
        if col.startswith('Google'):
            cols.append(col)
    df = df.drop(cols, axis=1)
    df = to_long_format(df)
    df.rename(columns={'value': 'modulation', 'feature': 'trial_type'}, inplace=True)
    df.to_csv('events/face_events.csv')


def extract_speech():
    speech_events = pd.read_csv('events/audio_semantic_events.csv')
    vals = [1.0] * len(speech_events['onset'])
    res = pd.DataFrame({'onset': speech_events['onset'],
                        'duration': speech_events['duration'],
                        'speech': vals})
    res = to_long_format(res)
    res.rename(columns={'value': 'modulation', 'feature': 'trial_type'}, inplace=True)
    res.to_csv('events/speech_events.csv')


if __name__ == '__main__':
    if len(sys.argv) != 1:
        print('Usage: python extract_features.py')
        sys.exit(0)

    video = VideoStim('stims/Merlin.mp4')
    # extract_image_labels(video, save_frames=True)
    # print('Done with label extraction')
    # extract_visual_semantics('events/raw_visual_events.csv')
    # print('Done with visual semantics extraction')
    # extract_visual_objects('events/raw_visual_events.csv')
    # print('Done with visual object extraction')
    # extract_audio_semantics(parse_p2fa('stims/transcription/Merlin_trimmed.TextGrid'))
    # print('Done with audio semantics extraction')
    # extract_audio_energy(video)
    # extract_speech()
    # extract_brightness(video)
    extract_faces(video)
    print('Done with controls extraction')
