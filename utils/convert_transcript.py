from pliers.stimuli import ComplexTextStim, TextStim

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
                text = str(all_lines[i+3].split()[-1])[1:-1].lower()
                if not (text == 'sp'):
                    texts.append(TextStim(text=text, onset=onset, duration=duration))

    return texts

stim = ComplexTextStim(elements=parse_p2fa('stims/transcription/Merlin_trimmed.TextGrid'))
stim.save('stims/transcription/aligned_transcript.txt')