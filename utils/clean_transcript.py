''' Utility code for converting SRT files to P2FA input files '''

import re
from pliers.stimuli import ComplexTextStim

cts = ComplexTextStim('stims/subtitles.srt')

with open('stims/exact_transcript.txt', 'w') as new_file:
    for el in cts.elements:
        filtered = el.text
        matches = re.findall('\(\s*[A-Z]{2,}\s*\)', filtered)
        for m in matches:
            filtered = filtered.replace(m, '')
        new_file.write('01\tBob\t%f\t%f\t%s\n' % (el.onset, el.onset + el.duration, filtered))


# The rest of the cleaning was done manually
