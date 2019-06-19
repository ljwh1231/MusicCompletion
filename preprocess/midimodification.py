from pprint import pprint
import midi

pattern = midi.read_midifile("19.mid")
# print(pattern)
# piano bass guitar melody drum
del pattern[0]
pattern[3], pattern[2] = pattern[2], pattern[3]
pattern[0], pattern[4] = pattern[4], pattern[0]
pattern[3], pattern[0] = pattern[0], pattern[3]
# pattern[5], pattern[4] = pattern[4], pattern[5]

for idx, val in enumerate(pattern):
    # if idx == 0:
    #     continue

    for event in val:
        if isinstance(event, midi.ProgramChangeEvent):
            if event.channel == 9:
                print(idx, 'drum')
            else:
                print(idx, event.data)
            break
        elif hasattr(event, 'channel') and event.channel == 9:
            print(idx, 'drum')
            break

    # if idx == 2:
    #     new_pattern = midi.Pattern(resolution=pattern.resolution)
    #     new_pattern.append(val)
    #     midi.write_midifile('./temp.mid', new_pattern)

midi.write_midifile('./19_modify.mid', pattern)
