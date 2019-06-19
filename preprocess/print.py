from pprint import pprint
import midi
pattern = midi.read_midifile("3.mid")

preprocessed = midi.Pattern(resolution=pattern.resolution)
print(pattern)
for it in pattern:
    tick_sum = 0
    for line in it:
        tick_sum += line.tick
    print(tick_sum)
