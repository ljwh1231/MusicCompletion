from pprint import pprint
import midi

pattern = midi.read_midifile("12_modify.mid")
print(pattern[0])
# piano bass guitar melody drum
# del pattern[0]

# midi.write_midifile('./12_modify.mid', pattern)
