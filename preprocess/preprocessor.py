import midi
import csv
import sys

# total_tick / resolution / total_seconds ~= 2.05
# 1s = (2.05 * resolution) tick
# 256 states per session (on : 128, off : 128)
# instrument_idx is melody : 0, guitar : 1, piano : 2, bass : 3, drum : 4

# instrument_sequence
# if 0 is piano, 1 is bass, 2 is guitar, 3 is melody, 4 is drum, instrument_sequence is [2, 3, 1, 0, 4]
# (idx, value) is (idx, instrument_idx)
# TODO: resolution is needed in csv ( how? )
# TODO: also piano, bass, guitar, melody's detail is needed   ex.) acoustic piano or electric piano


def preprocess(input_path, output_path, instrument_sequence, pattern_instrument_indexes=[0, 1, 2, 3, 4]):
    with open(output_path, 'w') as f:
        wr = csv.writer(f)

        pattern = midi.read_midifile(input_path)

        resolution = pattern.resolution
        result = []
        for idx, track in enumerate(pattern):
            time_slice = 0  # 0.01 seconds
            tick_sum = 0
            cur_state = [0 for _ in range(256)]

            if idx in pattern_instrument_indexes:
                for event in track:
                    tick_sum += event.tick
                    while time_slice * 0.0205 * resolution < tick_sum:  # (2.05 * resolution) tick * 0.01
                        if len(result) <= time_slice:
                            result.append([0 for _ in range(256*5)])

                        result[time_slice][instrument_sequence[idx]*256:instrument_sequence[idx]*256+256] = cur_state

                        for i in range(256):
                            cur_state[i] = 0
                        time_slice += 1

                    if isinstance(event, midi.NoteOnEvent):
                        cur_state[event.data[0]] = 1
                    elif isinstance(event, midi.NoteOffEvent):
                        cur_state[128 + event.data[0]] = 1

        for state in result:
            wr.writerow(state)


def main():
    # for i in range(20):
    i = 12
    preprocess('{}_modify.mid'.format(i), 'output{}.csv'.format(i), [0, 1, 2, 4, 3])  # piano, bass, guitar, melody, drum


if __name__ == '__main__':
    main()
