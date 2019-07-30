import midi
import csv

# total_tick / resolution / total_seconds ~= 2.05
# 1s = (2.05 * resolution) tick
# 256 + 100 states per session (on : 128, off : 128, time slice : 100)
# instrument_idx is melody : 0, guitar : 1, piano : 2, bass : 3, drum : 4


def process(input_path, output_path, resolution):
    output = midi.Pattern(resolution=resolution)
    print(resolution)

    melody_track = midi.Track()
    melody_track.append(midi.ProgramChangeEvent(tick=0, data=[53], channel=0))
    guitar_track = midi.Track()
    guitar_track.append(midi.ProgramChangeEvent(tick=0, data=[29], channel=1))
    piano_track = midi.Track()
    piano_track.append(midi.ProgramChangeEvent(tick=0, data=[0], channel=2))
    bass_track = midi.Track()
    bass_track.append(midi.ProgramChangeEvent(tick=0, data=[34], channel=3))
    drum_track = midi.Track()
    # drum_track.append(midi.TrackNameEvent(tick=0, text='tk10', data=[116, 107, 49, 48]))
    # drum_track.append(midi.PortEvent(tick=0, data=[0]))
    # drum_track.append(midi.ChannelPrefixEvent(tick=0, data=[9]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[0, 127]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[32, 0]))
    drum_track.append(midi.ProgramChangeEvent(tick=0, channel=9, data=[1]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[7, 110]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[11, 100]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[91, 34]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[99, 29]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[99, 28]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[98, 47]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[6, 121]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[99, 24]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[98, 47]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[6, 60]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[99, 24]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[98, 45]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[6, 64]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[99, 24]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[98, 43]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[6, 63]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[99, 28]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[98, 40]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[6, 53]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[99, 28]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[98, 57]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[6, 84]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[99, 24]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[98, 42]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[6, 60]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[98, 24]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[98, 46]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[6, 59]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[99, 24]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[98, 49]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[6, 59]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[99, 28]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[98, 49]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[6, 19]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[99, 28]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[98, 51]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[6, 95]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[99, 29]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[98, 55]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[6, 60]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[101, 127]))
    # drum_track.append(midi.ControlChangeEvent(tick=0, channel=9, data=[100, 127]))

    with open(input_path, 'r') as f:
        rdr = csv.reader(f)

        cur_tick = 0.0
        plus_tick_per_time_slice = 0.0205 * resolution
        print(plus_tick_per_time_slice)

        lines = 0
        for line in rdr:
            lines += 1
            ones = [i for i, x in enumerate(line) if x == '1']
            for i in ones:
                if 0 <= i < 128:
                    melody_track.append(midi.NoteOnEvent(tick=0, data=[i, 70], channel=0))
                elif 128 <= i < 256:
                    melody_track.append(midi.NoteOffEvent(tick=0, data=[i-128, 0], channel=0))
                elif 0 <= i-356 < 128:
                    guitar_track.append(midi.NoteOnEvent(tick=0, data=[i-356, 30], channel=1))
                elif 128 <= i-356 < 256:
                    guitar_track.append(midi.NoteOffEvent(tick=0, data=[i-356-128, 0], channel=1))
                elif 0 <= i-356*2 < 128:
                    piano_track.append(midi.NoteOnEvent(tick=0, data=[i-356*2, 30], channel=2))
                elif 128 <= i-356*2 < 256:
                    piano_track.append(midi.NoteOffEvent(tick=0, data=[i-356*2-128, 0], channel=2))
                elif 0 <= i-356*3 < 128:
                    bass_track.append(midi.NoteOnEvent(tick=0, data=[i-356*3, 30], channel=3))
                elif 128 <= i-356*3 < 256:
                    bass_track.append(midi.NoteOffEvent(tick=0, data=[i-356*3-128, 0], channel=3))
                elif 0 <= i-356*4 < 128:
                    drum_track.append(midi.NoteOnEvent(tick=0, data=[i-356*4, 30], channel=9))
                elif 128 <= i-356*4 < 256:
                    drum_track.append(midi.NoteOffEvent(tick=0, data=[i-356*4-128, 0], channel=9))

            prev_tick = cur_tick
            cur_tick += plus_tick_per_time_slice
            plus_tick = int(cur_tick - prev_tick)

            melody_track[-1].tick += plus_tick
            guitar_track[-1].tick += plus_tick
            piano_track[-1].tick += plus_tick
            bass_track[-1].tick += plus_tick
            drum_track[-1].tick += plus_tick
        print(lines)
        print(cur_tick)

    melody_track.append(midi.EndOfTrackEvent(tick=0, channel=0))
    guitar_track.append(midi.EndOfTrackEvent(tick=0, channel=1))
    piano_track.append(midi.EndOfTrackEvent(tick=0, channel=2))
    bass_track.append(midi.EndOfTrackEvent(tick=0, channel=3))
    drum_track.append(midi.EndOfTrackEvent(tick=0, channel=9))

    output.append(melody_track)
    output.append(guitar_track)
    output.append(piano_track)
    output.append(bass_track)
    output.append(drum_track)

    midi.write_midifile(output_path, output)


def main():
    process('output1.csv', 'output1_mid.mid', 480)


if __name__ == '__main__':
    main()