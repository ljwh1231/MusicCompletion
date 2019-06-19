import midi


def main():
    pattern = midi.read_midifile("1.mid")

    output = midi.Pattern(resolution=480)
    output.append(pattern[4])
    midi.write_midifile('temp.mid', output)
    print(output)
    return
    preprocessed = midi.Pattern(resolution=pattern.resolution)

    for idx, it in enumerate(pattern):
        if idx == 0:
            preprocessed.append(it)

    print(preprocessed)
    midi.write_midifile('ebubu.mid', preprocessed)


if __name__ == '__main__':
    main()
