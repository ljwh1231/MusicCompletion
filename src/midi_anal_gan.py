import mido
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import colorConverter
import midi


class MidiFile(mido.MidiFile):
    def __init__(self, filename):
        mido.MidiFile.__init__(self, filename)
        self.sr = 10
        self.meta = {}
        self.events = self.get_events()

    def get_events(self):
        mid = self
        print(mid)
        # There is > 16 channel in midi.tracks. However there is only 16 channel related to "music" events.
        # We store music events of 16 channel in the list "events" with form [[ch1],[ch2]....[ch16]]
        # Lyrics and meta data used a extra channel which is not include in "events"

        events = [[] for x in range(16)]
        # Iterate all event in the midi and extract to 16 channel form
        for track in mid.tracks:
            for msg in track:
                try:
                    channel = msg.channel
                    events[channel].append(msg)
                except AttributeError:
                    try:
                        if not isinstance(msg, mido.UnknownMetaMessage):
                            self.meta[msg.type] = msg.dict()
                        else:
                            pass
                    except:
                        print("error", type(msg))

        return events

    def get_roll(self):
        events = self.get_events()
        # Identify events, then translate to piano roll
        # choose a sample ratio(sr) to down-sample through time axis
        sr = self.sr
        # compute total length in tick unit
        length = self.get_total_ticks()

        # allocate memory to numpy array
        roll = np.zeros((16, 128, length // sr), dtype="int8")
        # print(roll.shape)

        # use a register array to save the state(no/off) for each key
        note_register = [int(-1) for x in range(128)]

        # use a register array to save the state(program_change) for each channel
        timbre_register = [1 for x in range(16)]

        for idx, channel in enumerate(events):
            time_counter = 0
            volume = 100
            # Volume would change by control change event (cc) cc7 & cc11
            # Volume 0-100 is mapped to 0-127

            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> channel", idx, "start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            for msg in channel:
                if msg.type == "control_change":
                    if msg.control == 7:
                        volume = msg.value
                        # directly assign volume
                    if msg.control == 11:
                        volume = volume * msg.value // 127
                        # change volume by percentage
                    # print("cc", msg.control, msg.value, "duration", msg.time)

                if msg.type == "program_change":
                    timbre_register[idx] = msg.program
                    # print("channel", idx, "pc", msg.program, "time", time_counter, "duration", msg.time)

                if msg.type == "note_on":
                    # print("on ", msg.note, "time", time_counter, "duration", msg.time, "velocity", msg.velocity)
                    note_on_start_time = time_counter // sr
                    note_on_end_time = (time_counter + msg.time) // sr
                    intensity = volume * msg.velocity // 127

                    # When a note_on event *ends* the note start to be play
                    # Record end time of note_on event if there is no value in register
                    # When note_off event happens, we fill in the color
                    if note_register[msg.note] == -1:
                        note_register[msg.note] = (note_on_end_time,intensity)
                    else:
                        # When note_on event happens again, we also fill in the color
                        old_end_time = note_register[msg.note][0]
                        old_intensity = note_register[msg.note][1]
                        roll[idx, msg.note, old_end_time: note_on_end_time] = old_intensity
                        note_register[msg.note] = (note_on_end_time,intensity)

                if msg.type == "note_off":
                    # print("off", msg.note, "time", time_counter, "duration", msg.time, "velocity", msg.velocity)
                    note_off_start_time = time_counter // sr
                    note_off_end_time = (time_counter + msg.time) // sr
                    note_on_end_time = note_register[msg.note][0]
                    intensity = note_register[msg.note][1]
                    # fill in color
                    roll[idx, msg.note, note_on_end_time:note_off_end_time] = intensity

                    note_register[msg.note] = -1  # reinitialize register

                time_counter += msg.time

                # TODO : velocity -> done, but not verified
                # TODO: Pitch wheel
                # TODO: Channel - > Program Changed / Timbre catagory
                # TODO: real time scale of roll

            # if there is a note not closed at the end of a channel, close it
            for key, data in enumerate(note_register):
                if data != -1:
                    note_on_end_time = data[0]
                    intensity = data[1]
                    # print(key, note_on_end_time)
                    note_off_start_time = time_counter // sr
                    roll[idx, key, note_on_end_time:] = intensity
                note_register[idx] = -1

        return roll

    def get_roll_image(self):
        roll = self.get_roll()
        plt.ioff()

        K = 16

        transparent = colorConverter.to_rgba('black')
        colors = [mpl.colors.to_rgba(mpl.colors.hsv_to_rgb((i / K, 1, 1)), alpha=1) for i in range(K)]
        cmaps = [mpl.colors.LinearSegmentedColormap.from_list('my_cmap', [transparent, colors[i]], 128) for i in
                 range(K)]

        for i in range(K):
            cmaps[i]._init()  # create the _lut array, with rgba values
            # create your alpha array and fill the colormap with them.
            # here it is progressive, but you can create whathever you want
            alphas = np.linspace(0, 1, cmaps[i].N + 3)
            cmaps[i]._lut[:, -1] = alphas

        fig = plt.figure(figsize=(4, 3))
        a1 = fig.add_subplot(111)
        a1.axis("equal")
        a1.set_facecolor("black")

        array = []

        for i in range(K):
            try:
                img = a1.imshow(roll[i], interpolation='nearest', cmap=cmaps[i], aspect='auto')
                array.append(img.get_array())
            except IndexError:
                pass
        return array

    def get_tempo(self):
        try:
            return self.meta["set_tempo"]["tempo"]
        except:
            return 500000

    def get_total_ticks(self):
        max_ticks = 0
        for channel in range(16):
            ticks = sum(msg.time for msg in self.events[channel])
            if ticks > max_ticks:
                max_ticks = ticks
        return max_ticks

    def draw_sessions(self, roll):
        for track_num in range(len(roll)):
            track_img = roll[track_num]
            try:
                max_val = -1
                min_val = 128
                for line in track_img:
                    for val in line:
                        max_val = max(max_val, val)
                        min_val = min(min_val, val)
                track_img[track_img > min_val] = max_val
                plt.imsave('result_images/' + 'track' + str(track_num) + '.png', track_img, cmap=plt.get_cmap('gray'))
                print("image save complete")
            except:
                print("can't save image")


def extract_track(file):
    pattern = midi.read_midifile(file)
    resolution = pattern.resolution

    extracted = midi.Pattern(resolution=resolution)
    # DLBIA : 1 = piano -> track0
    #         2 = bass -> track1
    #         3 = guitar -> track7
    #         4 = drum -> track9
    #         5 = melody -> track13
    for idx, it in enumerate(pattern):
        if idx == 1:
            extracted.append(it)

    midi.write_midifile("./result_midi/result_DLBIA.mid", extracted)


def image_to_roll(img, min_val, max_val):
    ret_img = np.array([[int(img[i][j][0] * (max_val - min_val) + min_val) for j in range(img.shape[1])]
                        for i in range(img.shape[0])])
    plt.imsave('result_images/temp.png', ret_img, cmap=plt.get_cmap('gray'))
    return ret_img


def roll_to_midi(rolls, tick_per_time_slice, resolution=120):
    # melody is index 0
    # guitar is index 1
    # piano is index 2
    # bass is index 3
    # drum is index 4

    result = midi.Pattern(resolution=resolution*100)
    for i in range(5):
        if i >= len(rolls):
            continue

        roll = rolls[i]
        channel = i

        # for drum
        if channel == 4:
            channel = 9

        track = midi.Track()
        if i == 0:
            track.append(midi.ProgramChangeEvent(tick=0, data=[53], channel=channel))
        elif i == 1:
            track.append(midi.ProgramChangeEvent(tick=0, data=[29], channel=channel))
        elif i == 2:
            track.append(midi.ProgramChangeEvent(tick=0, data=[0], channel=channel))
        elif i == 3:
            track.append(midi.ProgramChangeEvent(tick=0, data=[34], channel=channel))
        else:
            track.append(midi.ProgramChangeEvent(tick=0, data=[1], channel=channel))

        current_tick = 0
        prev_intensity = [0 for _ in range(128)]
        prev_tick = 0

        for time_slice in range(roll.shape[1]):
            for pitch in range(roll.shape[0]):
                if roll[pitch][time_slice] != prev_intensity[pitch]:
                    diff_tick = current_tick - prev_tick
                    if roll[pitch][time_slice] == 0:
                        track.append(midi.NoteOffEvent(tick=diff_tick,
                                                       data=[pitch, prev_intensity[pitch]],
                                                       channel=channel))
                        prev_intensity[pitch] = 0
                        prev_tick = current_tick
                    else:
                        if prev_intensity[pitch] == 0:
                            track.append(midi.NoteOnEvent(tick=diff_tick,
                                                          data=[pitch, roll[pitch][time_slice]],
                                                          channel=channel))
                            prev_intensity[pitch] = roll[pitch][time_slice]
                            prev_tick = current_tick
                        else:
                            track.append(midi.NoteOffEvent(tick=diff_tick,
                                                           data=[pitch, prev_intensity[pitch]],
                                                           channel=channel))
                            track.append(midi.NoteOnEvent(tick=diff_tick,
                                                          data=[pitch, roll[pitch][time_slice]],
                                                          channel=channel))
                            prev_intensity[pitch] = roll[pitch][time_slice]
                            prev_tick = current_tick
                current_tick += tick_per_time_slice

        track.append(midi.EndOfTrackEvent(tick=0, channel=channel))
        # if i == 1:
        result.append(track)
    return result


# DLBIA : 1 = piano -> track0
#         2 = bass -> track1
#         3 = guitar -> track7
#         4 = drum -> track9
#         5 = melody -> track13
if __name__ == "__main__":
    mid = MidiFile("./midi_files/DLBIA.mid")
    extract_track("./midi_files/DLBIA.mid")
    # mid = MidiFile("./result_midi/result_DLBIA.mid")
    # track_ext = extract_track("./result_midi/result_DLBIA.mid")
    # get the list of all events
    # events = mid.get_events()

    # get the np array of piano roll image
    roll = mid.get_roll()
    # ret_midi = roll_to_midi([roll[13], roll[7], roll[0], roll[1], roll[9]], 10, 240)
    # bass_img = plt.imread('input_images/bass.png')
    # ret_midi = roll_to_midi([roll[13], roll[7], roll[0], image_to_roll(bass_img, 0, 128), roll[9]], 10, 240)
    # midi.write_midifile('result.mid', ret_midi)
    mid.draw_sessions(roll)
    # draw piano roll by pyplot
    # mid.draw_roll()
