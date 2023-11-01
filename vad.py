from pydub import AudioSegment
import collections
import contextlib
import webrtcvad
import wave
import os
import soundfile as sf
import shutil

class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

class Segment(object):

    def __init__(self, bytes, start, end):
        self.bytes = bytes
        self.start = start
        self.end = end

def format_wave(wave_path):
    if wave_path.endswith(('mp3', 'MP3')): 
        sound = AudioSegment.from_mp3(wave_path)
        wave_path = wave_path[:-4] + '.wav'
        sound = sound.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        sound.export(wave_path, format='wav')
    elif wave_path.endswith(('wav', 'WAV')):
        sound = AudioSegment.from_wav(wave_path)
    
    if sound.channels > 1 and sound.frame_rate != 16000 and sound.sample_width !=2:
        sound = sound.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        sound.export(wave_path, format='wav')
    return wave_path

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)
        frames = wf.getnframes()
        pcm_data = wf.readframes(frames)
        duration = frames / sample_rate
        return pcm_data, sample_rate, duration

def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield Segment(b''.join([f.bytes for f in voiced_frames]),
                              voiced_frames[0].timestamp,
                              voiced_frames[-1].timestamp + voiced_frames[-1].duration)
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        pass
    if voiced_frames:
        yield Segment(b''.join([f.bytes for f in voiced_frames]),
                      voiced_frames[0].timestamp,
                      voiced_frames[-1].timestamp + voiced_frames[-1].duration)

def frame_generator(frame_duration_ms, audio, sample_rate):
    frame_duration_s = frame_duration_ms / 1000.0
    frame_byte_count = int(sample_rate * frame_duration_s * 2)
    offset = 0
    timestamp = 0.0
    while offset + frame_byte_count -1 < len(audio):
        yield Frame(audio[offset:offset + frame_byte_count], timestamp, frame_duration_s)
        timestamp += frame_duration_s
        offset += frame_byte_count


def vad_segment_generator(wavFile, aggressiveness, frame_duration_ms=30, padding_duration_ms=300):
    wavFile = format_wave(wavFile)
    audio, sample_rate, audio_length = read_wave(wavFile) 
    vad = webrtcvad.Vad(int(aggressiveness))
    frames = frame_generator(frame_duration_ms, audio, sample_rate)
    segments = vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, list(frames))
    return [segment for segment in segments], sample_rate, audio_length


def segmentation(filename, min_duration=10, max_duration=30, aggressiveness=3):
    vad_segments, sample_rate, audio_length = vad_segment_generator(filename, aggressiveness=aggressiveness)
    output = []
    new_start = None
    for i, s in enumerate(vad_segments):
        start, end = float(s.start), float(s.end)
        if new_start is None:
            new_start = start

        duration = end - new_start
        if duration >= min_duration:
            if duration >= max_duration:
                n_splits = int(duration // max_duration) + 1
                split_val = duration / n_splits
                for ni in range(n_splits):
                    output.append((new_start + ni*split_val, new_start + (ni+1)*split_val))
            else:
                output.append((new_start, end))
            new_start = None

    if new_start is not None:
        output.append((new_start, end))

    # print(output)
    # print(max_duration)

    merge_output = []
    current_seg = None
    for seg in output:
        seg = list(seg)
        if current_seg is None:
            current_seg = seg
        
        duration = current_seg[1] - current_seg[0]
        # print(duration, current_seg)
        if duration > max_duration:
            merge_output.append(current_seg)
            current_seg = None
        else:
            if seg[-1] - current_seg[0] <= max_duration:
                current_seg[-1] = seg[-1]
            else:
                merge_output.append(current_seg)
                current_seg = seg

    if current_seg is not None:
        merge_output.append(current_seg)
    return merge_output

def cut_audio(filename,start,end,save_path):
    speech_array, sampling_rate = sf.read(filename)
    start_cut = int(float(start) * sampling_rate)
    end_cut = int(float(end) * sampling_rate)
    cut_audio = speech_array[start_cut:end_cut]
    sf.write(save_path,cut_audio,sampling_rate)


if __name__ == '__main__':
    import torchaudio
    max_length = 155.6485625
    for r, dirs, files in os.walk("./Tin_Tuc"): 
        for file in files:
            if file.endswith("16k.wav"):
                pure_name = os.path.splitext(file)[0]
                basename = os.path.basename(file)
                full_path = os.path.join(r,file)
                dir_name = r.split("/")[-1]
                if not os.path.exists("./Tin_Tuc_segment/"+dir_name):
                    os.makedirs("./Tin_Tuc_segment/"+dir_name, exist_ok=True)
                samples, sr = torchaudio.load(full_path)
                duration = samples.shape[1] / sr
                if duration <= max_length:
                    shutil.copyfile(full_path,f'./Tin_Tuc_segment/{dir_name}/{basename}')
                    continue
                else:
                    vad_segments, sample_rate, audio_length = vad_segment_generator(full_path, aggressiveness=0)
                    # for start, end in segmentation(filename, min_duration=2, max_duration=10):
                    #     print(start, end)
                    start = 0
                    end = 0
                    is_cut = False
                    cur_index = 0
                    for i, s in enumerate(vad_segments):
                        cur_index = i
                        if i == 0:
                            start = s.start
                        if is_cut:
                            start = s.start
                            is_cut = False
                        print(i, s.start, s.end)
                        if (s.end - start) > 120:
                            cut_audio(full_path,start,s.end,f'./Tin_Tuc_segment/{dir_name}/{pure_name}_{i}.wav')
                            is_cut = True
                        end = s.end
                    if not os.path.exists(f'./Tin_Tuc_segment/{dir_name}/{pure_name}_{cur_index}.wav'):
                        cut_audio(full_path,start,end,f'./Tin_Tuc_segment/{dir_name}/{pure_name}_{cur_index}.wav')
    
    
    # vad_segments, sample_rate, audio_length = vad_segment_generator("PTV_dataset/b1/id_198_16k.wav", aggressiveness=0)
    
    # for i, s in enumerate(vad_segments):
    #     print(i, s.start, s.end)
                        

