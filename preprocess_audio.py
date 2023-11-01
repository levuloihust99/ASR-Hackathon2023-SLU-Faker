import os
import glob
import numpy as np
import soundfile as sf
import math
from vad import vad_segment_generator
from decouple import config
import argparse
class NormalizeSegment(object):
    """
    Normalize segment can be combined from other segments to maximize length and minimize data loss due to vad
    """
    audio_path: str

    def __init__(self, start, end, speech, length):
        self.start = start
        self.end = end
        self.speech = speech
        self.length = length  # Độ dài này là độ dài thật của audio này đã cắt hết toàn bộ phần ko có tiếng nói, do đó
        # không thể lấy start - end để lấy length được

def segment_large_file_ver_2(filename, tmp_dir='public_test_16k_1ac_vad', aggressiveness=2):
    """

    :param filename:
    :param tmp_dir:
    :param aggressiveness: 2, giá trị càng nhỏ thì sẽ cắt càng chi tiết, có thể lọt nhiễu nền
    :return:
    """
    audio,_ = sf.read(filename)
    basename  = os.path.basename(filename)
    vad_segments, sample_rate, audio_length = vad_segment_generator(filename, aggressiveness=aggressiveness)
    output = []  # [(start, end, audio_path)]
    count = 1
    tmp_norm_seg = None  # NormalizeSegment
    
    full_audio = []
    for i, s in enumerate(vad_segments):
        # Lỗi ở đây là start, end thì theo audio gốc và start cái sau không liền end cái trước (khoảng trống ở giữa thì mất dữ liệu),
        # Trong khi đó, vòng for chạy qua nhiều segment thì tính duration lại lấy end segment hiện tại từ start segment trước
        # khiến thực tế tính độ dài segment không chính xác, đồng thời gây mất mát dữ liệu do vad bỏ những đoạn rất ngắn nhưng
        # ảnh hưởng trực tiếp đến audio vì bị mất âm các từ
        # Chỉnh sửa: thêm 1 yếu tố nữa là khoảng cách giữa các segment, nếu >5s thì lấy s.bytes, còn k thì merge cơ học, k bỏ các phần audio mà vad coi là k có âm thanh
        start, end = float(s.start), float(s.end)
        duration = end - start
        
        if tmp_norm_seg:
            distance = start - tmp_norm_seg.end
            if distance > 3:
                # write_wave(tmp_dir + f'/{count}.wav', b''.join([f for f in tmp_norm_seg.speech]), sample_rate)
                if tmp_norm_seg.length >= 1:
                    # sf.write(tmp_dir + f'/{count}.wav', tmp_norm_seg.speech, sample_rate)
                    full_audio.append(tmp_norm_seg.speech)
                    output.append((tmp_norm_seg.start, tmp_norm_seg.end, tmp_dir + f'/{count}.wav'))
                    count += 1

                tmp_norm_seg = NormalizeSegment(start, end, audio[int(start * sample_rate):int(end * sample_rate)],
                                                length=duration)
            else:
                tmp_norm_seg.end = end
                tmp_norm_seg.length += duration
                tmp_norm_seg.speech = audio[int(tmp_norm_seg.start * sample_rate):int(tmp_norm_seg.end * sample_rate)]

        else:
            tmp_norm_seg = NormalizeSegment(start, end,
                                            audio[int(start * sample_rate):int(end * sample_rate)],
                                            length=duration)
            
        if tmp_norm_seg.length > 25:
            n_splits = int(tmp_norm_seg.length // 25) + 1
            split_data = math.floor(len(tmp_norm_seg.speech) / n_splits)
            split_val = tmp_norm_seg.length / n_splits
            cur_split_data = 0
            for ni in range(n_splits):
                if ni == n_splits - 1:
                    #sf.write(tmp_dir + f'/{count}.wav',tmp_norm_seg.speech[cur_split_data:],sample_rate)
                    full_audio.append(tmp_norm_seg.speech)
                    output.append(
                        (tmp_norm_seg.start + ni * split_val, tmp_norm_seg.end, tmp_dir + f'/{count}.wav'))
                else:
                    next_split_data = cur_split_data + split_data
                    # sf.write(tmp_dir + f'/{count}.wav',tmp_norm_seg.speech[cur_split_data:next_split_data],sample_rate)
                    full_audio.append(tmp_norm_seg.speech)
                    output.append(
                        (tmp_norm_seg.start + ni * split_val, tmp_norm_seg.start + (ni + 1) * split_val,
                         tmp_dir + f'/{count}.wav'))
                    cur_split_data = next_split_data
                count += 1

            tmp_norm_seg = None

        elif tmp_norm_seg.length >= 12 or i == len(vad_segments) - 1:
            # write_wave(tmp_dir + f'/{count}.wav', b''.join([f for f in tmp_norm_seg.speech]), sample_rate)
            # sf.write(tmp_dir + f'/{count}.wav', tmp_norm_seg.speech, sample_rate)
            full_audio.append(tmp_norm_seg.speech)
            output.append((tmp_norm_seg.start, tmp_norm_seg.end, tmp_dir + f'/{count}.wav'))
            count += 1
            tmp_norm_seg = None
    try:
        
        if len(full_audio) > 1:
            final_audio = full_audio[0]
            
            for i in range(1,len(full_audio)):
                final_audio = np.concatenate((final_audio,full_audio[i]),axis=None)
            sf.write(tmp_dir + f'/{basename}', final_audio, sample_rate,"PCM_16")
        else:
            full_audio = np.array(full_audio,dtype=np.float64)
            
            full_audio = full_audio.reshape(-1)
            
            sf.write(tmp_dir + f'/{basename}', full_audio, sample_rate,"PCM_16")
    except:
        print(basename)
        
    return output

def convert_16k_1ac(folder):
    full_paths = os.path.join(config('ORIGINAL_DATA'),folder)
    audio_paths = glob.glob(f'{full_paths}/*')
    if not os.path.exists(f'{folder}_16k_1ac'):
        os.makedirs(f'{folder}_16k_1ac',exist_ok=True)
    for audio_path in audio_paths:
        base_name = os.path.basename(audio_path)
        save_name = f'{folder}_16k_1ac/{base_name}'
        command = f'ffmpeg -i {audio_path} -ar 16000 -ac 1 -resampler soxr {save_name}'
        os.system(command)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode","-m",default="train",type=str)
    args = parser.parse_args()
    if args.mode == "train":
        folder = "Train"
    elif args.mode == "public":
        folder = "public_test"
    elif args.mode == "private":
        folder = "private_test"
    
    
    convert_16k_1ac(folder)
    audio_formats = glob.glob(f"{folder}_16k_1ac/*")
    if not os.path.exists(f'{folder}_16k_1ac_vad'):
        os.makedirs(f'{folder}_16k_1ac_vad',exist_ok=True)
    for audio in audio_formats:
        out = segment_large_file_ver_2(audio,tmp_dir=f'{folder}_16k_1ac_vad')