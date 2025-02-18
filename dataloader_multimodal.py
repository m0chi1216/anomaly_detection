import os
import re

import numpy as np
import torch
import torch.utils.data
import av

from av.video.frame import VideoFrame
from transformers import VideoMAEImageProcessor

from torchvggish import vggish, vggish_input


def make_audio_list(root_path):

    # 動画を画像データにしたフォルダへのファイルパスリスト
    audio_list = list()

    # root_pathにある、クラスの種類とパスを取得
    class_list = os.listdir(path=root_path)

    # 各クラスの動画ファイルを画像化したフォルダへのパスを取得
    for class_list_i in (class_list):  # クラスごとのループ

        # クラスのフォルダへのパスを取得
        class_path = os.path.join(root_path, class_list_i)
        class_path = os.path.join(class_path, "audio")

        # 各クラスのフォルダ内の画像フォルダを取得するループ
        for file_name in os.listdir(class_path):

            # ファイル名と拡張子に分割
            name, ext = os.path.splitext(file_name)

            if ext != '.wav':
                continue

            audio_list.append(os.path.join(class_path, file_name))

    return audio_list


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def read_video_pyav(container, indices, frame_width=224):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    face_num = []
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            # frame = frame.reformat(width=frame_width, height=frame_width)
            img = frame.to_image()
            img = img.resize((int(img.width * (frame_width / img.height)), frame_width))
            img = crop_center(img, frame_width, frame_width)
            frame = VideoFrame.from_image(img)
            frames.append(frame)

    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    # end_idx = np.random.randint(converted_len, seg_len)
    end_idx = seg_len
    # start_idx = end_idx - converted_len
    start_idx = 0
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


class VideoDataset(torch.utils.data.Dataset):
    """
    動画のDataset
    """

    def __init__(self, video_list, audio_list):
        self.video_list = video_list  # 動画画像のフォルダへのパスリスト
        self.audio_list = audio_list

    def __len__(self):
        '''動画の数を返す'''
        return len(self.video_list)

    def __getitem__(self, index):
        video_dir_path = self.video_list[index]  # 画像が格納されたフォルダ
        if 'abnormal' in video_dir_path:
            label_id = 1
        elif 'normal' in video_dir_path:
            label_id = 0

        # 3. 前処理を実施
        # imgs_transformed = self.transform(img_group, phase=self.phase)
        container = av.open(video_dir_path)
        clip_len = 16
        indices = sample_frame_indices(clip_len=clip_len, frame_sample_rate=int(container.streams.video[0].frames / clip_len), seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container, indices, frame_width=224)

        image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        video_embeddings = image_processor(list(video), return_tensors="pt")
        video_embeddings = list(video_embeddings.data.values())[0]

        audio_dir_path = self.audio_list[index]

        audio_embeddings = vggish_input.wavfile_to_examples(audio_dir_path)
        if (audio_embeddings.shape[0] > 10):
            audio_embeddings = audio_embeddings[-10:]
        elif (audio_embeddings.shape[0] < 10):
            while audio_embeddings.shape[0] != 10:
                audio_embeddings = torch.cat([audio_embeddings[:1], audio_embeddings], dim=0)

        audio_embeddings = audio_embeddings.detach()

        return video_embeddings, audio_embeddings, label_id


