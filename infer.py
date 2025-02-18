import sys
import time
import asyncio
import logging
import pickle

import av
import cv2
import numpy as np
import pyaudio

from multiprocessing import Process, Queue
from sklearn.neighbors import NearestNeighbors

from models import CMAFeatureModel
from dataloader_multimodal import (
        VideoMAEImageProcessor, VideoFrame, crop_center, torch, vggish_input,
        )
from websocket_client import send_result


CLIP_LEN = 16
FRAME_WIDTH = 224

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameBuffer:
    def __init__(self, window: int = 10):
        '''
        window: 取得したいフレームの期間（直近N秒間）
        '''
        self.frames: list[av.VideoFrame | av.AudioFrame] = []
        self.window = window

    def add_frame(self, frame: av.VideoFrame | av.AudioFrame) -> None:
        '''
        フレームを追加する
        '''
        if frame.time:
            self.frames.append(frame)
            if frame.time > self.window * 2:
                self.cleanup_frames(frame.time - self.window)

    def get_recent_frames(self, current_time: float | None) -> list[av.VideoFrame | av.AudioFrame]:
        '''
        直近のwindow秒間のフレームを取得する
        '''
        if current_time:
            recent_frames = [frame for frame in self.frames if frame.time >= current_time - self.window]
            return recent_frames
        else:
            return []

    def cleanup_frames(self, current_time: float | None = None) -> None:
        '''
        ウィンドウ期間外の古いフレームを削除する
        '''
        self.frames = self.get_recent_frames(current_time)


def inference(current_time, video_embeddings, audio_embeddings, seed, device, robot_id):
    global model, train_labels, train_reasons, nbrs
    model.eval()
    logger.info('Start inference...')
    with torch.no_grad():       
        video_embeddings, audio_embeddings = video_embeddings.to(device), audio_embeddings.to(device)
        model.load_state_dict(torch.load(f'./checkpoints/rank-{seed}-best.pth'))
        output = model(video_embeddings, audio_embeddings).detach().cpu().numpy()

        indices = nbrs.kneighbors(output, return_distance=False)
        labels = [int(train_labels[idx]) for idx in indices[0]]
        reasons = [str(train_reasons[idx]) for idx in indices[0]]
        reason = ""
        score = labels.count(0)
        if score >= 5:
            predicted = 0
        else:
            predicted = 1
            for j in range(len(labels)):
                if labels[j] == 1:
                    reason = reasons[j]
                    break

        if predicted:
            logger.info(f'\x1b[41m[{current_time}] ID: {robot_id}, 推論結果: {predicted}, 理由: {reason}\x1b[0m')
        else:
            logger.info(f'\x1b[44m[{current_time}] ID: {robot_id}, 推論結果: {predicted}, 理由: {reason}\x1b[0m')


def infer(
        input_queue: Queue, output_queue: Queue,
        seed: int, device: str, robot_id: int,
        ) -> int:
    '''
    推論する
    '''
    # モデルを読み込む
    try:
        model = CMAFeatureModel(seed=seed, is_freeze=True)
        model = model.to(device)
        model.load_state_dict(torch.load(f'./checkpoints/rank-{seed}-best.pth'))
        train_embeddings = np.load(f'./embeddings/rank-embedding-{seed}.npy')
        train_labels = np.load(f'./embeddings/rank-label-{seed}.npy')
        f = open(f'./embeddings/rank-reason-{seed}.txt', 'rb')
        train_reasons = pickle.load(f)
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(train_embeddings)
        logger.info('モデルのロード完了')

    except Exception as error:
        logger.error(error)
        logger.info('モデルを読み込めませんでした')
        return

    model.eval()
    with torch.no_grad():
        while True:
            if not input_queue.empty():
                # logger.info('Start Inference...')
                data = input_queue.get()
                try:
                    current_time = data["current_time"]
                    video_embeddings = data["video_embeddings"]
                    audio_embeddings = data["audio_embeddings"]
                except:
                    logger.warning(f"[{robot_id}] Can't get data...")
                    input_queue.put(None)
                    continue

                if video_embeddings is None or audio_embeddings is None:
                    input_queue.put(None)
                    continue
                
                video_embeddings, audio_embeddings = video_embeddings.to(device), audio_embeddings.to(device)
                model.load_state_dict(torch.load(f'./checkpoints/rank-{seed}-best.pth'))
                output = model(video_embeddings, audio_embeddings).detach().cpu().numpy()

                indices = nbrs.kneighbors(output, return_distance=False)
                labels = [int(train_labels[idx]) for idx in indices[0]]
                reasons = [str(train_reasons[idx]) for idx in indices[0]]
                reason = ""
                logger.info(f'{labels.count(0)=}')
                score = labels.count(0)
                if score >= 5:
                    predicted = 0
                else:
                    predicted = 1
                    for j in range(len(labels)):
                        if labels[j] == 1:
                            reason = reasons[j]
                            break

                output_queue.put({
                    "current_time": current_time,
                    "predicted": predicted,
                    "score": score,
                    "reason": reason,
                    })
                # logger.info(f'{predicted=}, {output[0][:3]=}')
                # asyncio.run(send_result({"predicted": str(predicted), "reason": str(reason), "robotId": str(robot_id)}))

                # if predicted:
                #     logger.info(f'\x1b[41m[{current_time}] ID: {robot_id}, 推論結果: {predicted}, 理由: {reason}\x1b[0m')
                # else:
                #     logger.info(f'\x1b[44m[{current_time}] ID: {robot_id}, 推論結果: {predicted}, 理由: {reason}\x1b[0m')
            else:
                # logger.warning(f'[{robot_id}] input queue is empty')
                pass


def preview_video(queue: Queue) -> None:
    '''
    映像をプレビューする
    '''
    while True:
        try:
            frame_data = queue.get()
        except:
            pass
        else:
            if frame_data is None:
                break

            # ウィンドウに表示する
            # frame_data = cv2.resize(frame_data, (320, 240))
            cv2.imshow('frame', frame_data)
            cv2.waitKey(1)


def preview_audio(queue: Queue, channels: int, rate: int) -> None:
    '''
    音声をプレビューする
    '''
    audio = pyaudio.PyAudio()
    # 何も指定しない（現状）とデフォルトのスピーカーで出力される
    out_audio_stream = audio.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            output=True,
            )

    while True:
        try:
            frame_data = queue.get()
        except:
            pass
        else:
            if frame_data is None:
                break

            # 再生する
            out_audio_stream.write(frame_data)


def preprocess(
        container: av.container.Container,
        video_frames: list[av.VideoFrame],
        audio_frames: list[av.AudioFrame],
        current_time: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    動画と音声を埋め込み表現にする
    '''
    # 動画の前処理
    # video_framesから等間隔に16フレーム分をサンプリング
    indices = np.linspace(0, len(video_frames), num=CLIP_LEN)
    indices = np.clip(indices, 0, len(video_frames) - 1).astype(np.int64)
    video_frames = [video_frames[index] for index in indices]
    assert len(video_frames) == CLIP_LEN

    video = []
    for frame in video_frames:
        img = frame.to_image()
        img = img.resize((int(img.width * (FRAME_WIDTH / img.height)), FRAME_WIDTH))
        img = crop_center(img, FRAME_WIDTH, FRAME_WIDTH)
        frame_data = VideoFrame.from_image(img)
        video.append(frame_data)
    video = np.stack([x.to_ndarray(format='rgb24') for x in video])

    image_processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics')
    video_embeddings = image_processor(list(video), return_tensors='pt')
    video_embeddings = list(video_embeddings.data.values())[0]

    # 音声の前処理
    audio = []
    for frame in audio_frames:
        frame_data = np.array(frame.to_ndarray(), dtype='float64')
        # FLTPの場合はインターリーブが必要
        if frame.format.name == 'fltp':
            frame_data = np.stack([frame_data[ch] for ch in range(frame_data.shape[0])], axis=-1)
        audio.append(frame_data)
    audio = np.array(audio).reshape(-1, 2)

    audio_embeddings = vggish_input.waveform_to_examples(audio, container.streams.audio[0].sample_rate)
    if (audio_embeddings.shape[0] > 10):
        audio_embeddings = audio_embeddings[-10:]
    elif (audio_embeddings.shape[0] < 10):
        while audio_embeddings.shape[0] != 10:
            audio_embeddings = torch.cat([audio_embeddings[:1], audio_embeddings], dim=0)

    audio_embeddings = audio_embeddings.detach()
    audio_embeddings = audio_embeddings.reshape(-1, *audio_embeddings.shape)

    # logger.info(f"embeddings shape: {video_embeddings.shape}, {audio_embeddings.shape}")

    return video_embeddings, audio_embeddings


def infer_from_stream(
        input_source: str, infer_flag: bool, seed: int, device: str, window: int, interval: int, robot_id: int,
        ) -> None:
    '''
    input_sourceからinterval秒ごとに直近window秒のフレームを取得して推論する
    動画はウィンドウに表示される（処理が重いとカクカクするので注意）
    音声はデフォルトのスピーカーで出力される（discordとかに繋いでると出力されないので注意）
    '''
    # モデルを読み込む
    try:
        model = CMAFeatureModel(seed=seed, is_freeze=True)
        model = model.to(device)
        model.load_state_dict(torch.load(f'./checkpoints/rank-{seed}-best.pth'))
        train_embeddings = np.load(f'./embeddings/rank-embedding-{seed}.npy')
        train_labels = np.load(f'./embeddings/rank-label-{seed}.npy')
        f = open(f'./embeddings/rank-reason-{seed}.txt', 'rb')
        train_reasons = pickle.load(f)
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(train_embeddings)
        model.eval()
        logger.info('モデルのロード完了')

    except Exception as error:
        logger.error(error)
        logger.info('モデルを読み込めませんでした')
        return

    while True:
        # 推論するプロセス
        logger.info(f'[{robot_id}] Start Process...')

        # 取得したフレームを格納する変数
        video_frame_buffer = FrameBuffer(window)
        audio_frame_buffer = FrameBuffer(window)

        # 音声と動画のストリームに接続する
        # RTSPじゃなくて普通の音声付きの動画ファイルを入力してもOK
        get_stream = False
        while not get_stream:
            try:
                container = av.open(input_source)
                get_stream = True
            except av.error.HTTPNotFoundError:
                logger.error(f"[{robot_id}] Can't get stream...")
                time.sleep(10)
        
        if get_stream:
            assert container.streams.audio
            assert container.streams.video
            audio_stream = container.streams.audio[0]
            video_stream = container.streams.video[0]

            start_time = 0.0
            current_time = 0.0

            try:
                # ストリームの読み込みを開始する
                for frame in container.decode(video_stream, audio_stream):
                    # 映像フレームの場合
                    if isinstance(frame, av.VideoFrame):
                        video_frame_buffer.add_frame(frame)

                    # 音声フレームの場合
                    elif isinstance(frame, av.AudioFrame):
                        audio_frame_buffer.add_frame(frame)

                    else:
                        continue

                    if frame.time:
                        current_time = frame.time

                    # 指定時間が経過
                    if current_time - start_time >= interval and current_time >= window:
                        start_time = current_time

                        window_video_frames = video_frame_buffer.get_recent_frames(current_time)
                        window_audio_frames = audio_frame_buffer.get_recent_frames(current_time)

                        # ストリームの読み込み開始直後は動画フレームが読み込めないことが多いため
                        # 動画フレームの数が0になっていることがある
                        # logger.info(f'[{current_time}] {interval}秒経過')
                        logger.info(f'[{robot_id}] video:{len(window_video_frames)}, audio:{len(window_audio_frames)}')

                        # 推論するプロセスに渡す
                        if infer_flag and len(window_video_frames) >= CLIP_LEN and len(window_audio_frames) > 0:
                            video_embeddings, audio_embeddings = preprocess(container, window_video_frames, window_audio_frames, current_time)

                            with torch.no_grad():
                                video_embeddings, audio_embeddings = video_embeddings.to(device), audio_embeddings.to(device)
                                model.load_state_dict(torch.load(f'./checkpoints/rank-{seed}-best.pth'))
                                output = model(video_embeddings, audio_embeddings).detach().cpu().numpy()

                                indices = nbrs.kneighbors(output, return_distance=False)
                                labels = [int(train_labels[idx]) for idx in indices[0]]
                                reasons = [str(train_reasons[idx]) for idx in indices[0]]
                                reason = ""
                                # logger.info(f'{labels.count(0)=}')
                                score = labels.count(0)
                                if score > 5:
                                    predicted = 0
                                else:
                                    predicted = 1
                                    for j in range(len(labels)):
                                        if labels[j] == 1:
                                            reason = reasons[j]
                                            break
                            
                            try:
                                asyncio.run(send_result({"predicted": str(predicted), "score": score, "reason": str(reason), "robotId": str(robot_id)}))
                            except:
                                logger.warning(f'[{robot_id}] Couldnt send result')

                            if predicted:
                                logger.info(f'\x1b[41m[{robot_id}] 推論結果: {predicted}, スコア: {score}, 理由: {reason}\x1b[0m')
                            else:
                                logger.info(f'\x1b[44m[{robot_id}] 推論結果: {predicted}, スコア: {score}\x1b[0m')

                            # video_frame_buffer.cleanup_frames()
                            # audio_frame_buffer.cleanup_frames()

            # Ctrl+Cで止める
            except KeyboardInterrupt:
                # cleanup(infer_flag, infer_input_queue, infer_output_queue, infer_process)
                sys.exit()
    

def cleanup(infer_flag: bool, infer_input_queue: Queue, infer_output_queue: Queue, infer_process: Process):
    '''
    プロセスを終了する
    '''
    if infer_flag:
        infer_input_queue.put(None)
        infer_output_queue.put(None)
        infer_process.join()
        infer_process.terminate()

    sys.exit()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #         'input_source', type=str, default='rtsp://localhost:8554/stream', help='入力ソース')
    parser.add_argument(
            '--infer-flag', action='store_true', help='推論するかどうか')
    parser.add_argument(
            '--seed', type=int, default=96, help='モデル使用時のシード値')
    parser.add_argument(
            '--device', type=str, default='cuda:0', help='モデル使用時のデバイス')
    parser.add_argument(
            '--window', type=int, default=10, help='取得したいフレームの期間（直近N秒間）')
    parser.add_argument(
            '--interval', type=int, default=8, help='推論を行う間隔')
    parser.add_argument(
            '--preview-video-flag', action='store_true', help='入力映像をプレビューするどうか')
    parser.add_argument(
            '--preview-audio-flag', action='store_true', help='入力音声をプレビューするどうか')
    parser.add_argument(
            '--robot-id', type=int, default=0, help='対象ロボットのID')
    args = parser.parse_args()

    process0 = Process(target=infer_from_stream, args=('rtsp://localhost:8554/stream0', args.infer_flag, args.seed, args.device, args.window, args.interval, 0),)
    process0.start()

    process1 = Process(target=infer_from_stream, args=('rtsp://localhost:8554/stream1', args.infer_flag, args.seed, args.device, args.window, args.interval, 1),)
    process1.start()

    process2 = Process(target=infer_from_stream, args=('rtsp://localhost:8554/stream2', args.infer_flag, args.seed, args.device, args.window, args.interval, 2),)
    process2.start()

    process3 = Process(target=infer_from_stream, args=('rtsp://localhost:8554/stream3', args.infer_flag, args.seed, args.device, args.window, args.interval, 3),)
    process3.start()

    process4 = Process(target=infer_from_stream, args=('rtsp://localhost:8554/stream4', args.infer_flag, args.seed, args.device, args.window, args.interval, 4),)
    process4.start()

    process5 = Process(target=infer_from_stream, args=('rtsp://localhost:8554/stream5', args.infer_flag, args.seed, args.device, args.window, args.interval, 5),)
    process5.start()
