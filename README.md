# うごかしかた

1. メディアサーバを立ち上げる（ずっと立ち上げておく必要あり）

Installation > Standalone binary からインストールしておく
https://github.com/bluenviron/mediamtx

```
./mediamtx
```


2. ffmpegを使ってRTSPでストリームを開始する

動画ファイル（ループ再生する）
```
ffmpeg -re -stream_loop -1 -i data/video.mp4 -c copy -f rtsp rtsp://localhost:8554/stream
```

または、カメラから映像、マイクから音声を入力
0はデフォルトのデバイス
```
ffmpeg -f avfoundation -video_size 640x480 -framerate 30 -i "0:0" -f avfoundation -i ":0" -vcodec libx264 -preset veryfast -maxrate 800k -bufsize 1600k -acodec aac -ac 2 -f rtsp rtsp://localhost:8554/stream
```
デバイス名を指定したい場合は次のようにして調べる
```
ffmpeg -f avfoundation -list_devices true -i ""
```


3. 入力をもとに推論する

CPUでも動いた
スペックにもよるが、interval=1だとやや忙しすぎるかもしれない
```
python3 infer_from_stream.py rtsp://localhost:8554/stream --infer-flag --seed 6 --device cpu --interval 2 --preview-video-flag --preview-audio-flag
```
