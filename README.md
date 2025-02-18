## 概要

ローカルのRTSPサーバに送られたインタラクション映像を入力として，インタラクションに問題が生じているかを二値で判定するスクリプトです．<br>
最大で6つの映像を並列に処理できます．<br>
リアルタイムに推論する場合，GPUが必要です．3並列程度であればGPU搭載のノートPC上でも動作しますが，6並列の場合はAWSのEC2インスタンス (g4dn.xlargeなど) を使用したほうが良いです．

## 準備
- Python 3.10での動作を確認済み
- パッケージのインストール
```
pip install -r requirements.txt
```

- モデル等の配置
  - `checkpoints/`
    - 推論用のモデル (.pth) を配置してください
  - `embeddings/`
    - 訓練データの埋め込み (.npy) を配置してください
    - アノテーションの理由が書かれたテキストファイル (.txt) も配置してください

## 手順

1. [こちら](https://github.com/m0chi1216/interaction_video_server)のリポジトリを使用して，ローカルのRTSPサーバに映像が送られるようにする
2. 推論用スクリプトを実行する
```
python3 infer.py --infer-flag --seed 96 --window 10 --interval 8
```