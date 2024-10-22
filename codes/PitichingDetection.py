import cv2
import torch
import torch.nn as nn
import numpy as np
from transformers import ViTModel

# Transformer Encoder (Vision Transformer) モデル
class PitchingDetector(nn.Module):
    def __init__(self):
        super(PitchingDetector, self).__init__()
        # Vision Transformer (事前学習済みのモデルを使う)
        self.transformer = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.fc = nn.Linear(768, 1)  # 768次元の特徴量からピッチング確率を算出

    def forward(self, x):
        outputs = self.transformer(x).last_hidden_state[:, 0, :]  # CLSトークンを使用
        prob = torch.sigmoid(self.fc(outputs))  # ピッチングの確率を算出
        return prob

# モデルの初期化
model = PitchingDetector()
model.eval()

# 動画からフレームごとの特徴量を抽出する関数
def extract_frame_features(frame):
    # フレームをRGBに変換し、リサイズ
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(frame_rgb, (224, 224))
    # 画素値を正規化してテンソル化
    tensor_frame = torch.tensor(resized_frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor_frame

# 動画の読み込み
video_path = "../../baseball_video/video_.mp4"
video_capture = cv2.VideoCapture(video_path)

# 各フレームに対して処理を行う
frame_probs = []
success, frame = video_capture.read()

while success:
    # フレームをテンソル形式に変換して特徴量を抽出
    frame_tensor = extract_frame_features(frame)

    # Transformer Encoderにフレームを入力して、ピッチングの確率を予測
    with torch.no_grad():
        prob = model(frame_tensor).item()

    # ピッチングの確率をリストに追加
    frame_probs.append(prob)

    # 次のフレームを読み込む
    success, frame = video_capture.read()

# 結果の表示
for idx, prob in enumerate(frame_probs):
    print(f"Frame {idx}: Pitching Probability = {prob:.4f}")

# 動画を閉じる
video_capture.release()