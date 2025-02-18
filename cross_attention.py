import torch
import torch.nn as nn
import logging

from vggish import VGGish
from torchvggish import vggish

from transformers import  VideoMAEForVideoClassification


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pretrained_weights(model_dict, pretrained_model_dict, model_name):

    # 現在のネットワークモデルのパラメータ名
    param_names = []  # パラメータの名前を格納していく
    for name, param in model_dict.items():
        param_names.append(name)

    # 現在のネットワークの情報をコピーして新たなstate_dictを作成
    new_state_dict = model_dict.copy()

    # 新たなstate_dictに学習済みの値を代入
    print(f"学習済みのパラメータをロードします model: {model_name}", flush=True)
    for index, (key_name, value) in enumerate(pretrained_model_dict.items()):
        if model_name == 'eco':
            if 'fc' in str(key_name):
                continue
        elif model_name == 'vggish':
            if 'fc_final' in str(key_name):
                continue
        elif model_name == 'mae':
            if 'fc_final' in str(key_name):
                continue
            
        name = param_names[index]  # 現在のネットワークでのパラメータ名を取得
        new_state_dict[name] = value  # 値を入れる

        # 何から何にロードされたのかを表示
        # print(str(key_name)+"→"+str(name), flush=True)

    return new_state_dict


# Cross-modal attention model
class CMAModel(nn.Module):
    def __init__(self, seed=6, is_freeze=False):
        super(CMAModel, self).__init__()

        # 1つめのEncoder (この例では動画エンコーダ)
        mae_model = VideoMAEFeatureModel()
        self.mae_model = mae_model

        mae_pretrain = torch.load(f'./models/videomae-{seed}-best.pth')
        new_state_dict = load_pretrained_weights(self.mae_model.state_dict(), mae_pretrain, "mae")
        self.mae_model.load_state_dict(new_state_dict)

        # 2つめのEncoder（この例では音声エンコーダ）
        vggish_model = AudioFeatureModel()
        self.vggish_model = vggish_model

        vggish_pretrain = torch.load(f'./models/vggish-{seed}-best.pth')
        new_state_dict = load_pretrained_weights(self.vggish_model.state_dict(), vggish_pretrain, "vggish")
        self.vggish_model.load_state_dict(new_state_dict)

        if is_freeze:
            print('-- freeze encoder parameter --', flush=True)
            for param in self.mae_model.parameters():
                param.requires_grad = False
            for param in self.vggish_model.parameters():
                param.requires_grad = False

        # Cross-modal attentionのための，Multi-head attention layer
        self.mha1 = nn.MultiheadAttention(embed_dim=512, num_heads=4)
        self.mha2 = nn.MultiheadAttention(embed_dim=512, num_heads=4)

        # Cross-modal attentionの出力から2値のベクトルを出力する2層の多層パーセプトロン
        self.fusion_layer = nn.Linear(in_features=1024, out_features=512, bias=True)
        self.fc_final = nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self, x1, x2):
        out_video = self.mae_model(x1)
        out_audio = self.vggish_model(x2)

        attn_output_v, attn_weights_v = self.mha1(out_video, out_audio, out_audio)
        attn_output_a, attn_weights_a = self.mha2(out_audio, out_video, out_video)

        out_video = attn_output_v + out_video
        out_audio = attn_output_a + out_audio

        out = torch.cat([out_video, out_audio], dim=1)

        out = self.fusion_layer(out)

        out = self.fc_final(out)

        return out


# Video Encoder（512次元の動画特徴を出力）
class VideoMAEFeatureModel(nn.Module):
    def __init__(self):
        super(VideoMAEFeatureModel, self).__init__()

        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

        model.classifier = torch.nn.Linear(768, 768)

        self.main_model = model

        self.fc_final = nn.Linear(in_features=768, out_features=512, bias=True)

    def forward(self, x):

        bs = x.shape[0]
        out = x.view(bs, x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1])
        out = self.main_model(out)
        out = out.logits
        out = self.fc_final(out)
        return out


# Audio Encoder（512次元の音声特徴を出力）
class AudioFeatureModel(nn.Module):
    def __init__(self):
        super(AudioFeatureModel, self).__init__()

        self.vggish = VGGish()

        pretrained_model = vggish()
        pretrained_model_dict = pretrained_model.state_dict()

        # 現在のモデルの変数名などを取得
        model_dict = self.vggish.state_dict()

        # 学習済みモデルのstate_dictを取得
        new_state_dict = load_pretrained_weights(model_dict, pretrained_model_dict, "vggish")

        # 学習済みモデルのパラメータを代入
        self.vggish.load_state_dict(new_state_dict)

        self.fc_middle = nn.Linear(in_features=1280, out_features=512, bias=True)

        self.fc_final = nn.Linear(in_features=512, out_features=512, bias=True)

    def forward(self, x):

        bs = x.shape[0]
        out = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        out = self.vggish(out)

        # バッチに戻す
        out = out.view(bs, x.shape[1], 128)

        out = torch.flatten(out, 1)

        out = self.fc_middle(out)

        out = self.fc_final(out)

        return out
    

class CMAFeatureModel(nn.Module):
    def __init__(self, seed=25, is_freeze=True):
        super(CMAFeatureModel, self).__init__()

        mae_model = VideoMAEFeatureModel()
        self.mae_model = mae_model

        mae_pretrain = torch.load(f'./models/videomae-arc-{seed}-best.pth')

        new_state_dict = load_pretrained_weights(self.mae_model.state_dict(), mae_pretrain, "feature")
        self.mae_model.load_state_dict(new_state_dict)

        vggish_model = AudioFeatureModel()
        self.vggish_model = vggish_model

        vggish_pretrain = torch.load(f'./models/vggish-arc-{seed}-best.pth')

        new_state_dict = load_pretrained_weights(self.vggish_model.state_dict(), vggish_pretrain, "feature")
        self.vggish_model.load_state_dict(new_state_dict)

        if is_freeze:
            print('-- freeze encoder parameter --', flush=True)
            for param in self.mae_model.parameters():
                param.requires_grad = False
            for param in self.vggish_model.parameters():
                param.requires_grad = False

        self.mha1 = nn.MultiheadAttention(embed_dim=512, num_heads=4)

        self.mha2 = nn.MultiheadAttention(embed_dim=512, num_heads=4)

        self.fusion_layer = nn.Linear(in_features=1024, out_features=512, bias=True)

        self.fc_final = nn.Linear(in_features=512, out_features=512, bias=True)

    def forward(self, x1, x2):

        out_video = self.mae_model(x1)
        out_audio = self.vggish_model(x2)

        attn_output_v, attn_weights_v = self.mha1(out_video, out_audio, out_audio)
        attn_output_a, attn_weights_a = self.mha2(out_audio, out_video, out_video)

        out_video = attn_output_v + out_video
        out_audio = attn_output_a + out_audio

        out = torch.cat([out_video, out_audio], dim=1)

        out = self.fusion_layer(out)

        out = self.fc_final(out)

        return out


class VideoMAEModel(nn.Module):
    def __init__(self):
        super(VideoMAEModel, self).__init__()

        # print(f'model: {model_name}', flush=True)
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        model.classifier = torch.nn.Linear(768, 768)
        self.fc_middle = nn.Linear(in_features=768, out_features=512, bias=True)

        self.fc_final = nn.Linear(in_features=512, out_features=2, bias=True)

        self.main_model = model

    def forward(self, x):
        bs = x.shape[0]
        out = x.view(bs, x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1])
        out = self.main_model(out)

        out = out.logits

        out = self.fc_middle(out)
        out = self.fc_final(out)

        return out


class VggishModel(nn.Module):
    def __init__(self):
        super(VggishModel, self).__init__()

        # Vggish
        self.vggish = VGGish()

        # self.conv1d = nn.Conv1d(10, 1, 1)
        self.fc_middle = nn.Linear(in_features=1280, out_features=512, bias=True)

        # クラス分類の全結合層
        self.fc_final = nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self, x):

        bs = x.shape[0]
        out = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        out = self.vggish(out)

        # バッチに戻す
        out = out.view(bs, x.shape[1], 128)

        out = torch.flatten(out, 1)

        out = self.fc_middle(out)

        out = self.fc_final(out)

        return out