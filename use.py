import torch
import torch.nn as nn
import torchaudio
from train2 import AudioClassifier


def predict(audio_path, model_path, label_map):
    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = AudioClassifier(num_classes=len(label_map))
    model.load_state_dict(torch.load(model_path))
    model = model.to(device).eval()

    # 预处理流程（必须与训练一致）
    waveform, sr = torchaudio.load(audio_path)

    # 重采样到统一频率
    if sr != 44100:
        resampler = torchaudio.transforms.Resample(sr, 44100)
        waveform = resampler(waveform)

    # 生成梅尔谱图
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100,
        n_fft=4096,
        hop_length=1024,
        n_mels=32,
    )
    mel_spec = transform(waveform)

    # 预处理
    mel_spec = mel_spec.repeat(3, 1, 1)  # 3通道
    mel_spec = nn.AdaptiveAvgPool2d((224, 224))(mel_spec)
    mel_spec = mel_spec.unsqueeze(0).to(device)  # 添加batch维度

    # 预测
    with torch.no_grad():
        outputs = model(mel_spec)
    probs = torch.nn.functional.softmax(outputs, dim=1)

    # 输出结果
    pred_idx = outputs.argmax().item()
    return {
        "label": label_map[pred_idx],
        "confidence": probs[0][pred_idx].item(),
        "all_probabilities": probs.tolist()[0],
    }


# 使用示例
if __name__ == "__main__":
    label_map = {
        0: "Bearded Seal",
        1: "Beluga White Whale",
        2: "Harp Seal",
        3: "Humpback Whale",
        4: "Killer Whale",
        5: "Long-Finned Pilot Whale",
        6: "Northern Right Whale",
        7: "Pantropical Spotted Dolphin",
        8: "Ross Seal",
        9: "Sperm Whale",
        10: "Walrus",
        11: "White-beaked Dolphin",
    }  # 与训练时的label_to_idx对应
    result = predict("test.wav", "audio_classifier.pth", label_map)
    print(f"预测结果: {result['label']} (置信度: {result['confidence']:.2%})")
