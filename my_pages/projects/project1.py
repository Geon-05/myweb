import os
import glob
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights

import subprocess

# Streamlit Secrets로 Kaggle API Key 설정
os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE"]["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE"]["KAGGLE_KEY"]

# Kaggle 데이터셋 다운로드
DATASET_NAME = "geon05/dacon_image/pyTorch/default"

# (1) 다운로드할 폴더 지정
DOWNLOAD_DIR = "downloads"   # 원하는 폴더 경로(없으면 자동 생성됨)
ZIP_FILENAME = "dacon_image.zip"  # 실제로 생성될 zip 파일명(데이터셋 이름과 동일하거나, Kaggle에서 제공하는 파일명과 맞춰주세요)

# (2) Kaggle 데이터셋 다운로드
subprocess.run([
    "kaggle", "datasets", "download",
    "-d", DATASET_NAME,
    "-p", DOWNLOAD_DIR,   # 원하는 다운로드 경로
    "--unzip"             # 자동으로 압축해제까지 원하시면 주석 해제
])

# 모델 파일 경로
model_path = "downloads/color.pth"

def app():
    st.title("이미지 색상화 및 손실 부분 복원 AI 경진대회")
    st.write("알고리즘 | 월간 데이콘 | 비전 | 이미지 복원 | 이미지 색상화 | SSIM")
    st.image("imgpro/img/damage.png", caption="Project 1 Image")
    
    # -----------------------
    # (1) 학습 때 쓰인 함수/클래스 재정의
    # -----------------------
    def lab_to_rgb(L, a, b):
        lab_0_255 = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.float32)
        lab_0_255[:,:,0] = L * 255.0
        lab_0_255[:,:,1] = a * 128.0 + 128.0
        lab_0_255[:,:,2] = b * 128.0 + 128.0
        lab_0_255 = np.clip(lab_0_255, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(lab_0_255, cv2.COLOR_Lab2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb / 255.0

    class UpConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        def forward(self, x):
            return self.up(x)

    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        def forward(self, x):
            return self.conv(x)

    class ResNetEncoder(nn.Module):
        def __init__(self, pretrained=True):
            super().__init__()
            net = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            self.initial = nn.Sequential(net.conv1, net.bn1, net.relu)
            self.maxpool = net.maxpool
            self.layer1 = net.layer1
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            self.layer4 = net.layer4

        def forward(self, x):
            x0 = self.initial(x)
            x1 = self.maxpool(x0)
            x1 = self.layer1(x1)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            return x0, x1, x2, x3, x4

    class ResNetUNet(nn.Module):
        def __init__(self, out_ch=2, pretrained=False):
            super().__init__()
            self.encoder = ResNetEncoder(pretrained=pretrained)
            self.up3 = UpConv(2048, 1024)
            self.dec3 = DoubleConv(2048, 1024)
            self.up2 = UpConv(1024, 512)
            self.dec2 = DoubleConv(1024, 512)
            self.up1 = UpConv(512, 256)
            self.dec1 = DoubleConv(512, 256)
            self.up0 = UpConv(256, 64)
            self.dec0 = DoubleConv(128, 64)
            self.up_final = UpConv(64,64)
            self.dec_final = DoubleConv(64,64)
            self.final_out = nn.Conv2d(64, out_ch, 1)

        def forward(self, x):
            # 입력 텐서 x는 [B,1,H,W] 형태이며, 
            # RGB 특징 추출을 위해 동일 채널로 3채널 확장
            x = x.repeat(1,3,1,1)
            x0, x1, x2, x3, x4 = self.encoder(x)

            x_up3 = self.up3(x4)
            x_cat3 = torch.cat([x_up3, x3], dim=1)
            x_dec3 = self.dec3(x_cat3)

            x_up2 = self.up2(x_dec3)
            x_cat2 = torch.cat([x_up2, x2], dim=1)
            x_dec2 = self.dec2(x_cat2)

            x_up1 = self.up1(x_dec2)
            x_cat1 = torch.cat([x_up1, x1], dim=1)
            x_dec1 = self.dec1(x_cat1)

            x_up0 = self.up0(x_dec1)
            x_cat0 = torch.cat([x_up0, x0], dim=1)
            x_dec0 = self.dec0(x_cat0)

            x_upf = self.up_final(x_dec0)
            x_decf = self.dec_final(x_upf)
            out = self.final_out(x_decf)
            return out

    def load_checkpoint(checkpoint_path, model, map_location="cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(checkpoint["model"])
        print(f"[Test] Checkpoint loaded from {checkpoint_path} (epoch={checkpoint['epoch']})")

    def test_model(model_path, 
                   test_gray_dir, 
                   test_mask_dir, 
                   output_dir,
                   device="cuda" if torch.cuda.is_available() else "cpu",
                   img_ext=".png"):
        """
        폴더 단위로 이미지 변환 & 출력 저장
        """
        # 1) 모델 생성 & 체크포인트 로드
        model = ResNetUNet(out_ch=2, pretrained=False).to(device)
        load_checkpoint(model_path, model, map_location=device)
        model.eval()

        transform_gray = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()
        ])

        os.makedirs(output_dir, exist_ok=True)

        gray_image_paths = sorted(glob.glob(os.path.join(test_gray_dir, f"*{img_ext}")))

        with torch.no_grad():
            for gray_path in gray_image_paths:
                fname = os.path.basename(gray_path)
                mask_path = os.path.join(test_mask_dir, fname)

                if not os.path.exists(mask_path):
                    print(f"No matching mask found for {fname}, skipping...")
                    continue

                # (1) 흑백 이미지 로드
                gray_img = Image.open(gray_path).convert('L')
                gray_tensor = transform_gray(gray_img)  # [1,H,W]

                # (2) 마스크 로드
                mask_img = Image.open(mask_path).convert('L')
                mask_np = np.array(mask_img)
                mask_bin = (mask_np > 128).astype(np.float32)
                mask_bin = torch.from_numpy(mask_bin).unsqueeze(0)  # [1,H,W]

                # (3) 모델 추론
                gray_tensor = gray_tensor.unsqueeze(0).to(device)  # [1,1,H,W]
                pred_ab = model(gray_tensor)                      # [1,2,H,W]

                # (4) Lab -> RGB 변환
                pred_ab_np = pred_ab[0].cpu().permute(1,2,0).numpy()  # [H,W,2]
                L_np = gray_tensor[0,0].cpu().numpy()
                pred_rgb = lab_to_rgb(L_np, pred_ab_np[:,:,0], pred_ab_np[:,:,1])

                # (5) 결과 저장
                out_path = os.path.join(output_dir, fname)
                out_img = (pred_rgb*255).astype(np.uint8)
                Image.fromarray(out_img).save(out_path)
                print(f"Saved: {out_path}")

    # -----------------------
    # Streamlit 데모용 메인 함수
    # -----------------------
    
    st.title("흑백->컬러 변환 모델 데모")

    # --- 입력 파라미터 ---
    st.subheader("모델 & 폴더 경로 설정")
    model_ckpt = st.text_input("모델 체크포인트 경로", value="model/color.pth")
    test_gray_dir = st.text_input("테스트용 흑백 이미지 폴더", value="imgpro/sampleData/test_input")
    test_mask_dir = st.text_input("테스트용 마스크 폴더", value="imgpro/sampleData/output_01_mask")
    output_dir = st.text_input("결과 저장 폴더", value="imgpro/sampleData/tmp")
    run_inference = st.button("테스트 실행")

    if run_inference:
        with st.spinner("모델 추론 중... 잠시만 기다려주세요."):
            test_model(
                model_path=model_ckpt,
                test_gray_dir=test_gray_dir,
                test_mask_dir=test_mask_dir,
                output_dir=output_dir
            )
        st.success("테스트 완료!")

        # 결과 확인하기
        st.subheader("결과 이미지 미리보기")
        result_image_paths = sorted(glob.glob(os.path.join(output_dir, "*")))
        if len(result_image_paths) == 0:
            st.write("결과 이미지가 없습니다.")
        else:
            # 결과 이미지와 대응되는 흑백 이미지도 함께 보여주기
            for path in result_image_paths:
                file_name = os.path.basename(path)
                # 복원 전(흑백) 이미지 경로
                original_gray_path = os.path.join(test_gray_dir, file_name)
                
                st.write(f"**파일명**: {file_name}")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(Image.open(original_gray_path), caption="복원 전 (흑백)", use_column_width=True)
                with col2:
                    st.image(Image.open(path), caption="복원 후 (컬러)", use_column_width=True)
                st.markdown("---")
