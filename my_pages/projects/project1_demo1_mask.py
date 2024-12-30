import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import os
import tempfile
import random

# -----------------------------
# (1) 사용자 코드: mask_function
# -----------------------------
def mask_function(input_path, gt_path):
    """
    input_path: 손상 이미지 경로
    gt_path   : 정답(손상 전) 이미지 경로
    """
    try:
        # Load and preprocess images
        input_image = Image.open(input_path).convert("RGB")
        input_image_np = np.array(input_image)
        gt_image_gray = Image.open(gt_path).convert("L")  # GT를 흑백으로
        gt_image_gray_np = np.array(gt_image_gray)

        # Convert input_image_np to grayscale
        input_image_gray_np = cv2.cvtColor(input_image_np, cv2.COLOR_RGB2GRAY)

        # Compute the difference
        difference = cv2.absdiff(gt_image_gray_np, input_image_gray_np)

        # Threshold the difference to create a binary mask
        _, binary_difference = cv2.threshold(difference, 1, 255, cv2.THRESH_BINARY)

        # Remove small noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        binary_difference = cv2.morphologyEx(binary_difference, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary_difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Fill the contours to create a mask
        mask_filled = np.zeros_like(binary_difference)
        cv2.drawContours(mask_filled, contours, -1, color=255, thickness=cv2.FILLED)

        # Expand the filled mask (dilation)
        mask_filled = cv2.dilate(mask_filled, kernel, iterations=1)

        # Convert mask to torch tensor (0~1 범위)
        mask_tensor = torch.tensor(mask_filled, dtype=torch.float32).unsqueeze(0) / 255.0

        return mask_tensor
    except Exception as e:
        print(f"Mask creation failed: {e}")
        # 오류 시 빈 마스크(0) 반환
        return torch.zeros((1, 512, 512), dtype=torch.float32)

# -----------------------------
# (2) 손상 함수: 랜덤으로 흑백화 & 검은 사각형 삽입
# -----------------------------
def random_damage_image(image: Image.Image) -> Image.Image:
    """
    원본 이미지를 PIL.Image로 받아
    일부 영역을 임의로 흑백 처리하거나 검은 박스를 넣어 '손상'시키고
    그 결과를 리턴하는 예시 함수
    """
    img_np = np.array(image)

    # 1) 전체 or 일부분을 흑백으로 바꾸기
    h, w, c = img_np.shape
    # 임의의 사각형 범위
    x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
    x2, y2 = random.randint(w//2, w), random.randint(h//2, h)

    # ROI를 추출해서 흑백 변환
    roi = img_np[y1:y2, x1:x2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    roi_gray_3ch = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
    img_np[y1:y2, x1:x2] = roi_gray_3ch

    # 2) 검은 사각형(블록)도 일부 삽입
    for _ in range(2):  # 2회 정도 삽입
        bx1, by1 = random.randint(0, w-50), random.randint(0, h-50)
        bw, bh = random.randint(20, 80), random.randint(20, 80)
        cv2.rectangle(img_np, (bx1, by1), (bx1+bw, by1+bh), (0, 0, 0), -1)

    return Image.fromarray(img_np)

# -----------------------------
# (3) Streamlit 메인 앱
# -----------------------------
def main():
    st.title("이미지 손상 & 마스크 생성 데모")

    uploaded_file = st.file_uploader("컬러 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # 원본 이미지 불러오기
        original_image = Image.open(uploaded_file).convert("RGB")

        # 손상 이미지 만들기
        damaged_image = random_damage_image(original_image)

        # (임시) original_image를 흑백으로 만들 'gt'를 정의할 수도 있고,
        # 여기서는 "손상 전"이라고 치고 그대로 사용해도 됨
        # 실제론 원본 컬러를 GT로 삼을 수도, 별도의 흑백 변환을 GT로 삼을 수도 있음
        # 편의상 여기선 "original_image"를 gt로 간주

        # 임시 파일에 저장해 mask_function에 넘기기
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_input:
            damaged_image.save(tmp_input.name)
            input_path = tmp_input.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_gt:
            # 여기서는 original_image가 "손상 전"이라고 가정
            original_image_gray = original_image.convert("L")
            original_image_gray.save(tmp_gt.name)
            gt_path = tmp_gt.name

        # mask_function 호출
        mask_tensor = mask_function(input_path, gt_path)

        # mask_tensor → numpy → PIL 변환
        mask_np = (mask_tensor.squeeze(0).numpy() * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_np)

        # 화면에 시각화
        st.subheader("원본(컬러) 이미지")
        st.image(original_image, use_column_width=True)

        st.subheader("손상된 이미지")
        st.image(damaged_image, use_column_width=True)

        st.subheader("예측 마스크(손상 영역)")
        st.image(mask_image, use_column_width=True)

        # 임시 파일 삭제(윈도우 환경에서 바로 삭제가 안 될 수 있으므로 필요시 예외처리)
        try:
            os.remove(input_path)
            os.remove(gt_path)
        except:
            pass

    else:
        st.info("이미지를 업로드하면 결과를 볼 수 있습니다.")

if __name__ == "__main__":
    main()
