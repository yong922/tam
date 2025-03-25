# run_tracking.py
import os
import cv2
import numpy as np
import PIL.Image
from track_anything import TrackingAnything, parse_augment


def extract_frames(video_path, track_every_n_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % track_every_n_frames == 0:  # 5프레임마다 하나씩 추출
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames

def resize_frames(frames, target_size=(640, 360)):
    # 프레임 크기 축소
    return [cv2.resize(frame, target_size) for frame in frames]

def save_video(frames, save_path, fps=30):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()

if __name__ == "__main__":
    # 설정
    video_path = "../data/singapore_front_reencoded.mp4"  
    save_path = "results/tracking_result.mp4"
    os.makedirs("results", exist_ok=True)

    # 1. 프레임 추출 (5프레임마다)
    frames = extract_frames(video_path, track_every_n_frames=5)

    # 2. 프레임 크기 축소 (해상도 줄이기)
    # frames = resize_frames(frames)

    # 3. 클릭 좌표 지정 (영상 첫 프레임 기준)
    h, w = frames[0].shape[:2]
    input_point = np.array([[w // 2, h // 2]])  # 영상 정중앙 클릭
    input_label = np.array([1])  # foreground

    # 4. 추적기 세팅 (CPU로 설정)
    args = parse_augment()
    args.device = "cpu"
    trackany = TrackingAnything(
        sam_checkpoint="checkpoints/sam_vit_h_4b8939.pth",
        xmem_checkpoint="checkpoints/XMem-s012.pth",
        e2fgvi_checkpoint=None,  # inpainting 안 쓸 거면 None
        args=args
    )

    # 5. 첫 프레임 마스크 생성 (SAM)
    mask, logit, painted = trackany.first_frame_click(frames[0], input_point, input_label)

    # 6. 전체 프레임 추적 (XMem)
    masks, logits, painted_images = trackany.generator(frames, mask)

    # 7. 마스크 시각화 (간단한 오버레이)
    overlay_frames = []
    for i in range(len(frames)):
        frame = frames[i].copy()
        mask = masks[i]
        frame[mask > 0] = [0, 255, 0]  # 마스크 부분을 초록색으로
        overlay_frames.append(frame)

    # 8. 결과 저장
    save_video(overlay_frames, save_path)
    print(f"✅ 추적 결과 저장 완료: {save_path}")