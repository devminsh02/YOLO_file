import sys
import pathlib
import argparse
import os
import time
from pathlib import Path
import torch
import cv2
import numpy as np
from sklearn.cluster import KMeans

# 윈도우 시스템에서 PosixPath를 WindowsPath로 매핑
if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath

# YOLOv5 관련 모듈 임포트
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 루트 디렉토리
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # ROOT을 PATH에 추가
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 상대 경로로 변경

from utils.plots import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
    cv2,
)
from utils.torch_utils import select_device, smart_inference_mode

# ---------------------- ColorThief 관련 함수 ----------------------

def extract_dominant_color(image, k=3):
    """
    주어진 이미지에서 주요 색상을 추출합니다.
    Args:
        image (numpy.ndarray): BGR 이미지
        k (int): 클러스터 수
    Returns:
        dominant_color (tuple): RGB 형식의 주요 색상
        dominant_percentage (float): 주요 색상의 비율
    """
    # 이미지 확대
    scale_percent = 200  # 200% 확대
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

    # 노이즈 제거
    blur = cv2.GaussianBlur(resized_img, (5, 5), 0)

    # 색상 공간 변환
    img_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

    # K-평균 클러스터링 적용
    pixel_data = img_rgb.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixel_data)

    # 주요 색상 추출
    colors_centers = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    percentages = counts / counts.sum()

    # 가장 비율이 높은 색상 선택
    dominant_color = colors_centers[np.argmax(percentages)]
    dominant_percentage = np.max(percentages)

    return tuple(dominant_color), dominant_percentage

def create_color_image(color, size=(100, 100)):
    """
    지정된 색상으로 단색 이미지를 생성합니다.
    Args:
        color (tuple): RGB 형식의 색상
        size (tuple): 이미지 크기 (width, height)
    Returns:
        color_img (numpy.ndarray): 단색 이미지
    """
    color_img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    color_img[:] = color  # 이미지를 지정된 색상으로 채우기
    return color_img

def process_detected_objects(save_dir, im0, detections, names, line_thickness=3):
    """
    감지된 객체들에 대해 색상을 추출하고 저장합니다.
    Args:
        save_dir (Path): 저장할 디렉토리 경로
        im0 (numpy.ndarray): 원본 이미지
        detections (Tensor): 감지된 객체들
        names (list): 클래스 이름 리스트
        line_thickness (int): 바운딩 박스 두께
    """
    # 각 클래스별 최고 신뢰도 및 해당 크롭된 이미지 저장을 위한 딕셔너리 초기화
    max_conf_per_class = {}  # {class_index: (confidence, cropped_image)}

    for *xyxy, conf, cls in reversed(detections):
        c = int(cls)  # 클래스 인덱스
        confidence = float(conf)

        # 크롭 박스의 크기를 줄이기 위한 스케일 팩터 (예: 20% 줄임)
        scale_factor = 0.8  # 원하는 만큼 줄일 수 있음 (0.8은 20% 줄이는 의미)

        # xyxy 좌표를 조정하여 크기를 줄임
        x1, y1, x2, y2 = xyxy  # 기존 좌표

        # 박스의 중심점 계산
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # 너비와 높이 계산
        width = (x2 - x1) * scale_factor
        height = (y2 - y1) * scale_factor

        # 새로운 좌표 계산 (이미지 경계를 벗어나지 않도록 조정)
        new_x1 = max(int(cx - width / 2), 0)
        new_y1 = max(int(cy - height / 2), 0)
        new_x2 = min(int(cx + width / 2), im0.shape[1])
        new_y2 = min(int(cy + height / 2), im0.shape[0])

        # 크롭된 이미지 얻기
        crop_img = im0[new_y1:new_y2, new_x1:new_x2].copy()

        # 클래스별 최대 신뢰도 업데이트
        if (c not in max_conf_per_class) or (confidence > max_conf_per_class[c][0]):
            max_conf_per_class[c] = (confidence, crop_img)

    # 각 클래스별 최고 신뢰도의 크롭된 이미지 처리
    for c, (conf, crop_img) in max_conf_per_class.items():
        class_name = names[c]
        final_save_path = save_dir / f'best_{class_name}.jpg'
        cv2.imwrite(str(final_save_path), crop_img)
        print(f"클래스 '{class_name}'의 최고 신뢰도 크롭 이미지를 저장했습니다: {final_save_path}")

def extract_and_save_colors(save_dir):
    """
    저장된 이미지들에서 주요 색상을 추출하고 저장합니다.
    Args:
        save_dir (Path): 이미지가 저장된 디렉토리 경로
    """
    # 저장된 이미지들 순회
    for filename in os.listdir(save_dir):
        file_path = os.path.join(save_dir, filename)

        # 이미지 파일만 처리
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"{filename} 처리 중...")
            img = cv2.imread(file_path)
            if img is None:
                print(f"이미지를 불러올 수 없습니다: {file_path}")
                continue

            dominant_color, dominant_percentage = extract_dominant_color(img)

            if dominant_color is None:
                continue  # 이미지 로드 실패 시 건너뜀

            # 추출된 주요 색상을 파일로 저장 (파일명 + "_color.txt")
            color_filename = os.path.splitext(filename)[0] + '_color.txt'
            color_file_path = os.path.join(save_dir, color_filename)

            with open(color_file_path, 'w') as f:
                f.write(f'주요 색상 (RGB): {dominant_color}\n')
                f.write(f'비율: {dominant_percentage * 100:.2f}%\n')

            # 주요 색상 이미지를 저장 (파일명 + "_color.jpg")
            color_image = create_color_image(dominant_color)
            color_image_filename = os.path.splitext(filename)[0] + '_color.jpg'
            color_image_path = os.path.join(save_dir, color_image_filename)
            cv2.imwrite(color_image_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

            print(f'{filename} 처리 완료: 주요 색상이 {color_file_path}와 {color_image_path}에 저장되었습니다.')

# ---------------------- YOLOv5 Detection 함수 ----------------------

@smart_inference_mode()
def run(
    weights=ROOT / "best.pt",  # 모델 경로 또는 Triton URL
    source="0",  # 파일/디렉토리/URL/글로벌 패턴/스크린/웹캠
    data=ROOT / "data/coco128.yaml",  # 데이터셋 YAML 경로
    imgsz=(416, 416),  # 추론 이미지 크기 (높이, 너비)
    conf_thres=0.5,  # 신뢰도 임계값
    iou_thres=0.45,  # NMS IoU 임계값
    max_det=1000,  # 이미지당 최대 검출 수
    device="",  # CUDA 디바이스 또는 CPU
    view_img=True,  # 결과를 화면에 표시할지 여부
    save_txt=False,  # 결과를 텍스트 파일로 저장할지 여부
    save_format=0,  # 박스 좌표를 YOLO 또는 Pascal-VOC 형식으로 저장
    save_csv=False,  # 결과를 CSV로 저장할지 여부
    save_conf=False,  # 신뢰도 점수를 저장할지 여부
    save_crop=False,  # 검출된 객체를 크롭하여 저장할지 여부
    nosave=False,  # 이미지/비디오를 저장하지 않을지 여부
    classes=None,  # 특정 클래스만 필터링할지 여부
    agnostic_nms=False,  # 클래스에 무관한 NMS 적용 여부
    augment=False,  # 추론 시 데이터 증강 적용 여부
    visualize=False,  # 특징 맵 시각화 여부
    update=False,  # 모든 모델 업데이트 여부
    exist_ok=False,  # 기존 폴더에 덮어쓸지 여부
    line_thickness=3,  # 바운딩 박스 두께
    hide_labels=False,  # 레이블 숨길지 여부
    hide_conf=False,  # 신뢰도 점수 숨길지 여부
    half=False,  # FP16 반정밀도 추론 사용 여부
    dnn=False,  # ONNX 추론 시 OpenCV DNN 사용 여부
    vid_stride=1,  # 비디오 프레임 간격
    **kwargs
):
    # opt에서 전달된 project와 name을 사용
    project = kwargs.get('project', Path(r"C:\Users\minsh\OneDrive\Desktop\resultOfPicture"))
    name = kwargs.get('name', "result")

    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # 추론 이미지 저장 여부
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # 다운로드

    # 디렉토리 설정
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 저장 경로 설정
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 디렉토리 생성

    # 모델 로드
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # 이미지 크기 확인

    # 데이터 로더 설정
    bs = 1  # 배치 크기
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # 추론 실행
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # 워밍업
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # 시간 측정을 위한 변수 설정
    start_time = time.time()
    duration = 5  # 캡처 지속 시간 (초)

    for path, im, im0s, vid_cap, s in dataset:
        current_time = time.time()
        if current_time - start_time > duration:
            LOGGER.info("5초가 경과하여 캡처를 종료합니다.")
            break  # 5초가 지나면 루프를 종료

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8에서 fp16/32로 변환
            im /= 255  # 0~255를 0.0~1.0으로 스케일링
            if len(im.shape) == 3:
                im = im[None]  # 배치 차원 추가

        # 추론
        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)

        # NMS 적용
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 예측 결과 처리
        for i, det in enumerate(pred):  # 이미지별로 처리
            seen += 1
            if webcam:  # 배치 크기가 1 이상인 경우
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # 경로 객체로 변환
            import datetime
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')  # 중복 방지를 위해 마이크로초 추가
            s += "{:g}x{:g} ".format(*im.shape[2:])  # 출력 문자열
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 정규화 스케일
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # 바운딩 박스를 원본 이미지 크기에 맞게 조정
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 결과 출력
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # 클래스별 검출 수
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 출력 문자열에 추가

                # 결과 저장 및 ColorThief 적용
                process_detected_objects(save_dir, im0, det, names, line_thickness)

            # 추론 시간 출력
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # 감지된 객체에 대한 ColorThief 처리가 모두 끝난 후, 전체 이미지에 대한 색 추출 (선택 사항)
    # extract_and_save_colors(save_dir)

    # 결과 출력
    t = tuple(x.t / seen * 1e3 for x in dt if seen)  # 이미지당 속도
    if t:
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # 모델 업데이트 (SourceChangeWarning 수정)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "best.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default="0", help="file/dir/URL/glob/screen/0(webcam)")  # 기본값을 웹캠으로 설정
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[416], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    # 카메라를 on시키는 코드
    parser.add_argument("--view-img", action="store_true", default=True, help="show results")
    # 카메라를 off 시키는 코드
    # parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    )
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    # 여기서 project와 name의 기본값을 원하는 경로로 설정합니다.
    parser.add_argument("--project", default=Path(r"C:\Users\minsh\OneDrive\Desktop\resultOfPicture"), help="save results to project/name")
    parser.add_argument("--name", default="result", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    # 이미지 크기가 하나의 값이면 (예: 416), 이를 (416, 416)으로 변환
    if len(opt.imgsz) == 1:
        opt.imgsz *= 2
    print_args(vars(opt))
    return opt

def main(opt):
    # 필요한 요구 사항을 확인합니다.
    check_requirements(exclude=("tensorboard", "thop"))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

    # ---------------------- ColorThief 관련 코드 ----------------------
    # YOLO 감지가 완료된 후, 저장된 이미지들에 대해 ColorThief 기능을 적용합니다.

    # 저장된 이미지들이 저장된 디렉토리 경로
    save_dir = Path(opt.project) / opt.name

    def extract_dominant_color_after_detection(save_dir, k=3):
        """
        YOLO 감지가 완료된 후, 저장된 이미지들에서 주요 색상을 추출하고 저장합니다.
        Args:
            save_dir (Path): 이미지가 저장된 디렉토리 경로
            k (int): 클러스터 수
        """
        # 저장된 이미지들 순회
        for filename in os.listdir(save_dir):
            file_path = os.path.join(save_dir, filename)

            # 이미지 파일만 처리
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and not filename.endswith('_color.jpg'):
                print(f"{filename} 처리 중...")
                img = cv2.imread(file_path)
                if img is None:
                    print(f"이미지를 불러올 수 없습니다: {file_path}")
                    continue

                dominant_color, dominant_percentage = extract_dominant_color(img, k=k)

                if dominant_color is None:
                    continue  # 이미지 로드 실패 시 건너뜀

                # 추출된 주요 색상을 파일로 저장 (파일명 + "_color.txt")
                color_filename = os.path.splitext(filename)[0] + '_color.txt'
                color_file_path = os.path.join(save_dir, color_filename)

                with open(color_file_path, 'w') as f:
                    f.write(f'주요 색상 (RGB): {dominant_color}\n')
                    f.write(f'비율: {dominant_percentage * 100:.2f}%\n')

                # 주요 색상 이미지를 저장 (파일명 + "_color.jpg")
                color_image = create_color_image(dominant_color)
                color_image_filename = os.path.splitext(filename)[0] + '_color.jpg'
                color_image_path = os.path.join(save_dir, color_image_filename)
                cv2.imwrite(color_image_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

                print(f'{filename} 처리 완료: 주요 색상이 {color_file_path}와 {color_image_path}에 저장되었습니다.')

    # ColorThief 기능 실행
    extract_dominant_color_after_detection(save_dir)
