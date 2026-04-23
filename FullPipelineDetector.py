import os
import shutil
import cv2
import math
import json
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import onnxruntime as ort


def runPipeline(videoPath):
    VIDEO_PATH = videoPath
    HELMET_MODEL_PATH = "C:\Coding Projects\FootballConcussionDetector\detModel1.pt"
    IMPACT_MODEL_PATH = "C:\Coding Projects\FootballConcussionDetector\clsVMode7.pt"

    OUTPUT_DIR = "Temp"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_ALL_DETECTIONS_CSV = os.path.join(OUTPUT_DIR, "all_scored_detections.csv")
    OUTPUT_FINAL_IMPACTS_CSV = os.path.join(OUTPUT_DIR, "final_impacts.csv")

    I3D_ONNX_PATH = os.path.join(OUTPUT_DIR, "i3d_binary.onnx")
    ONNX_OPSET = 13
    ORT_INTRA_OP_THREADS = max(1, os.cpu_count() or 1)
    ORT_INTER_OP_THREADS = 1

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE_HELMET = 8
    BATCH_SIZE_I3D = 4
    YOLO_IMGSZ = 1280
    YOLO_CONF = 0.25
    YOLO_IOU = 0.50

    N_FRAMES = 9
    STRIDE = 2
    CROP_SIZE = 64
    BOX_EXPANSION_RATIO = 0.22

    DET_THRESHOLD = 0.42
    CLS_THRESHOLD = 0.48
    SWITCH_FRAME = 150
    DET_THRESHOLD2 = 0.40
    CLS_THRESHOLD2 = 0.65
    DELTA_CLS = -0.07
    DELTA_DET = -0.05
    ADJ_IOU_THRESHOLD = 0.41
    ADJ_MAX_FRAME_DIST = 9
    ADJ_MIN_CLUSTER_SIZE = 0
    ADJ_N_TIMES = 1

    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = True

    class MaxPool3dSamePadding(nn.MaxPool3d):
        def compute_pad(self, dim, s):
            if s % self.stride[dim] == 0:
                return max(self.kernel_size[dim] - self.stride[dim], 0)
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

        def forward(self, x):
            (_, _, t, h, w) = x.size()
            pad_t = self.compute_pad(0, t)
            pad_h = self.compute_pad(1, h)
            pad_w = self.compute_pad(2, w)
            pad = (
                pad_w // 2, pad_w - pad_w // 2,
                pad_h // 2, pad_h - pad_h // 2,
                pad_t // 2, pad_t - pad_t // 2,
            )
            x = F.pad(x, pad)
            return super().forward(x)

    class Unit3D(nn.Module):
        def __init__(
            self,
            in_channels,
            output_channels,
            kernel_shape=(1, 1, 1),
            stride=(1, 1, 1),
            activation_fn=F.relu,
            use_batch_norm=True,
            use_bias=False,
        ):
            super().__init__()
            self._kernel_shape = kernel_shape
            self._stride = stride
            self._use_batch_norm = use_batch_norm
            self._activation_fn = activation_fn
            self.conv3d = nn.Conv3d(
                in_channels=in_channels,
                out_channels=output_channels,
                kernel_size=kernel_shape,
                stride=stride,
                padding=0,
                bias=use_bias,
            )
            if use_batch_norm:
                self.bn = nn.BatchNorm3d(output_channels, eps=0.001, momentum=0.01)

        def compute_pad(self, dim, s):
            if s % self._stride[dim] == 0:
                return max(self._kernel_shape[dim] - self._stride[dim], 0)
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

        def forward(self, x):
            (_, _, t, h, w) = x.size()
            pad_t = self.compute_pad(0, t)
            pad_h = self.compute_pad(1, h)
            pad_w = self.compute_pad(2, w)
            pad = (
                pad_w // 2, pad_w - pad_w // 2,
                pad_h // 2, pad_h - pad_h // 2,
                pad_t // 2, pad_t - pad_t // 2,
            )
            x = F.pad(x, pad)
            x = self.conv3d(x)
            if self._use_batch_norm:
                x = self.bn(x)
            if self._activation_fn is not None:
                x = self._activation_fn(x)
            return x

    class InceptionModule(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.b0 = Unit3D(in_channels, out_channels[0], kernel_shape=[1, 1, 1])
            self.b1a = Unit3D(in_channels, out_channels[1], kernel_shape=[1, 1, 1])
            self.b1b = Unit3D(out_channels[1], out_channels[2], kernel_shape=[3, 3, 3])
            self.b2a = Unit3D(in_channels, out_channels[3], kernel_shape=[1, 1, 1])
            self.b2b = Unit3D(out_channels[3], out_channels[4], kernel_shape=[3, 3, 3])
            self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1))
            self.b3b = Unit3D(in_channels, out_channels[5], kernel_shape=[1, 1, 1])

        def forward(self, x):
            b0 = self.b0(x)
            b1 = self.b1b(self.b1a(x))
            b2 = self.b2b(self.b2a(x))
            b3 = self.b3b(self.b3a(x))
            return torch.cat([b0, b1, b2, b3], dim=1)

    class InceptionI3d(nn.Module):
        def __init__(self, num_classes=400, in_channels=3):
            super().__init__()
            self.Conv3d_1a_7x7 = Unit3D(in_channels, 64, kernel_shape=[7, 7, 7], stride=(2, 2, 2))
            self.MaxPool3d_2a_3x3 = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2))
            self.Conv3d_2b_1x1 = Unit3D(64, 64, kernel_shape=[1, 1, 1])
            self.Conv3d_2c_3x3 = Unit3D(64, 192, kernel_shape=[3, 3, 3])
            self.MaxPool3d_3a_3x3 = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2))
            self.Mixed_3b = InceptionModule(192, [64, 96, 128, 16, 32, 32])
            self.Mixed_3c = InceptionModule(256, [128, 128, 192, 32, 96, 64])
            self.MaxPool3d_4a_3x3 = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2))
            self.Mixed_4b = InceptionModule(480, [192, 96, 208, 16, 48, 64])
            self.Mixed_4c = InceptionModule(512, [160, 112, 224, 24, 64, 64])
            self.Mixed_4d = InceptionModule(512, [128, 128, 256, 24, 64, 64])
            self.Mixed_4e = InceptionModule(512, [112, 144, 288, 32, 64, 64])
            self.Mixed_4f = InceptionModule(528, [256, 160, 320, 32, 128, 128])
            self.MaxPool3d_5a_2x2 = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2))
            self.Mixed_5b = InceptionModule(832, [256, 160, 320, 32, 128, 128])
            self.Mixed_5c = InceptionModule(832, [384, 192, 384, 48, 128, 128])
            self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
            self.dropout = nn.Dropout(0.5)
            self.logits = Unit3D(1024, num_classes, kernel_shape=[1, 1, 1], activation_fn=None, use_batch_norm=False, use_bias=True)

        def extract_features(self, x):
            x = self.Conv3d_1a_7x7(x)
            x = self.MaxPool3d_2a_3x3(x)
            x = self.Conv3d_2b_1x1(x)
            x = self.Conv3d_2c_3x3(x)
            x = self.MaxPool3d_3a_3x3(x)
            x = self.Mixed_3b(x)
            x = self.Mixed_3c(x)
            x = self.MaxPool3d_4a_3x3(x)
            x = self.Mixed_4b(x)
            x = self.Mixed_4c(x)
            x = self.Mixed_4d(x)
            x = self.Mixed_4e(x)
            x = self.Mixed_4f(x)
            x = self.MaxPool3d_5a_2x2(x)
            x = self.Mixed_5b(x)
            x = self.Mixed_5c(x)
            return x

        def forward(self, x):
            x = self.extract_features(x)
            x = self.avg_pool(x)
            x = self.dropout(x)
            x = self.logits(x)
            x = x.squeeze(3).squeeze(3)
            x = torch.mean(x, 2)
            return x, None

    def build_repo_i3d_binary():
        model = InceptionI3d(num_classes=400, in_channels=3)
        model.Conv3d_1a_7x7.conv3d.stride = (1, 1, 1)
        model.MaxPool3d_4a_3x3.stride = (1, 2, 2)
        model.MaxPool3d_5a_2x2.stride = (1, 2, 2)
        nb_ft = model.logits.conv3d.in_channels
        model.logits = nn.Linear(nb_ft, 1)

        def forward_binary(x):
            x = model.extract_features(x)
            x = F.adaptive_avg_pool3d(x, (1, 1, 1))
            x = x.flatten(1)
            x = model.dropout(x)
            x = model.logits(x)
            return x, None

        model.forward = forward_binary
        return model

    def load_impact_model(weight_path, device):
        model = build_repo_i3d_binary().to(device)
        ckpt = torch.load(weight_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            raise ValueError("Unsupported checkpoint format for IMPACT_MODEL_PATH")
        cleaned = {}
        for k, v in state.items():
            nk = k
            for prefix in ["module.", "model."]:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            cleaned[nk] = v
        model.load_state_dict(cleaned, strict=False)
        model.eval()
        return model

    class I3DOnnxWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            logits, _ = self.model(x)
            return logits

    def export_i3d_to_onnx(weight_path, onnx_path, batch_size=1):
        model = load_impact_model(weight_path, device="cpu")
        wrapper = I3DOnnxWrapper(model).eval()
        dummy = torch.randn(batch_size, 3, N_FRAMES, CROP_SIZE, CROP_SIZE, dtype=torch.float32)
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        torch.onnx.export(
            wrapper,
            dummy,
            onnx_path,
            input_names=["clips"],
            output_names=["logits"],
            dynamic_axes={"clips": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=ONNX_OPSET,
            do_constant_folding=True,
        )
        return onnx_path

    def get_onnx_session(onnx_path):
        if not os.path.exists(onnx_path):
            export_i3d_to_onnx(IMPACT_MODEL_PATH, onnx_path, batch_size=1)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = ORT_INTRA_OP_THREADS
        so.inter_op_num_threads = ORT_INTER_OP_THREADS
        return ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])

    def get_adjacent_frames(frame, max_frame, n_frames=9, stride=2):
        frames = np.arange(n_frames) * stride
        frames = frames - frames[n_frames // 2] + frame
        if frames.min() < 1:
            frames -= frames.min() - 1
        elif frames.max() > max_frame:
            frames += max_frame - frames.max()
        return frames.astype(int)

    def extend_box(box_xyxy, size=64):
        x1, y1, x2, y2 = box_xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        half = size / 2.0
        return np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)

    def adapt_box_to_shape(box_xyxy, height, width):
        x1, y1, x2, y2 = box_xyxy.copy()
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > width:
            shift = x2 - width
            x1 -= shift
            x2 = width
        if y2 > height:
            shift = y2 - height
            y1 -= shift
            y2 = height
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        if (x2 - x1) <= 1 or (y2 - y1) <= 1:
            cx = np.clip((x1 + x2) / 2.0, 0, width)
            cy = np.clip((y1 + y2) / 2.0, 0, height)
            half = min(width, height, CROP_SIZE) / 2.0
            x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
        return np.array([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))], dtype=np.int32)

    def iou_xyxy(a, b):
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])
        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - inter
        return 0.0 if union <= 0 else inter / union

    def expand_boxes_inplace(df, r=0.0, frame_w=1280, frame_h=720):
        df = df.copy()
        if r <= 0 or len(df) == 0:
            return df
        df["left"] = df["left"] - df["width"] * r / 2
        df["top"] = df["top"] - df["height"] * r / 2
        df["width"] = df["width"] * (1 + r)
        df["height"] = df["height"] * (1 + r)
        df["left"] = np.clip(df["left"], 0, None)
        df["top"] = np.clip(df["top"], 0, None)
        df["width"] = np.clip(df["width"], 0, frame_w - df["left"])
        df["height"] = np.clip(df["height"], 0, frame_h - df["top"])
        right = np.round(df["left"] + df["width"], 0)
        bottom = np.round(df["top"] + df["height"], 0)
        df["left"] = np.round(df["left"], 0).astype(int)
        df["top"] = np.round(df["top"], 0).astype(int)
        df["width"] = (right - df["left"]).astype(int)
        df["height"] = (bottom - df["top"]).astype(int)
        return df

    def sideline_keep(row):
        det_thr = DET_THRESHOLD if row.frame <= SWITCH_FRAME else DET_THRESHOLD2
        cls_thr = CLS_THRESHOLD if row.frame <= SWITCH_FRAME else CLS_THRESHOLD2
        det_thr = det_thr - DELTA_DET
        cls_thr = cls_thr - DELTA_CLS
        return (row.det_score > det_thr) and (row.pred_cls > cls_thr)

    def adjacency_postprocess(df, iou_threshold=0.41, max_dist=9, min_cluster_size=0, n_times=1):
        if len(df) == 0:
            return df.copy()
        out = df.copy().sort_values(["video", "frame"]).reset_index(drop=True)
        for _ in range(n_times):
            kept_rows = []
            for video, g in out.groupby("video", sort=False):
                g = g.sort_values("frame").reset_index(drop=True)
                clusters = [[0]] if len(g) else []
                for i in range(1, len(g)):
                    assigned = False
                    for cl in reversed(clusters):
                        j = cl[-1]
                        if abs(int(g.loc[i, "frame"]) - int(g.loc[j, "frame"])) > max_dist:
                            continue
                        box_i = [g.loc[i, "left"], g.loc[i, "top"], g.loc[i, "left"] + g.loc[i, "width"], g.loc[i, "top"] + g.loc[i, "height"]]
                        box_j = [g.loc[j, "left"], g.loc[j, "top"], g.loc[j, "left"] + g.loc[j, "width"], g.loc[j, "top"] + g.loc[j, "height"]]
                        if iou_xyxy(box_i, box_j) > iou_threshold:
                            cl.append(i)
                            assigned = True
                            break
                    if not assigned:
                        clusters.append([i])
                centroids = []
                for cl in clusters:
                    if len(cl) < min_cluster_size:
                        continue
                    centroids.append(cl[len(cl) // 2])
                if len(centroids):
                    kept_rows.append(g.iloc[centroids])
            out = pd.concat(kept_rows, axis=0).reset_index(drop=True) if kept_rows else out.iloc[0:0].copy()
        return out

    class HelmetClipDataset(Dataset):
        def __init__(self, detections_df, frames_bgr, adj_frame_cache):
            self.frames = frames_bgr
            self.center_frames = detections_df["frame"].to_numpy(np.int32)
            self.crop_boxes = np.stack(detections_df["crop_box"].to_list()).astype(np.int32)
            self.adj_frame_cache = adj_frame_cache

        def __len__(self):
            return len(self.center_frames)

        def __getitem__(self, idx):
            center_frame = int(self.center_frames[idx])
            adj_frames = self.adj_frame_cache[center_frame]
            x1, y1, x2, y2 = self.crop_boxes[idx]
            clip = []
            for f in adj_frames:
                img = self.frames[f - 1]
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    crop = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
                elif crop.shape[0] != CROP_SIZE or crop.shape[1] != CROP_SIZE:
                    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
                clip.append(crop[:, :, ::-1])
            clip = np.stack(clip, axis=0).astype(np.float32)
            clip = clip / 255.0
            clip = (clip - 0.5) / 0.5
            clip = clip.transpose(3, 0, 1, 2)
            return torch.from_numpy(clip), idx

    assert os.path.exists(VIDEO_PATH), f"VIDEO_PATH not found: {VIDEO_PATH}"
    assert os.path.exists(HELMET_MODEL_PATH), f"HELMET_MODEL_PATH not found: {HELMET_MODEL_PATH}"
    assert os.path.exists(IMPACT_MODEL_PATH), f"IMPACT_MODEL_PATH not found: {IMPACT_MODEL_PATH}"

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_bgr = []
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
    cap.release()
    frame_count = len(frames_bgr)

    adj_frame_cache = {
        i: get_adjacent_frames(i, frame_count, n_frames=N_FRAMES, stride=STRIDE)
        for i in range(1, frame_count + 1)
    }

    helmet_model = YOLO(HELMET_MODEL_PATH)
    helmet_rows = []
    for start in range(0, frame_count, BATCH_SIZE_HELMET):
        batch = frames_bgr[start:start + BATCH_SIZE_HELMET]
        results = helmet_model.predict(
            source=batch,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            imgsz=YOLO_IMGSZ,
            device=0 if DEVICE == "cuda" else "cpu",
            verbose=False,
            stream=False,
        )
        for bi, res in enumerate(results):
            frame_idx = start + bi + 1
            if res.boxes is None or len(res.boxes) == 0:
                continue
            xyxy = res.boxes.xyxy.detach().cpu().numpy()
            confs = res.boxes.conf.detach().cpu().numpy()
            clss = res.boxes.cls.detach().cpu().numpy() if res.boxes.cls is not None else np.zeros(len(confs))
            for box, det_score, cls_id in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = box.tolist()
                x1 = max(0, min(frame_w - 1, int(round(x1))))
                y1 = max(0, min(frame_h - 1, int(round(y1))))
                x2 = max(x1 + 1, min(frame_w, int(round(x2))))
                y2 = max(y1 + 1, min(frame_h, int(round(y2))))
                helmet_rows.append({
                    "video": os.path.basename(VIDEO_PATH),
                    "frame": frame_idx,
                    "left": x1,
                    "top": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "det_score": float(det_score),
                    "yolo_cls": int(cls_id),
                    "view": "Sideline",
                })

    helmet_df = pd.DataFrame(helmet_rows)

    if len(helmet_df):
        crop_boxes = []
        for row in helmet_df.itertuples(index=False):
            box_xyxy = np.array([row.left, row.top, row.left + row.width, row.top + row.height], dtype=np.float32)
            crop_box = adapt_box_to_shape(extend_box(box_xyxy, size=CROP_SIZE), frame_h, frame_w)
            crop_boxes.append(crop_box)
        helmet_df = helmet_df.copy()
        helmet_df["crop_box"] = crop_boxes

    if len(helmet_df) == 0:
        pd.DataFrame(columns=["video", "frame", "left", "top", "width", "height", "det_score", "pred_cls", "impact_score"]).to_csv(OUTPUT_ALL_DETECTIONS_CSV, index=False)
        pd.DataFrame(columns=["video", "frame", "left", "top", "width", "height", "det_score", "pred_cls", "impact_score"]).to_csv(OUTPUT_FINAL_IMPACTS_CSV, index=False)
        raise SystemExit(0)

    clip_ds = HelmetClipDataset(helmet_df, frames_bgr, adj_frame_cache)
    clip_loader = DataLoader(
        clip_ds,
        batch_size=BATCH_SIZE_I3D,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    ort_session = get_onnx_session(I3D_ONNX_PATH)
    ort_input_name = ort_session.get_inputs()[0].name
    pred_cls = np.zeros(len(helmet_df), dtype=np.float32)
    for clips, indices in clip_loader:
        clips_np = clips.numpy().astype(np.float32, copy=False)
        logits = ort_session.run(None, {ort_input_name: clips_np})[0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        pred_cls[indices.numpy()] = probs.reshape(-1).astype(np.float32)

    helmet_df["pred_cls"] = pred_cls
    helmet_df["impact_score"] = helmet_df["det_score"] * helmet_df["pred_cls"]
    helmet_df.drop(columns=["crop_box"], errors="ignore").to_csv(OUTPUT_ALL_DETECTIONS_CSV, index=False)

    filtered_df = helmet_df[helmet_df.apply(sideline_keep, axis=1)].copy().reset_index(drop=True)
    filtered_df = filtered_df.drop(columns=["crop_box"], errors="ignore")
    filtered_df = expand_boxes_inplace(filtered_df, r=BOX_EXPANSION_RATIO, frame_w=frame_w, frame_h=frame_h)
    for _ in range(ADJ_N_TIMES):
        filtered_df = adjacency_postprocess(
            filtered_df,
            iou_threshold=ADJ_IOU_THRESHOLD,
            max_dist=ADJ_MAX_FRAME_DIST,
            min_cluster_size=ADJ_MIN_CLUSTER_SIZE,
            n_times=1,
        )
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df.to_csv(OUTPUT_FINAL_IMPACTS_CSV, index=False)

    openCvEventsCsv = os.path.join(OUTPUT_DIR, "potentialConcussions.csv")
    openCvEventsJson = os.path.join(OUTPUT_DIR, "finalConcussions.json")

    trackOnlySeconds = 2.0
    motionCheckSeconds = 2.0
    analysisFrameStride = 2

    minValidMotionSamples = 5
    seedFrameGap = 8
    seedIouThresh = 0.35
    minPredCls = 0.80

    uprightBodyWidthMult = 4.0
    uprightBodyHeightMult = 8.0
    sidewaysBodyWidthMult = 8.0
    sidewaysBodyHeightMult = 4.5
    bodyUpMult = 0.75
    sidewaysAspectThresh = 1.20

    refineSidewaysWidthMult = 1.30
    refineSidewaysHeightMult = 1.05
    refineUprightWidthMult = 1.10
    refineUprightHeightMult = 1.25

    bodyMotionThr = 0.035
    headMotionThr = 0.028
    bodyStrongMoveThr = 0.080
    headStrongMoveThr = 0.065
    helmetMotionThr = 0.030
    helmetStrongMoveThr = 0.055

    minHeadTrackedPoints = 10
    minBodyTrackedPoints = 16
    minConfidentVisibleSamples = 3

    bodyMissingStopSeconds = 1.0

    roiHistBins = 32
    roiHistSimilarityThr = 0.55
    minAppearanceCheckSamples = 2

    playerMaxCorners = 120
    playerQuality = 0.02
    playerMinDistance = 7
    playerBlockSize = 7

    cameraMaxCorners = 250
    cameraQuality = 0.01
    cameraMinDistance = 7
    cameraMinFeatures = 30
    cameraRansacThresh = 3.0

    lkParams = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )

    def boxIouXyxy(a, b):
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        inter = interW * interH
        if inter <= 0:
            return 0.0
        areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        denom = areaA + areaB - inter
        return 0.0 if denom <= 0 else inter / denom

    def clampRoi(x1, y1, x2, y2, frameW, frameH):
        x1 = int(max(0, min(frameW - 1, round(x1))))
        y1 = int(max(0, min(frameH - 1, round(y1))))
        x2 = int(max(x1 + 1, min(frameW, round(x2))))
        y2 = int(max(y1 + 1, min(frameH, round(y2))))
        return [x1, y1, x2, y2]

    def bodyScaleFromBox(boxXyxy):
        x1, y1, x2, y2 = boxXyxy
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return max(1.0, math.hypot(w, h))

    def helmetCenter(boxXyxy):
        x1, y1, x2, y2 = boxXyxy
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    def toGray(frameBgr):
        return cv2.cvtColor(frameBgr, cv2.COLOR_BGR2GRAY)

    def roiHistSimilarity(prevGray, currGray, roi, bins=32):
        x1, y1, x2, y2 = roi
        a = prevGray[y1:y2, x1:x2]
        b = currGray[y1:y2, x1:x2]
        if a is None or b is None or a.size == 0 or b.size == 0:
            return 0.0
        ha = cv2.calcHist([a], [0], None, [bins], [0, 256])
        hb = cv2.calcHist([b], [0], None, [bins], [0, 256])
        cv2.normalize(ha, ha)
        cv2.normalize(hb, hb)
        return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))

    def dedupeSeedImpacts(df):
        if len(df) == 0:
            return df.copy()
        df = df.sort_values(["frame", "impact_score"], ascending=[True, False]).reset_index(drop=True)
        keep = []
        for row in df.itertuples(index=False):
            currBox = [row.left, row.top, row.left + row.width, row.top + row.height]
            dup = False
            for kept in keep:
                if abs(int(row.frame) - int(kept.frame)) > seedFrameGap:
                    continue
                keptBox = [kept.left, kept.top, kept.left + kept.width, kept.top + kept.height]
                if boxIouXyxy(currBox, keptBox) >= seedIouThresh:
                    dup = True
                    break
            if not dup:
                keep.append(row)
        return pd.DataFrame([r._asdict() for r in keep])

    def createCsrtTracker():
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        raise RuntimeError("CSRT tracker not available in this OpenCV build.")

    def initTrackerOnHelmet(frameBgr, helmetBoxXyxy):
        tracker = createCsrtTracker()
        x1, y1, x2, y2 = helmetBoxXyxy
        x = int(round(float(x1)))
        y = int(round(float(y1)))
        w = int(round(float(x2 - x1)))
        h = int(round(float(y2 - y1)))
        w = max(1, w)
        h = max(1, h)
        fh, fw = frameBgr.shape[:2]
        x = max(0, min(fw - 1, x))
        y = max(0, min(fh - 1, y))
        w = max(1, min(w, fw - x))
        h = max(1, min(h, fh - y))
        tracker.init(frameBgr, (x, y, w, h))
        return tracker

    def updateTrackerBox(tracker, frameBgr, fallbackBox):
        ok, box = tracker.update(frameBgr)
        if not ok:
            return False, fallbackBox
        x, y, w, h = box
        x = int(round(float(x)))
        y = int(round(float(y)))
        w = max(1, int(round(float(w))))
        h = max(1, int(round(float(h))))
        tracked = clampRoi(x, y, x + w, y + h, frame_w, frame_h)
        return True, tracked

    def helmetBoxToBodyRoi(helmetBoxXyxy, frameW, frameH):
        hx1, hy1, hx2, hy2 = helmetBoxXyxy
        w = max(1.0, hx2 - hx1)
        h = max(1.0, hy2 - hy1)
        cx = 0.5 * (hx1 + hx2)
        aspect = w / h
        if aspect > sidewaysAspectThresh:
            bodyW = max(w * sidewaysBodyWidthMult, 180)
            bodyH = max(h * sidewaysBodyHeightMult, 140)
        else:
            bodyW = max(w * uprightBodyWidthMult, 90)
            bodyH = max(h * uprightBodyHeightMult, 180)
        x1 = cx - bodyW / 2.0
        x2 = cx + bodyW / 2.0
        y1 = hy1 - bodyUpMult * h
        y2 = y1 + bodyH
        return clampRoi(x1, y1, x2, y2, frameW, frameH)

    def refineBodyRoi(bodyRoi):
        x1, y1, x2, y2 = bodyRoi
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        aspect = w / h
        if aspect > 1.15:
            rw = w * refineSidewaysWidthMult
            rh = h * refineSidewaysHeightMult
        else:
            rw = w * refineUprightWidthMult
            rh = h * refineUprightHeightMult
        return clampRoi(cx - rw / 2.0, cy - rh / 2.0, cx + rw / 2.0, cy + rh / 2.0, frame_w, frame_h)

    def bodySubregionsFromBodyRoi(bodyRoi):
        x1, y1, x2, y2 = bodyRoi
        h = max(1, y2 - y1)
        headY2 = y1 + int(0.28 * h)
        torsoY1 = y1 + int(0.20 * h)
        torsoY2 = y1 + int(0.72 * h)
        headRoi = [x1, y1, x2, max(y1 + 1, headY2)]
        torsoRoi = [x1, torsoY1, x2, max(torsoY1 + 1, torsoY2)]
        return headRoi, torsoRoi

    def surroundingRingRoi(bodyRoi, expand=35):
        x1, y1, x2, y2 = bodyRoi
        return clampRoi(x1 - expand, y1 - expand, x2 + expand, y2 + expand, frame_w, frame_h)

    def getFeaturesInRoi(gray, roi, mask=None, maxCorners=100, quality=0.02, minDistance=7, blockSize=7):
        x1, y1, x2, y2 = roi
        crop = gray[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None
        roiMask = None
        if mask is not None:
            roiMask = mask[y1:y2, x1:x2]
        pts = cv2.goodFeaturesToTrack(
            crop,
            maxCorners=maxCorners,
            qualityLevel=quality,
            minDistance=minDistance,
            mask=roiMask,
            blockSize=blockSize,
        )
        if pts is None:
            return None
        pts[:, 0, 0] += x1
        pts[:, 0, 1] += y1
        return pts.astype(np.float32)

    def trackPointsLk(prevGray, currGray, pts0):
        if pts0 is None or len(pts0) == 0:
            return None, None
        pts1, st, err = cv2.calcOpticalFlowPyrLK(prevGray, currGray, pts0, None, **lkParams)
        if pts1 is None or st is None:
            return None, None
        good0 = pts0[st.flatten() == 1].reshape(-1, 2)
        good1 = pts1[st.flatten() == 1].reshape(-1, 2)
        if len(good0) == 0:
            return None, None
        return good0, good1

    def estimateCameraMotion(prevGray, currGray, bodyRoi):
        outer = surroundingRingRoi(bodyRoi, expand=35)
        bx1, by1, bx2, by2 = bodyRoi
        mask = np.ones(prevGray.shape, dtype=np.uint8) * 255
        mask[by1:by2, bx1:bx2] = 0
        pts0 = getFeaturesInRoi(
            prevGray,
            outer,
            mask=mask,
            maxCorners=cameraMaxCorners,
            quality=cameraQuality,
            minDistance=cameraMinDistance,
            blockSize=7,
        )
        if pts0 is None or len(pts0) < cameraMinFeatures:
            return np.zeros(2, dtype=np.float32)
        good0, good1 = trackPointsLk(prevGray, currGray, pts0)
        if good0 is None or len(good0) < cameraMinFeatures:
            return np.zeros(2, dtype=np.float32)
        m, _ = cv2.estimateAffinePartial2D(
            good0,
            good1,
            method=cv2.RANSAC,
            ransacReprojThreshold=cameraRansacThresh
        )
        if m is None:
            delta = np.median(good1 - good0, axis=0)
            return delta.astype(np.float32)
        return np.array([float(m[0, 2]), float(m[1, 2])], dtype=np.float32)

    def estimateMotionInRoi(prevGray, currGray, roi, cameraVec):
        pts0 = getFeaturesInRoi(
            prevGray,
            roi,
            maxCorners=playerMaxCorners,
            quality=playerQuality,
            minDistance=playerMinDistance,
            blockSize=playerBlockSize,
        )
        if pts0 is None or len(pts0) < 6:
            return None, 0
        good0, good1 = trackPointsLk(prevGray, currGray, pts0)
        if good0 is None or len(good0) < 6:
            return None, 0
        rawVecs = good1 - good0
        compVecs = rawVecs - cameraVec.reshape(1, 2)
        mags = np.linalg.norm(compVecs, axis=1)
        return float(np.median(mags)), int(len(good0))

    def analyzeFramePair(prevFrameBgr, currFrameBgr, bodyRoi):
        prevGray = toGray(prevFrameBgr)
        currGray = toGray(currFrameBgr)
        cameraVec = estimateCameraMotion(prevGray, currGray, bodyRoi)
        headRoi, torsoRoi = bodySubregionsFromBodyRoi(bodyRoi)
        headMotionPx, headPts = estimateMotionInRoi(prevGray, currGray, headRoi, cameraVec)
        bodyMotionPx, bodyPts = estimateMotionInRoi(prevGray, currGray, torsoRoi, cameraVec)
        scale = bodyScaleFromBox(bodyRoi)
        headMotionNorm = None if headMotionPx is None else float(headMotionPx / scale)
        bodyMotionNorm = None if bodyMotionPx is None else float(bodyMotionPx / scale)
        bodyVisibleConfidently = (headPts >= minHeadTrackedPoints) and (bodyPts >= minBodyTrackedPoints)
        return {
            "head_motion": headMotionNorm,
            "body_motion": bodyMotionNorm,
            "body_visible_confidently": bodyVisibleConfidently,
        }

    def trackHelmetWindow(startFrameIdx, endFrameIdx, initialHelmetBox):
        if startFrameIdx < 1:
            startFrameIdx = 1
        if endFrameIdx > len(frames_bgr):
            endFrameIdx = len(frames_bgr)
        if startFrameIdx > endFrameIdx:
            return {}, initialHelmetBox
        tracker = initTrackerOnHelmet(frames_bgr[startFrameIdx - 1], initialHelmetBox)
        trackedBoxes = {}
        currentHelmetBox = initialHelmetBox
        for frameIdx in range(startFrameIdx, endFrameIdx + 1):
            frame = frames_bgr[frameIdx - 1]
            if frameIdx == startFrameIdx:
                trackedHelmet = initialHelmetBox
            else:
                _, trackedHelmet = updateTrackerBox(tracker, frame, currentHelmetBox)
            currentHelmetBox = trackedHelmet
            trackedBoxes[frameIdx] = trackedHelmet
        return trackedBoxes, currentHelmetBox

    def analyzeMotionWindow(startFrameIdx, endFrameIdx, initialHelmetBox):
        if startFrameIdx < 1:
            startFrameIdx = 1
        if endFrameIdx > len(frames_bgr):
            endFrameIdx = len(frames_bgr)
        if startFrameIdx > endFrameIdx:
            return [], [], initialHelmetBox, False, False, False
        tracker = initTrackerOnHelmet(frames_bgr[startFrameIdx - 1], initialHelmetBox)
        motionRecords = []
        helmetMotionRecords = []
        prevFrameIdx = None
        currentHelmetBox = initialHelmetBox
        prevHelmetBox = initialHelmetBox
        stoppedEarlyForMotion = False
        stoppedEarlyForVisibility = False
        confidentVisibleSamples = 0
        badAppearanceSamples = 0
        abandonedForMissingBody = False
        missingBodyRun = 0
        missingBodyLimit = max(1, int(math.ceil(bodyMissingStopSeconds * fps / analysisFrameStride)))
        for frameIdx in range(startFrameIdx, endFrameIdx + 1, analysisFrameStride):
            frame = frames_bgr[frameIdx - 1]
            if frameIdx == startFrameIdx:
                trackedHelmet = initialHelmetBox
            else:
                _, trackedHelmet = updateTrackerBox(tracker, frame, currentHelmetBox)
            currentHelmetBox = trackedHelmet
            bodyRoi = refineBodyRoi(helmetBoxToBodyRoi(currentHelmetBox, frame_w, frame_h))
            if prevFrameIdx is not None:
                prevFrame = frames_bgr[prevFrameIdx - 1]
                prevGray = toGray(prevFrame)
                currGray = toGray(frame)
                histSim = roiHistSimilarity(prevGray, currGray, bodyRoi, bins=roiHistBins)
                if histSim < roiHistSimilarityThr:
                    badAppearanceSamples += 1
                    missingBodyRun += 1
                    if missingBodyRun >= missingBodyLimit:
                        stoppedEarlyForVisibility = True
                        abandonedForMissingBody = True
                        motionRecords = []
                        helmetMotionRecords = []
                        break
                    prevFrameIdx = frameIdx
                    prevHelmetBox = currentHelmetBox
                    continue
                result = analyzeFramePair(prevFrame, frame, bodyRoi)
                currCenter = helmetCenter(currentHelmetBox)
                prevCenter = helmetCenter(prevHelmetBox)
                helmetPx = float(np.linalg.norm(currCenter - prevCenter))
                helmetNorm = helmetPx / bodyScaleFromBox(bodyRoi)
                helmetMotionRecords.append(helmetNorm)
                if helmetNorm > helmetStrongMoveThr:
                    stoppedEarlyForMotion = True
                    break
                if not result["body_visible_confidently"]:
                    missingBodyRun += 1
                    if missingBodyRun >= missingBodyLimit:
                        stoppedEarlyForVisibility = True
                        abandonedForMissingBody = True
                        motionRecords = []
                        helmetMotionRecords = []
                        break
                    prevFrameIdx = frameIdx
                    prevHelmetBox = currentHelmetBox
                    continue
                missingBodyRun = 0
                confidentVisibleSamples += 1
                headMotion = result["head_motion"]
                bodyMotion = result["body_motion"]
                if headMotion is not None or bodyMotion is not None:
                    motionRecords.append(result)
                strongHead = headMotion is not None and not np.isnan(headMotion) and headMotion > headStrongMoveThr
                strongBody = bodyMotion is not None and not np.isnan(bodyMotion) and bodyMotion > bodyStrongMoveThr
                if strongHead or strongBody:
                    stoppedEarlyForMotion = True
                    break
            prevFrameIdx = frameIdx
            prevHelmetBox = currentHelmetBox
        if confidentVisibleSamples < minConfidentVisibleSamples:
            stoppedEarlyForVisibility = True
            motionRecords = []
        if badAppearanceSamples >= minAppearanceCheckSamples and confidentVisibleSamples == 0:
            stoppedEarlyForVisibility = True
            motionRecords = []
            helmetMotionRecords = []
        return (
            motionRecords,
            helmetMotionRecords,
            currentHelmetBox,
            stoppedEarlyForMotion,
            stoppedEarlyForVisibility,
            abandonedForMissingBody,
        )

    def evaluateWindow(motionList, helmetMotionList):
        headVals = [m["head_motion"] for m in motionList if m["head_motion"] is not None and not np.isnan(m["head_motion"])]
        bodyVals = [m["body_motion"] for m in motionList if m["body_motion"] is not None and not np.isnan(m["body_motion"])]
        helmetVals = [v for v in helmetMotionList if v is not None and not np.isnan(v)]
        enoughData = (
            len(headVals) >= minValidMotionSamples
            and len(bodyVals) >= minValidMotionSamples
            and len(helmetVals) >= minValidMotionSamples
        )
        if not enoughData:
            return True, False, {
                "head_med": np.nan,
                "body_med": np.nan,
                "helmet_med": np.nan,
                "head_max": np.nan,
                "body_max": np.nan,
                "helmet_max": np.nan,
                "enough_data": False,
            }
        headMed = float(np.median(headVals))
        bodyMed = float(np.median(bodyVals))
        helmetMed = float(np.median(helmetVals))
        headMax = float(np.max(headVals))
        bodyMax = float(np.max(bodyVals))
        helmetMax = float(np.max(helmetVals))
        headMoving = headMed > headMotionThr
        bodyMoving = bodyMed > bodyMotionThr
        helmetMoving = helmetMed > helmetMotionThr
        strongHeadMovement = headMax > headStrongMoveThr
        strongBodyMovement = bodyMax > bodyStrongMoveThr
        strongHelmetMovement = helmetMax > helmetStrongMoveThr
        purposefulMovement = (
            headMoving
            or bodyMoving
            or helmetMoving
            or strongHeadMovement
            or strongBodyMovement
            or strongHelmetMovement
        )
        motionless = (
            headMed <= headMotionThr
            and bodyMed <= bodyMotionThr
            and helmetMed <= helmetMotionThr
            and (not purposefulMovement)
        )
        return purposefulMovement, motionless, {
            "head_med": headMed,
            "body_med": bodyMed,
            "helmet_med": helmetMed,
            "head_max": headMax,
            "body_max": bodyMax,
            "helmet_max": helmetMax,
            "enough_data": True,
        }

    if len(filtered_df) == 0:
        eventsDf = pd.DataFrame(columns=[
            "event_id",
            "impact_frame",
            "helmet_left",
            "helmet_top",
            "helmet_width",
            "helmet_height",
            "det_score",
            "pred_cls",
            "impact_score",
            "head_motion_after_wait",
            "body_motion_after_wait",
            "helmet_motion_after_wait",
            "head_max_after_wait",
            "body_max_after_wait",
            "helmet_max_after_wait",
            "purposeful_movement_after_wait",
            "motionless_after_wait",
            "knocked_out_concussion_suspected",
        ])
    else:
        seedDf = dedupeSeedImpacts(filtered_df)
        trackOnlyFrames = max(1, int(math.ceil(trackOnlySeconds * fps)))
        motionCheckFrames = max(1, int(math.ceil(motionCheckSeconds * fps)))
        eventRows = []
        keptEventId = 0

        for row in seedDf.itertuples(index=False):
            if float(row.pred_cls) < minPredCls:
                continue

            keptEventId += 1
            impactFrame = int(row.frame)
            helmetBoxXyxy = [
                int(row.left),
                int(row.top),
                int(row.left + row.width),
                int(row.top + row.height),
            ]

            trackStart = impactFrame
            trackEnd = min(len(frames_bgr), impactFrame + trackOnlyFrames - 1)
            _, lastHelmetBox = trackHelmetWindow(trackStart, trackEnd, helmetBoxXyxy)
            trackWindowComplete = (trackEnd - trackStart + 1) >= trackOnlyFrames

            motionStart = trackEnd + 1
            motionEnd = min(len(frames_bgr), motionStart + motionCheckFrames - 1)

            if (not trackWindowComplete) or (motionStart > len(frames_bgr)):
                summary = {
                    "head_med": np.nan,
                    "body_med": np.nan,
                    "helmet_med": np.nan,
                    "head_max": np.nan,
                    "body_max": np.nan,
                    "helmet_max": np.nan,
                    "enough_data": False,
                }
                purposeful = True
                motionless = False
                motionWindowComplete = False
                stoppedEarlyForMotion = False
                stoppedEarlyForVisibility = True
                abandonedForMissingBody = False
                knockedOutConcussionSuspected = False
            else:
                (
                    motions,
                    helmetMotions,
                    _,
                    stoppedEarlyForMotion,
                    stoppedEarlyForVisibility,
                    abandonedForMissingBody,
                ) = analyzeMotionWindow(motionStart, motionEnd, lastHelmetBox)

                purposeful, motionless, summary = evaluateWindow(motions, helmetMotions)
                motionWindowComplete = (motionEnd - motionStart + 1) >= motionCheckFrames

                if abandonedForMissingBody:
                    knockedOutConcussionSuspected = False
                elif stoppedEarlyForVisibility:
                    knockedOutConcussionSuspected = False
                elif stoppedEarlyForMotion:
                    knockedOutConcussionSuspected = False
                elif not motionWindowComplete:
                    knockedOutConcussionSuspected = False
                elif not summary["enough_data"]:
                    knockedOutConcussionSuspected = False
                elif purposeful:
                    knockedOutConcussionSuspected = False
                elif motionless:
                    knockedOutConcussionSuspected = True
                else:
                    knockedOutConcussionSuspected = False

            eventRows.append({
                "event_id": keptEventId,
                "impact_frame": impactFrame,
                "helmet_left": int(row.left),
                "helmet_top": int(row.top),
                "helmet_width": int(row.width),
                "helmet_height": int(row.height),
                "det_score": float(row.det_score),
                "pred_cls": float(row.pred_cls),
                "impact_score": float(row.impact_score),
                "head_motion_after_wait": summary["head_med"],
                "body_motion_after_wait": summary["body_med"],
                "helmet_motion_after_wait": summary["helmet_med"],
                "head_max_after_wait": summary["head_max"],
                "body_max_after_wait": summary["body_max"],
                "helmet_max_after_wait": summary["helmet_max"],
                "purposeful_movement_after_wait": bool(purposeful),
                "motionless_after_wait": bool(motionless),
                "knocked_out_concussion_suspected": bool(knockedOutConcussionSuspected),
            })

        eventsDf = pd.DataFrame(eventRows)

    eventsDf.to_csv(openCvEventsCsv, index=False)

    jsonPayload = {
        "video": os.path.basename(VIDEO_PATH) if "VIDEO_PATH" in globals() else None,
        "num_events_analyzed": int(len(eventsDf)),
        "num_concussions_suspected": int(eventsDf["knocked_out_concussion_suspected"].sum()) if len(eventsDf) else 0,
        "concussion_events": eventsDf.loc[
            eventsDf["knocked_out_concussion_suspected"] == True
        ].to_dict(orient="records"),
        "all_events": eventsDf.to_dict(orient="records"),
    }

    with open(openCvEventsJson, "w") as f:
        json.dump(jsonPayload, f, indent=2, allow_nan=True)

    shutil.rmtree(OUTPUT_DIR)
    return openCvEventsJson