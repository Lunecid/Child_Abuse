"""
contextual_embedding_ca.py
===========================
Reviewer Defence: Contextual Embedding + CA / Procrustes 정합성 검증 모듈

목적
----
리뷰어가 제기한 "토큰 빈도 기반 CA가 아닌, 발화(utterance) 임베딩으로 만든
거리 구조가 CA와 정합(alignment)되는지 검증"에 대한 방어 코드.

핵심 아이디어
-----------
1. 기존 CA: 토큰 × 학대유형 빈도표 → CA 좌표 (row=학대유형, 2D)
2. BERT 접근: 발화 → BERT [CLS] 임베딩 → 학대유형별 중심(centroid) → 2D (PCA)
3. 두 공간을 Procrustes 정렬 + Mantel 거리 상관으로 비교
   → 정합성이 높으면, 토큰 빈도 CA 구조가 문맥적 의미(contextual semantics)와
     일관됨을 입증

추가 분석
--------
A) BERT 임베딩 기반 소프트 빈도표 생성 → CA 재수행 → 기존 CA와 비교
B) 토큰 수준(mean-pooled word embedding) → CA word 좌표와 Procrustes 비교
C) 학대유형 centroid 간 코사인 유사도 행렬 → CA 행좌표 거리와 상관

사용법
------
    from contextual_embedding_ca import run_bert_ca_validation

    results = run_bert_ca_validation(
        json_files=json_files,           # data/*.json 경로 리스트
        df_abuse_counts=df_abuse_counts,  # 기존 토큰×학대유형 빈도표
        abuse_stats_chi=abuse_stats_chi,  # chi² 통계
        bary_df=bary_df,                 # CA barycentric 좌표 (word × Dim1_bary, Dim2_bary)
        row_coords_2d=row_coords_2d,     # CA 행좌표 (abuse × Dim1, Dim2)
        out_dir="revision_output/bert_ca_validation",
    )

의존 라이브러리
-----------
- transformers (HuggingFace): BERT 모델 로딩
- torch: 텐서 연산
- scikit-learn: PCA, 클러스터링
- scipy: 통계 검정
- prince: CA (기존 파이프라인과 동일)
- 기존 파이프라인 모듈 (common, text, labels)
"""

from __future__ import annotations

import os
import sys
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── 기존 파이프라인 모듈 import ──
from abuse_pipeline.core import common as C
from abuse_pipeline.core.labels import classify_child_group, classify_abuse_main_sub
from abuse_pipeline.core.text import extract_child_speech, tokenize_korean

# ── 선택적 의존성 ──
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize as sk_normalize
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy.stats import spearmanr, pearsonr
    from scipy.spatial.distance import pdist, squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import prince
    HAS_PRINCE = True
except ImportError:
    HAS_PRINCE = False


# ═══════════════════════════════════════════════════════════════
#  Section 1: BERT 모델 로딩 및 임베딩 추출
# ═══════════════════════════════════════════════════════════════

# ── 지원 모델 목록 (한국어 BERT 계열) ──
SUPPORTED_MODELS = {
    "klue/bert-base":       "KLUE BERT (Korean Language Understanding Evaluation)",
    "monologg/kobert":      "KoBERT (SKT Brain)",
    "beomi/kcbert-base":    "KcBERT (댓글 기반 사전학습)",
    "klue/roberta-base":    "KLUE RoBERTa",
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS": "Korean Sentence-BERT",
}

DEFAULT_MODEL_NAME = "klue/bert-base"


class BERTUtteranceEncoder:
    """
    BERT 기반 발화(utterance) 인코더.

    각 발화 문장을 BERT에 넣어 [CLS] 토큰 또는 평균 풀링(mean pooling) 벡터를
    추출하는 클래스.

    Parameters
    ----------
    model_name : str
        HuggingFace 모델 이름 (기본: "klue/bert-base")
    pooling : str
        "cls" → [CLS] 토큰 벡터 사용
        "mean" → 모든 토큰 벡터의 평균 (문장 임베딩에 더 적합)
    device : str
        "cuda" 또는 "cpu"
    max_length : int
        토큰 최대 길이 (BERT 기본 512)

    Attributes
    ----------
    tokenizer : AutoTokenizer
    model : AutoModel
    hidden_size : int  (예: 768 for bert-base)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        pooling: str = "mean",
        device: str = None,
        max_length: int = 128,
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers 라이브러리가 필요합니다.\n"
                "  pip install transformers torch"
            )

        self.model_name = model_name
        self.pooling = pooling
        self.max_length = max_length

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[BERT] 모델 로딩: {model_name} (pooling={pooling}, device={self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size
        print(f"[BERT] 로딩 완료. hidden_size={self.hidden_size}")

    @(torch.no_grad() if HAS_TRANSFORMERS else lambda f: f)
    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        문장 리스트를 배치 단위로 인코딩하여 (n_texts, hidden_size) 행렬을 반환.

        Parameters
        ----------
        texts : list[str]
            인코딩할 문장 리스트
        batch_size : int
            한 번에 처리할 문장 수

        Returns
        -------
        np.ndarray, shape = (len(texts), hidden_size)
        """
        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encoded)
            # outputs.last_hidden_state: (batch, seq_len, hidden_size)

            if self.pooling == "cls":
                # [CLS] 토큰 = 시퀀스 첫 번째 토큰
                embeddings = outputs.last_hidden_state[:, 0, :]
            else:
                # mean pooling: attention_mask를 고려한 평균
                attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
                sum_embeddings = (outputs.last_hidden_state * attention_mask).sum(dim=1)
                count = attention_mask.sum(dim=1).clamp(min=1e-9)
                embeddings = sum_embeddings / count

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def encode_single(self, text: str) -> np.ndarray:
        """단일 문장 인코딩. shape = (hidden_size,)"""
        return self.encode_batch([text])[0]


# ═══════════════════════════════════════════════════════════════
#  Section 2: 발화 데이터 수집 (JSON → 학대유형별 발화 리스트)
# ═══════════════════════════════════════════════════════════════

def collect_utterances_by_abuse(
    json_files: list[str],
    abuse_order: list[str] = None,
    allowed_groups: set = None,
    min_utterance_len: int = 5,
) -> dict[str, list[str]]:
    """
    JSON 파일들에서 아동 발화(utterance)를 학대유형별로 수집.

    Parameters
    ----------
    json_files : list[str]
        data/*.json 경로 리스트
    abuse_order : list[str]
        학대유형 순서. 기본: ["방임", "정서학대", "신체학대", "성학대"]
    allowed_groups : set or None
        정서군 필터. None이면 전체, {"부정"}이면 부정 정서군만.
    min_utterance_len : int
        최소 발화 문자 수 (너무 짧은 발화 제외)

    Returns
    -------
    dict[str, list[str]]
        key=학대유형, value=해당 유형 아동들의 발화 리스트
        예: {"방임": ["엄마가 밥을 안 줘요", ...], "신체학대": [...], ...}
    """
    if abuse_order is None:
        abuse_order = ["방임", "정서학대", "신체학대", "성학대"]

    utterances_by_abuse = {a: [] for a in abuse_order}
    n_children = 0
    n_skipped = 0

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue

        # 정서군 필터링
        if allowed_groups is not None:
            group = classify_child_group(rec)
            if group not in allowed_groups:
                continue

        # 학대유형 분류
        main_abuse, _ = classify_abuse_main_sub(rec)
        if main_abuse not in abuse_order:
            n_skipped += 1
            continue

        # 아동 발화 추출
        texts = extract_child_speech(rec)
        texts = [t for t in texts if len(t) >= min_utterance_len]

        if texts:
            utterances_by_abuse[main_abuse].extend(texts)
            n_children += 1

    print(f"[BERT-DATA] 아동 수: {n_children}, 미분류: {n_skipped}")
    for a in abuse_order:
        print(f"  {a}: 발화 {len(utterances_by_abuse[a])}개")

    return utterances_by_abuse


def collect_utterances_with_metadata(
    json_files: list[str],
    abuse_order: list[str] = None,
    allowed_groups: set = None,
    min_utterance_len: int = 5,
) -> pd.DataFrame:
    """
    발화를 DataFrame으로 수집 (메타데이터 포함).

    Returns
    -------
    pd.DataFrame with columns:
        child_id, abuse_type, utterance, utterance_idx
    """
    if abuse_order is None:
        abuse_order = ["방임", "정서학대", "신체학대", "성학대"]

    rows = []
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue

        if allowed_groups is not None:
            group = classify_child_group(rec)
            if group not in allowed_groups:
                continue

        main_abuse, _ = classify_abuse_main_sub(rec)
        if main_abuse not in abuse_order:
            continue

        child_id = rec.get("info", {}).get("ID", os.path.basename(path))
        texts = extract_child_speech(rec)

        for idx, t in enumerate(texts):
            if len(t) >= min_utterance_len:
                rows.append({
                    "child_id": child_id,
                    "abuse_type": main_abuse,
                    "utterance": t,
                    "utterance_idx": idx,
                })

    df = pd.DataFrame(
        rows,
        columns=["child_id", "abuse_type", "utterance", "utterance_idx"],
    )
    n_children = int(df["child_id"].nunique()) if not df.empty else 0
    print(f"[BERT-DATA] 총 발화 수: {len(df)}, 아동 수: {n_children}")
    return df


# ═══════════════════════════════════════════════════════════════
#  Section 3: BERT 임베딩 기반 학대유형 Centroid 계산
# ═══════════════════════════════════════════════════════════════

def compute_abuse_centroids(
    df_utterances: pd.DataFrame,
    encoder: BERTUtteranceEncoder,
    abuse_order: list[str] = None,
    batch_size: int = 32,
    out_dir: str = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    학대유형별 발화 임베딩의 중심(centroid)을 계산.

    절차
    ----
    1. 모든 발화를 BERT로 인코딩 → (n_utterances, hidden_size)
    2. 학대유형별 발화 임베딩의 평균 = centroid → (n_abuse_types, hidden_size)
    3. PCA로 2D 투사 → 시각화 및 CA 비교용 좌표

    Parameters
    ----------
    df_utterances : pd.DataFrame
        columns: [child_id, abuse_type, utterance]
    encoder : BERTUtteranceEncoder
        BERT 인코더 인스턴스
    abuse_order : list[str]
        학대유형 순서
    batch_size : int
    out_dir : str or None
        결과 저장 디렉토리

    Returns
    -------
    df_centroids_2d : pd.DataFrame
        index=학대유형, columns=[Dim1_bert, Dim2_bert]
    all_embeddings : np.ndarray
        전체 발화 임베딩 (n_utterances, hidden_size)
    """
    if abuse_order is None:
        abuse_order = ["방임", "정서학대", "신체학대", "성학대"]

    # Step 1: 전체 발화 인코딩
    all_texts = df_utterances["utterance"].tolist()
    print(f"[BERT-ENCODE] 총 {len(all_texts)}개 발화 인코딩 시작...")
    all_embeddings = encoder.encode_batch(all_texts, batch_size=batch_size)
    print(f"[BERT-ENCODE] 완료. shape={all_embeddings.shape}")

    # Step 2: 학대유형별 centroid
    centroids = {}
    for abuse_type in abuse_order:
        mask = (df_utterances["abuse_type"] == abuse_type).values
        if mask.sum() == 0:
            print(f"  [WARN] {abuse_type}: 발화가 없어 centroid를 0으로 설정")
            centroids[abuse_type] = np.zeros(all_embeddings.shape[1])
        else:
            centroids[abuse_type] = all_embeddings[mask].mean(axis=0)
            print(f"  {abuse_type}: {mask.sum()}개 발화 → centroid norm={np.linalg.norm(centroids[abuse_type]):.4f}")

    centroid_matrix = np.vstack([centroids[a] for a in abuse_order])
    # shape: (n_abuse_types, hidden_size)

    # Step 3: PCA → 2D 투사
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn이 필요합니다.")

    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(centroid_matrix)

    df_centroids_2d = pd.DataFrame(
        coords_2d,
        index=abuse_order,
        columns=["Dim1_bert", "Dim2_bert"],
    )
    df_centroids_2d.index.name = "abuse_type"

    explained_var = pca.explained_variance_ratio_
    print(f"[BERT-PCA] 설명 분산: Dim1={explained_var[0]*100:.1f}%, Dim2={explained_var[1]*100:.1f}%")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        df_centroids_2d.to_csv(
            os.path.join(out_dir, "bert_abuse_centroids_2d.csv"),
            encoding="utf-8-sig",
        )
        print(f"[저장] BERT centroid 좌표 → {out_dir}/bert_abuse_centroids_2d.csv")

    return df_centroids_2d, all_embeddings


def compute_abuse_centroid_distances(
    df_centroids_2d: pd.DataFrame,
    abuse_order: list[str] = None,
    out_dir: str = None,
) -> pd.DataFrame:
    """
    학대유형 centroid 간 유클리드 거리 행렬 계산.

    Returns
    -------
    pd.DataFrame
        (n_abuse, n_abuse) 대칭 거리 행렬
    """
    if abuse_order is None:
        abuse_order = list(df_centroids_2d.index)

    coords = df_centroids_2d.loc[abuse_order].values
    dist_vec = pdist(coords, metric="euclidean")
    dist_mat = squareform(dist_vec)

    df_dist = pd.DataFrame(dist_mat, index=abuse_order, columns=abuse_order)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        df_dist.to_csv(
            os.path.join(out_dir, "bert_centroid_distance_matrix.csv"),
            encoding="utf-8-sig",
        )

    return df_dist


# ═══════════════════════════════════════════════════════════════
#  Section 4: CA 행좌표(학대유형)와 BERT Centroid의 Procrustes 정렬
# ═══════════════════════════════════════════════════════════════

def procrustes_ca_vs_bert(
    ca_row_coords: pd.DataFrame,
    bert_coords: pd.DataFrame,
    abuse_order: list[str] = None,
    out_dir: str = None,
    n_perm: int = 999,
) -> dict:
    """
    CA 행좌표(학대유형 좌표)와 BERT centroid 좌표를 Procrustes 분석으로 비교.

    수학적 배경
    ----------
    Procrustes 분석은 두 점 집합(Point Configuration)이 기하학적으로
    얼마나 유사한지 측정하는 방법입니다.

    두 행렬 X (CA 좌표)와 Y (BERT 좌표)가 있을 때:
    1. 중심화(centering): X₀ = X - mean(X), Y₀ = Y - mean(Y)
    2. 스케일링: X₀ = X₀ / ‖X₀‖_F, Y₀ = Y₀ / ‖Y₀‖_F
    3. 최적 회전 행렬 R 찾기: R = U Vᵀ (where M = X₀ᵀ Y₀ = U Σ Vᵀ, SVD)
    4. 정렬된 Y: Y_aligned = Y₀ R^T
    5. Disparity = ‖X₀ - Y_aligned‖²_F (0에 가까울수록 유사)

    [예시: 2×2 점 집합]
    X = [[1, 0], [0, 1]]  (CA: 방임, 정서학대의 2D 좌표)
    Y = [[0, 1], [-1, 0]] (BERT: 같은 학대유형의 2D 좌표, 90° 회전됨)
    → Procrustes는 Y를 X에 맞게 회전시켜 겹침 정도를 측정
    → disparity ≈ 0이면 "같은 기하 구조"

    Parameters
    ----------
    ca_row_coords : pd.DataFrame
        CA 행좌표. index=학대유형, columns=[Dim1, Dim2]
    bert_coords : pd.DataFrame
        BERT centroid 좌표. index=학대유형, columns=[Dim1_bert, Dim2_bert]
    abuse_order : list[str]
        학대유형 순서 (양쪽 행렬의 행 순서 정렬용)
    out_dir : str
        결과 저장 디렉토리
    n_perm : int
        Permutation test 반복 수 (p-value 계산용)

    Returns
    -------
    dict with keys:
        - n_points: 비교 점 수 (학대유형 수)
        - procrustes_disparity: Procrustes disparity (0에 가까울수록 유사)
        - procrustes_rmsd: Root Mean Square Deviation
        - procrustes_similarity: 1 - disparity (1에 가까울수록 유사)
        - mantel_r: Mantel 거리 상관 (Spearman)
        - mantel_p: Mantel p-value (permutation 기반)
        - ca_coords_centered: 중심화된 CA 좌표
        - bert_coords_aligned: Procrustes 정렬된 BERT 좌표
    """
    if abuse_order is None:
        abuse_order = ["방임", "정서학대", "신체학대", "성학대"]

    # 공통 학대유형만 사용
    common = [a for a in abuse_order if a in ca_row_coords.index and a in bert_coords.index]
    if len(common) < 3:
        print(f"[PROC] 공통 학대유형이 {len(common)}개밖에 없어 Procrustes 불가.")
        return None

    # CA 좌표 추출
    ca_cols = [c for c in ["Dim1", "Dim2"] if c in ca_row_coords.columns]
    if len(ca_cols) != 2:
        print(f"[PROC] CA 좌표 컬럼을 찾을 수 없습니다: {ca_row_coords.columns.tolist()}")
        return None

    bert_cols = [c for c in ["Dim1_bert", "Dim2_bert"] if c in bert_coords.columns]
    if len(bert_cols) != 2:
        print(f"[PROC] BERT 좌표 컬럼을 찾을 수 없습니다: {bert_coords.columns.tolist()}")
        return None

    X = ca_row_coords.loc[common, ca_cols].values.astype(float)
    Y = bert_coords.loc[common, bert_cols].values.astype(float)
    n = len(common)

    # Step 1: 중심화
    X0 = X - X.mean(axis=0, keepdims=True)
    Y0 = Y - Y.mean(axis=0, keepdims=True)

    # Step 2: 스케일링 (Frobenius norm)
    normX = np.linalg.norm(X0, "fro")
    normY = np.linalg.norm(Y0, "fro")
    if normX < 1e-12 or normY < 1e-12:
        print("[PROC] 좌표 분산이 너무 작습니다.")
        return None

    X0 /= normX
    Y0 /= normY

    # Step 3: SVD → 최적 회전 행렬
    M = X0.T @ Y0  # (2×2)
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt     # (2×2) 회전 행렬

    # Step 4: Y를 X에 정렬
    Y_aligned = Y0 @ R.T

    # Step 5: Disparity & RMSD
    diff = X0 - Y_aligned
    disparity = float(np.sum(diff ** 2))
    rmsd = float(np.sqrt(np.mean(diff ** 2)))
    similarity = 1.0 - disparity

    print(f"[PROC] Procrustes disparity = {disparity:.6f}")
    print(f"[PROC] Procrustes RMSD      = {rmsd:.6f}")
    print(f"[PROC] Procrustes similarity = {similarity:.6f}")

    # Step 6: Mantel 검정 (거리 행렬 상관)
    DX = squareform(pdist(X0, "euclidean"))
    DY = squareform(pdist(Y0, "euclidean"))
    iu = np.triu_indices(n, k=1)
    vX = DX[iu]
    vY = DY[iu]

    mantel_r = np.nan
    mantel_p = np.nan

    if HAS_SCIPY and len(vX) >= 3:
        mantel_r_val, _ = spearmanr(vX, vY)
        mantel_r = float(mantel_r_val)

        # Permutation test
        rng = np.random.default_rng(2025)
        perm_rs = []
        for _ in range(n_perm):
            perm_idx = rng.permutation(n)
            DY_perm = DY[perm_idx][:, perm_idx]
            vYp = DY_perm[iu]
            r_p, _ = spearmanr(vX, vYp)
            perm_rs.append(r_p)
        perm_rs = np.array(perm_rs)
        mantel_p = float((1 + np.sum(np.abs(perm_rs) >= np.abs(mantel_r))) / (n_perm + 1))

        print(f"[PROC] Mantel r = {mantel_r:.4f}, p = {mantel_p:.4f} (n_perm={n_perm})")

    # ── 시각화 ──
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

        # 색상 매핑
        abuse_colors = {
            "방임": "#1f77b4", "정서학대": "#ff7f0e",
            "신체학대": "#2ca02c", "성학대": "#d62728",
        }
        abuse_en = {
            "방임": "Neglect", "정서학대": "Emotional",
            "신체학대": "Physical", "성학대": "Sexual",
        }

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        # (A) CA 원래 좌표
        ax = axes[0]
        for i, a in enumerate(common):
            c = abuse_colors.get(a, "black")
            ax.scatter(X0[i, 0], X0[i, 1], s=200, color=c, zorder=5, edgecolor="black", linewidth=1.5)
            ax.annotate(abuse_en.get(a, a), (X0[i, 0], X0[i, 1]),
                        fontsize=11, ha="center", va="bottom",
                        xytext=(0, 12), textcoords="offset points",
                        fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title("(A) CA Row Coordinates\n(abuse types)", fontsize=12)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

        # (B) BERT centroid 좌표 (정렬 후)
        ax = axes[1]
        for i, a in enumerate(common):
            c = abuse_colors.get(a, "black")
            ax.scatter(Y_aligned[i, 0], Y_aligned[i, 1], s=200, color=c, zorder=5,
                       edgecolor="black", linewidth=1.5, marker="D")
            ax.annotate(abuse_en.get(a, a), (Y_aligned[i, 0], Y_aligned[i, 1]),
                        fontsize=11, ha="center", va="bottom",
                        xytext=(0, 12), textcoords="offset points",
                        fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title("(B) BERT Centroids\n(Procrustes-aligned)", fontsize=12)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

        # (C) 겹침(overlay)
        ax = axes[2]
        for i, a in enumerate(common):
            c = abuse_colors.get(a, "black")
            ax.scatter(X0[i, 0], X0[i, 1], s=160, color=c, zorder=5,
                       edgecolor="black", linewidth=1.0, marker="o", alpha=0.8)
            ax.scatter(Y_aligned[i, 0], Y_aligned[i, 1], s=160, color=c, zorder=5,
                       edgecolor="black", linewidth=1.0, marker="D", alpha=0.8)
            ax.plot(
                [X0[i, 0], Y_aligned[i, 0]],
                [X0[i, 1], Y_aligned[i, 1]],
                color=c, linewidth=1.5, linestyle="--", alpha=0.6,
            )
            mid_x = (X0[i, 0] + Y_aligned[i, 0]) / 2
            mid_y = (X0[i, 1] + Y_aligned[i, 1]) / 2
            ax.annotate(abuse_en.get(a, a), (mid_x, mid_y),
                        fontsize=10, ha="center", va="bottom",
                        xytext=(0, 8), textcoords="offset points")

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(
            f"(C) Overlay: CA ● vs BERT ◆\n"
            f"Procrustes d={disparity:.4f}, Mantel r={mantel_r:.3f}",
            fontsize=12,
        )
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

        legend_elements = [
            Line2D([0], [0], marker="o", color="gray", label="CA", markersize=10, linestyle="None"),
            Line2D([0], [0], marker="D", color="gray", label="BERT", markersize=10, linestyle="None"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        fig.suptitle(
            "Procrustes Alignment: Token-Frequency CA vs. BERT Contextual Embeddings",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()

        fig_path = os.path.join(out_dir, "procrustes_CA_vs_BERT_abuse_types.png")
        fig.savefig(fig_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"[저장] Procrustes 비교 시각화 → {fig_path}")

    return {
        "n_points": n,
        "abuse_types": common,
        "procrustes_disparity": disparity,
        "procrustes_rmsd": rmsd,
        "procrustes_similarity": similarity,
        "mantel_r": mantel_r,
        "mantel_p": mantel_p,
        "ca_coords_centered": X0,
        "bert_coords_aligned": Y_aligned,
    }


# ═══════════════════════════════════════════════════════════════
#  Section 5: BERT 기반 소프트 빈도표 → CA 재수행 → 비교
# ═══════════════════════════════════════════════════════════════

def build_bert_soft_frequency_table(
    df_utterances: pd.DataFrame,
    all_embeddings: np.ndarray,
    df_abuse_counts: pd.DataFrame,
    abuse_order: list[str] = None,
    n_clusters: int = 200,
    top_chi_for_ca: int = 200,
    out_dir: str = None,
) -> pd.DataFrame:
    """
    BERT 임베딩 기반 '소프트 빈도표'를 생성하여 기존 토큰 빈도표와 대비.

    아이디어
    ------
    기존 CA는 "토큰 w가 학대유형 a에 몇 번 등장"하는 빈도표를 사용합니다.
    그런데 같은 토큰이라도 문맥에 따라 다른 의미를 가질 수 있습니다.

    BERT 소프트 빈도표는 다음과 같이 만듭니다:
    1. 모든 발화 임베딩을 K-Means로 k개 클러스터로 나눔
       (각 클러스터 = "의미적 토픽" 또는 "contextual word sense")
    2. 각 클러스터 × 학대유형 교차표 생성
       → 기존 빈도표의 "토큰"이 "의미 클러스터"로 대체

    이렇게 만든 빈도표에 CA를 적용하면, 기존 토큰 빈도 CA와 비교 가능합니다.

    Parameters
    ----------
    df_utterances : pd.DataFrame
        columns: [child_id, abuse_type, utterance]
    all_embeddings : np.ndarray
        shape = (n_utterances, hidden_size)
    df_abuse_counts : pd.DataFrame
        기존 토큰 빈도표 (참고용)
    n_clusters : int
        K-Means 클러스터 수 (기존 CA에서 사용하는 단어 수에 맞춤)
    out_dir : str

    Returns
    -------
    pd.DataFrame
        index=cluster_id, columns=abuse_order
        (클러스터 × 학대유형 빈도표)
    """
    if abuse_order is None:
        abuse_order = ["방임", "정서학대", "신체학대", "성학대"]

    if not HAS_SKLEARN:
        raise ImportError("scikit-learn이 필요합니다.")

    n_utterances = len(df_utterances)
    assert all_embeddings.shape[0] == n_utterances, \
        f"발화 수({n_utterances})와 임베딩 수({all_embeddings.shape[0]})가 불일치"

    # Step 1: K-Means 클러스터링
    print(f"[BERT-SOFT] K-Means 클러스터링 (k={n_clusters})...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(all_embeddings)

    # Step 2: 클러스터 × 학대유형 교차표
    df_temp = df_utterances.copy()
    df_temp["cluster"] = cluster_labels

    cross_tab = pd.crosstab(df_temp["cluster"], df_temp["abuse_type"])

    # abuse_order에 맞게 컬럼 정렬 (누락 유형은 0으로)
    for a in abuse_order:
        if a not in cross_tab.columns:
            cross_tab[a] = 0
    cross_tab = cross_tab[abuse_order]
    cross_tab.index = [f"cluster_{i}" for i in cross_tab.index]

    print(f"[BERT-SOFT] 소프트 빈도표 shape: {cross_tab.shape}")
    print(f"[BERT-SOFT] 총 빈도 합: {cross_tab.values.sum()}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        cross_tab.to_csv(
            os.path.join(out_dir, "bert_soft_frequency_table.csv"),
            encoding="utf-8-sig",
        )

    return cross_tab


def run_ca_on_bert_soft_table(
    soft_freq_table: pd.DataFrame,
    abuse_order: list[str] = None,
    out_dir: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """
    BERT 소프트 빈도표에 대해 CA를 수행.

    Returns
    -------
    row_coords_2d : pd.DataFrame
        행좌표 (학대유형 × Dim1, Dim2)
    col_coords_2d : pd.DataFrame
        열좌표 (클러스터 × Dim1_bary, Dim2_bary)
    lam1, lam2 : float
        차원별 관성 비율
    """
    if not HAS_PRINCE:
        raise ImportError("prince 라이브러리가 필요합니다.")
    if abuse_order is None:
        abuse_order = ["방임", "정서학대", "신체학대", "성학대"]

    # rows=abuse, cols=cluster
    X = soft_freq_table.T  # (abuse × cluster) → CA 입력

    # 빈 행/열 제거
    X = X.loc[:, X.sum(axis=0) > 0]
    X = X.loc[X.sum(axis=1) > 0, :]

    ca = prince.CA(n_components=2, n_iter=10, copy=True, check_input=True, random_state=42)
    ca.fit(X)

    row_coords = ca.row_coordinates(X)
    row_coords_2d = row_coords.iloc[:, :2].copy()
    row_coords_2d.columns = ["Dim1", "Dim2"]

    col_coords = ca.column_coordinates(X)
    col_coords_2d = col_coords.iloc[:, :2].copy()
    col_coords_2d.columns = ["Dim1_bary", "Dim2_bary"]

    # eigenvalues / inertia
    eigenvalues = ca.eigenvalues_
    total_inertia = sum(eigenvalues) if len(eigenvalues) > 0 else 1e-12
    lam1 = eigenvalues[0] / total_inertia if len(eigenvalues) > 0 else 0
    lam2 = eigenvalues[1] / total_inertia if len(eigenvalues) > 1 else 0

    # chi² test
    obs = X.values
    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    N = float(obs.sum())
    expected = row_sums @ col_sums / max(N, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_sq_matrix = (obs - expected) ** 2 / expected
        chi_sq_matrix = np.nan_to_num(chi_sq_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    chi_total = float(chi_sq_matrix.sum())

    r, c = obs.shape
    df_val = (r - 1) * (c - 1)

    print(f"[BERT-CA] chi² = {chi_total:.2f}, df = {df_val}")
    print(f"[BERT-CA] Total inertia = {total_inertia:.6f}")
    print(f"[BERT-CA] Dim1 = {lam1*100:.1f}%, Dim2 = {lam2*100:.1f}%")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        row_coords_2d.to_csv(
            os.path.join(out_dir, "bert_ca_row_coords_2d.csv"),
            encoding="utf-8-sig",
        )
        summary = pd.DataFrame([{
            "N": N, "chi2": chi_total, "df": df_val,
            "total_inertia": total_inertia,
            "dim1_pct": lam1 * 100, "dim2_pct": lam2 * 100,
        }])
        summary.to_csv(
            os.path.join(out_dir, "bert_ca_summary.csv"),
            encoding="utf-8-sig", index=False,
        )

    return row_coords_2d, col_coords_2d, lam1, lam2


# ═══════════════════════════════════════════════════════════════
#  Section 6: 3중 비교 — Original CA vs BERT-CA vs BERT-PCA
# ═══════════════════════════════════════════════════════════════

def triple_comparison(
    ca_original_rows: pd.DataFrame,
    ca_bert_rows: pd.DataFrame,
    bert_pca_coords: pd.DataFrame,
    abuse_order: list[str] = None,
    out_dir: str = None,
    n_perm: int = 999,
) -> pd.DataFrame:
    """
    세 가지 공간의 학대유형 좌표를 쌍(pair)별로 Procrustes + Mantel 비교.

    비교 쌍:
    1. Original CA ↔ BERT-CA (소프트 빈도표 기반)
    2. Original CA ↔ BERT PCA (직접 임베딩)
    3. BERT-CA ↔ BERT PCA

    → 세 쌍 모두에서 높은 정합성이면,
      "토큰 빈도 CA = BERT 의미 구조 = BERT 소프트 빈도 CA"
      가 서로 일관됨을 강력히 입증.

    Returns
    -------
    pd.DataFrame
        각 쌍의 Procrustes disparity, RMSD, Mantel r/p 요약
    """
    if abuse_order is None:
        abuse_order = ["방임", "정서학대", "신체학대", "성학대"]

    pairs = [
        ("Original_CA", "BERT_SoftCA", ca_original_rows, ca_bert_rows,
         ["Dim1", "Dim2"], ["Dim1", "Dim2"]),
        ("Original_CA", "BERT_PCA", ca_original_rows, bert_pca_coords,
         ["Dim1", "Dim2"], ["Dim1_bert", "Dim2_bert"]),
        ("BERT_SoftCA", "BERT_PCA", ca_bert_rows, bert_pca_coords,
         ["Dim1", "Dim2"], ["Dim1_bert", "Dim2_bert"]),
    ]

    results = []
    for name1, name2, df1, df2, cols1, cols2 in pairs:
        common = [a for a in abuse_order if a in df1.index and a in df2.index]
        if len(common) < 3:
            results.append({
                "pair": f"{name1} ↔ {name2}",
                "n_points": len(common),
                "disparity": np.nan,
                "rmsd": np.nan,
                "mantel_r": np.nan,
                "mantel_p": np.nan,
            })
            continue

        X = df1.loc[common, cols1].values.astype(float)
        Y = df2.loc[common, cols2].values.astype(float)
        n = len(common)

        # Procrustes
        X0 = X - X.mean(0, keepdims=True)
        Y0 = Y - Y.mean(0, keepdims=True)
        nX = np.linalg.norm(X0, "fro")
        nY = np.linalg.norm(Y0, "fro")
        if nX < 1e-12 or nY < 1e-12:
            results.append({
                "pair": f"{name1} ↔ {name2}",
                "n_points": n,
                "disparity": np.nan,
                "rmsd": np.nan,
                "mantel_r": np.nan,
                "mantel_p": np.nan,
            })
            continue

        X0 /= nX
        Y0 /= nY
        M = X0.T @ Y0
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        Y_al = Y0 @ R.T
        disp = float(np.sum((X0 - Y_al) ** 2))
        rmsd_val = float(np.sqrt(np.mean((X0 - Y_al) ** 2)))

        # Mantel
        DX = squareform(pdist(X0))
        DY = squareform(pdist(Y0))
        iu = np.triu_indices(n, k=1)
        vX, vY = DX[iu], DY[iu]

        m_r, m_p = np.nan, np.nan
        if HAS_SCIPY and len(vX) >= 3:
            m_r, _ = spearmanr(vX, vY)
            m_r = float(m_r)
            rng = np.random.default_rng(2025)
            perm_rs = []
            for _ in range(n_perm):
                pi = rng.permutation(n)
                vYp = DY[pi][:, pi][iu]
                rp, _ = spearmanr(vX, vYp)
                perm_rs.append(rp)
            m_p = float((1 + np.sum(np.abs(perm_rs) >= np.abs(m_r))) / (n_perm + 1))

        results.append({
            "pair": f"{name1} ↔ {name2}",
            "n_points": n,
            "disparity": disp,
            "rmsd": rmsd_val,
            "similarity": 1.0 - disp,
            "mantel_r": m_r,
            "mantel_p": m_p,
        })
        print(f"[3-WAY] {name1} ↔ {name2}: d={disp:.4f}, Mantel r={m_r:.3f} (p={m_p:.4f})")

    df_results = pd.DataFrame(results)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        df_results.to_csv(
            os.path.join(out_dir, "triple_comparison_procrustes_mantel.csv"),
            encoding="utf-8-sig", index=False,
        )
        print(f"[저장] 3중 비교 요약 → {out_dir}/triple_comparison_procrustes_mantel.csv")

    return df_results


# ═══════════════════════════════════════════════════════════════
#  Section 7: BERT 토큰 수준 임베딩 vs CA word 좌표 비교
# ═══════════════════════════════════════════════════════════════

def compute_bert_word_embeddings(
    words: list[str],
    encoder: BERTUtteranceEncoder,
) -> pd.DataFrame:
    """
    CA에서 사용하는 단어(토큰)를 BERT로 인코딩하여 단어-수준 임베딩 생성.

    주의: BERT는 subword tokenizer를 사용하므로, 한 단어가 여러 subword로
    쪼개질 수 있음. 이 경우 subword 임베딩의 평균을 사용.

    Parameters
    ----------
    words : list[str]
        CA chi² 상위 단어 리스트
    encoder : BERTUtteranceEncoder

    Returns
    -------
    pd.DataFrame
        index=word, columns=[Dim1_bert_word, Dim2_bert_word]
    """
    # 단어 자체를 "문장"으로 인코딩 (단어 임베딩 근사)
    embeddings = encoder.encode_batch(words, batch_size=64)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    df = pd.DataFrame(
        coords,
        index=words,
        columns=["Dim1_bert_word", "Dim2_bert_word"],
    )
    df.index.name = "word"

    print(f"[BERT-WORD] {len(words)}개 단어 임베딩 → 2D PCA")
    print(f"[BERT-WORD] 설명 분산: {pca.explained_variance_ratio_[0]*100:.1f}%, {pca.explained_variance_ratio_[1]*100:.1f}%")

    return df


def procrustes_ca_words_vs_bert_words(
    bary_df: pd.DataFrame,
    bert_word_df: pd.DataFrame,
    out_dir: str = None,
    n_perm: int = 499,
) -> dict:
    """
    CA barycentric word 좌표와 BERT word 좌표를 Procrustes + Mantel로 비교.

    이 비교는 "단어 수준에서도 CA 구조와 BERT 구조가 일치하는가"를 검증합니다.

    Parameters
    ----------
    bary_df : pd.DataFrame
        CA barycentric 좌표. index=word, columns에 Dim1_bary, Dim2_bary
    bert_word_df : pd.DataFrame
        BERT 단어 좌표. index=word, columns에 Dim1_bert_word, Dim2_bert_word

    Returns
    -------
    dict: procrustes_disparity, mantel_r, mantel_p 등
    """
    # 공통 단어 찾기
    common_words = list(set(bary_df.index) & set(bert_word_df.index))
    if len(common_words) < 10:
        print(f"[WORD-PROC] 공통 단어가 {len(common_words)}개밖에 없어 비교 불가.")
        return None

    print(f"[WORD-PROC] 공통 단어 수: {len(common_words)}")

    X = bary_df.loc[common_words, ["Dim1_bary", "Dim2_bary"]].values.astype(float)
    Y = bert_word_df.loc[common_words, ["Dim1_bert_word", "Dim2_bert_word"]].values.astype(float)
    n = len(common_words)

    # Procrustes
    X0 = X - X.mean(0, keepdims=True)
    Y0 = Y - Y.mean(0, keepdims=True)
    nX, nY = np.linalg.norm(X0, "fro"), np.linalg.norm(Y0, "fro")
    if nX < 1e-12 or nY < 1e-12:
        return None

    X0 /= nX
    Y0 /= nY
    M = X0.T @ Y0
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    Y_aligned = Y0 @ R.T

    disparity = float(np.sum((X0 - Y_aligned) ** 2))
    rmsd = float(np.sqrt(np.mean((X0 - Y_aligned) ** 2)))

    # Mantel (거리 행렬 상관) — 단어 수가 많을 수 있으므로 서브샘플
    max_n_mantel = 250
    if n > max_n_mantel:
        rng = np.random.default_rng(42)
        sub_idx = rng.choice(n, size=max_n_mantel, replace=False)
        X_sub, Y_sub = X0[sub_idx], Y0[sub_idx]
        n_mantel = max_n_mantel
    else:
        X_sub, Y_sub = X0, Y0
        n_mantel = n

    DX = squareform(pdist(X_sub))
    DY = squareform(pdist(Y_sub))
    iu = np.triu_indices(n_mantel, k=1)
    vX, vY = DX[iu], DY[iu]

    mantel_r, mantel_p = np.nan, np.nan
    if HAS_SCIPY and len(vX) >= 3:
        mantel_r, _ = spearmanr(vX, vY)
        mantel_r = float(mantel_r)

        rng = np.random.default_rng(2025)
        perm_rs = []
        for _ in range(n_perm):
            pi = rng.permutation(n_mantel)
            vYp = DY[pi][:, pi][iu]
            rp, _ = spearmanr(vX, vYp)
            perm_rs.append(rp)
        mantel_p = float((1 + np.sum(np.abs(perm_rs) >= np.abs(mantel_r))) / (n_perm + 1))

    print(f"[WORD-PROC] Procrustes d={disparity:.4f}, RMSD={rmsd:.4f}")
    print(f"[WORD-PROC] Mantel r={mantel_r:.4f}, p={mantel_p:.4f}")

    # 시각화
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(X0[:, 0], X0[:, 1], s=15, alpha=0.5, label="CA (bary)", color="#1f77b4")
        ax.scatter(Y_aligned[:, 0], Y_aligned[:, 1], s=15, alpha=0.5, label="BERT (aligned)", color="#d62728")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title(
            f"Procrustes: CA word coords vs BERT word embeddings\n"
            f"n={n}, disparity={disparity:.4f}, Mantel r={mantel_r:.3f} (p={mantel_p:.4f})"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "procrustes_CA_words_vs_BERT_words.png"), dpi=200)
        plt.close(fig)
        print(f"[저장] 단어 수준 Procrustes 시각화 → {out_dir}")

    return {
        "n_words": n,
        "procrustes_disparity": disparity,
        "procrustes_rmsd": rmsd,
        "mantel_r": mantel_r,
        "mantel_p": mantel_p,
        "n_mantel_subsample": n_mantel,
    }


# ═══════════════════════════════════════════════════════════════
#  Section 8: 코사인 유사도 행렬 비교 (학대유형 간)
# ═══════════════════════════════════════════════════════════════

def cosine_similarity_comparison(
    ca_row_coords: pd.DataFrame,
    bert_centroids_full: np.ndarray,
    abuse_order: list[str],
    df_utterances: pd.DataFrame,
    out_dir: str = None,
) -> dict:
    """
    CA 학대유형 행좌표 간 코사인 유사도 행렬과
    BERT 학대유형 centroid 간 코사인 유사도 행렬을 비교.

    이 분석은 "학대유형 간 관계 구조"가 두 공간에서 일치하는지 검증합니다.

    비교 방법:
    - 두 유사도 행렬의 상삼각 원소들 간 Pearson/Spearman 상관
    - 높은 상관 = "두 공간에서 학대유형 간 관계 패턴이 동일"

    Returns
    -------
    dict with cosine matrices and correlations
    """
    # CA 코사인 유사도
    ca_cols = [c for c in ["Dim1", "Dim2"] if c in ca_row_coords.columns]
    ca_vecs = ca_row_coords.loc[abuse_order, ca_cols].values.astype(float)
    ca_norms = np.linalg.norm(ca_vecs, axis=1, keepdims=True)
    ca_norms[ca_norms < 1e-12] = 1e-12
    ca_unit = ca_vecs / ca_norms
    ca_cos = ca_unit @ ca_unit.T

    # BERT 코사인 유사도 (고차원 원래 centroid 사용)
    # bert_centroids_full: (n_abuse, hidden_size)
    centroids = {}
    for i, abuse_type in enumerate(abuse_order):
        mask = (df_utterances["abuse_type"] == abuse_type).values
        if mask.sum() > 0:
            centroids[abuse_type] = bert_centroids_full[mask].mean(axis=0)
        else:
            centroids[abuse_type] = np.zeros(bert_centroids_full.shape[1])

    bert_vecs = np.vstack([centroids[a] for a in abuse_order])
    bert_norms = np.linalg.norm(bert_vecs, axis=1, keepdims=True)
    bert_norms[bert_norms < 1e-12] = 1e-12
    bert_unit = bert_vecs / bert_norms
    bert_cos = bert_unit @ bert_unit.T

    # 상삼각 원소 추출 및 상관
    n = len(abuse_order)
    iu = np.triu_indices(n, k=1)
    v_ca = ca_cos[iu]
    v_bert = bert_cos[iu]

    pearson_r, pearson_p = np.nan, np.nan
    spearman_r, spearman_p = np.nan, np.nan

    if HAS_SCIPY and len(v_ca) >= 3:
        pearson_r, pearson_p = pearsonr(v_ca, v_bert)
        spearman_r, spearman_p = spearmanr(v_ca, v_bert)

    print(f"[COS-COMP] Pearson r={pearson_r:.4f} (p={pearson_p:.4f})")
    print(f"[COS-COMP] Spearman r={spearman_r:.4f} (p={spearman_p:.4f})")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

        abuse_en = {"방임": "Neglect", "정서학대": "Emotional",
                    "신체학대": "Physical", "성학대": "Sexual"}
        labels = [abuse_en.get(a, a) for a in abuse_order]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # CA 코사인
        im1 = axes[0].imshow(ca_cos, cmap="RdYlBu_r", vmin=-1, vmax=1)
        axes[0].set_xticks(range(n))
        axes[0].set_yticks(range(n))
        axes[0].set_xticklabels(labels, rotation=45, ha="right")
        axes[0].set_yticklabels(labels)
        axes[0].set_title("(A) CA Cosine Similarity")
        for i_r in range(n):
            for j_c in range(n):
                axes[0].text(j_c, i_r, f"{ca_cos[i_r, j_c]:.2f}",
                            ha="center", va="center", fontsize=9)
        plt.colorbar(im1, ax=axes[0], fraction=0.046)

        # BERT 코사인
        im2 = axes[1].imshow(bert_cos, cmap="RdYlBu_r", vmin=-1, vmax=1)
        axes[1].set_xticks(range(n))
        axes[1].set_yticks(range(n))
        axes[1].set_xticklabels(labels, rotation=45, ha="right")
        axes[1].set_yticklabels(labels)
        axes[1].set_title("(B) BERT Cosine Similarity")
        for i_r in range(n):
            for j_c in range(n):
                axes[1].text(j_c, i_r, f"{bert_cos[i_r, j_c]:.2f}",
                            ha="center", va="center", fontsize=9)
        plt.colorbar(im2, ax=axes[1], fraction=0.046)

        # 산점도
        axes[2].scatter(v_ca, v_bert, s=80, alpha=0.8, color="#2ca02c", edgecolor="black")
        for idx, (x, y) in enumerate(zip(v_ca, v_bert)):
            i_r, j_c = iu[0][idx], iu[1][idx]
            pair_label = f"{labels[i_r][:3]}-{labels[j_c][:3]}"
            axes[2].annotate(pair_label, (x, y), fontsize=8,
                            xytext=(5, 5), textcoords="offset points")
        axes[2].plot([-1, 1], [-1, 1], "k--", alpha=0.3)
        axes[2].set_xlabel("CA Cosine Similarity")
        axes[2].set_ylabel("BERT Cosine Similarity")
        axes[2].set_title(
            f"(C) CA vs BERT Cosine\n"
            f"Pearson r={pearson_r:.3f}, Spearman r={spearman_r:.3f}"
        )

        fig.suptitle(
            "Cosine Similarity Between Abuse Types:\nToken-Frequency CA vs. BERT Contextual Embeddings",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "cosine_similarity_CA_vs_BERT.png"),
                   dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[저장] 코사인 유사도 비교 시각화 → {out_dir}")

    return {
        "ca_cosine_matrix": ca_cos,
        "bert_cosine_matrix": bert_cos,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


# ═══════════════════════════════════════════════════════════════
#  Section 9: 통합 실행 함수
# ═══════════════════════════════════════════════════════════════

def run_bert_ca_validation(
    json_files: list[str],
    df_abuse_counts: pd.DataFrame = None,
    abuse_stats_chi: pd.DataFrame = None,
    bary_df: pd.DataFrame = None,
    row_coords_2d: pd.DataFrame = None,
    out_dir: str = "revision_output/bert_ca_validation",
    model_name: str = DEFAULT_MODEL_NAME,
    pooling: str = "mean",
    n_clusters: int = 200,
    batch_size: int = 32,
    n_perm: int = 999,
    abuse_order: list[str] = None,
    allowed_groups: set = None,
) -> dict:
    """
    BERT 기반 CA 검증 전체 파이프라인.

    실행 순서
    --------
    1. JSON 파일에서 학대유형별 발화 수집
    2. BERT 인코더 로딩 및 발화 임베딩 생성
    3. 학대유형별 centroid 계산 및 PCA 투사
    4. [Analysis A] CA 행좌표 vs BERT centroid → Procrustes + Mantel
    5. [Analysis B] BERT 소프트 빈도표 → CA 재수행 → 기존 CA와 비교
    6. [Analysis C] 3중 비교 (Original CA ↔ BERT-CA ↔ BERT PCA)
    7. [Analysis D] 단어 수준 BERT vs CA barycentric → Procrustes
    8. [Analysis E] 코사인 유사도 행렬 비교

    Parameters
    ----------
    json_files : list[str]
        data/*.json 경로 리스트
    df_abuse_counts : pd.DataFrame
        기존 토큰×학대유형 빈도표. None이면 일부 분석 건너뜀.
    abuse_stats_chi : pd.DataFrame
        chi² 통계. None이면 단어 수준 비교 건너뜀.
    bary_df : pd.DataFrame
        CA barycentric word 좌표. index=word, columns에 Dim1_bary, Dim2_bary.
    row_coords_2d : pd.DataFrame
        CA 행좌표 (학대유형 좌표). index=학대유형, columns=[Dim1, Dim2].
    out_dir : str
        결과 저장 디렉토리
    model_name : str
        BERT 모델 이름
    pooling : str
        "cls" or "mean"
    n_clusters : int
        소프트 빈도표용 K-Means 클러스터 수
    batch_size : int
    n_perm : int
        Permutation test 반복 수
    abuse_order : list[str]
    allowed_groups : set or None

    Returns
    -------
    dict with all analysis results
    """
    if abuse_order is None:
        abuse_order = ["방임", "정서학대", "신체학대", "성학대"]

    os.makedirs(out_dir, exist_ok=True)
    results = {}

    print("\n" + "=" * 72)
    print("[BERT-CA] Contextual Embedding + CA/Procrustes 검증 시작")
    print(f"[BERT-CA] 모델: {model_name}, 풀링: {pooling}")
    print("=" * 72)

    # ── Step 1: 발화 수집 ──
    print("\n── Step 1: 발화 수집 ──")
    df_utt = collect_utterances_with_metadata(
        json_files, abuse_order=abuse_order, allowed_groups=allowed_groups,
    )
    if df_utt.empty:
        print("[ERROR] 수집된 발화가 없습니다.")
        return results
    results["df_utterances"] = df_utt

    # ── Step 2: BERT 인코더 로딩 ──
    print("\n── Step 2: BERT 인코더 로딩 ──")
    encoder = BERTUtteranceEncoder(
        model_name=model_name, pooling=pooling,
    )

    # ── Step 3: 학대유형별 centroid ──
    print("\n── Step 3: 학대유형별 centroid 계산 ──")
    bert_centroids_2d, all_embeddings = compute_abuse_centroids(
        df_utt, encoder, abuse_order=abuse_order,
        batch_size=batch_size, out_dir=out_dir,
    )
    results["bert_centroids_2d"] = bert_centroids_2d
    results["all_embeddings"] = all_embeddings

    # ── Analysis A: CA 행좌표 vs BERT centroid ──
    if row_coords_2d is not None:
        print("\n── Analysis A: CA vs BERT Procrustes (학대유형 수준) ──")
        proc_result = procrustes_ca_vs_bert(
            ca_row_coords=row_coords_2d,
            bert_coords=bert_centroids_2d,
            abuse_order=abuse_order,
            out_dir=out_dir,
            n_perm=n_perm,
        )
        results["procrustes_abuse_level"] = proc_result

    # ── Analysis B: BERT 소프트 빈도표 → CA ──
    print("\n── Analysis B: BERT 소프트 빈도표 → CA ──")
    soft_freq = build_bert_soft_frequency_table(
        df_utt, all_embeddings, df_abuse_counts,
        abuse_order=abuse_order, n_clusters=n_clusters, out_dir=out_dir,
    )
    results["soft_frequency_table"] = soft_freq

    bert_ca_rows, bert_ca_cols, lam1, lam2 = run_ca_on_bert_soft_table(
        soft_freq, abuse_order=abuse_order, out_dir=out_dir,
    )
    results["bert_ca_row_coords"] = bert_ca_rows
    results["bert_ca_col_coords"] = bert_ca_cols

    # ── Analysis C: 3중 비교 ──
    if row_coords_2d is not None:
        print("\n── Analysis C: 3중 비교 ──")
        triple_df = triple_comparison(
            ca_original_rows=row_coords_2d,
            ca_bert_rows=bert_ca_rows,
            bert_pca_coords=bert_centroids_2d,
            abuse_order=abuse_order,
            out_dir=out_dir,
            n_perm=n_perm,
        )
        results["triple_comparison"] = triple_df

    # ── Analysis D: 단어 수준 비교 ──
    if bary_df is not None and abuse_stats_chi is not None:
        print("\n── Analysis D: 단어 수준 CA vs BERT Procrustes ──")
        chi_sorted = abuse_stats_chi.sort_values("chi2", ascending=False)
        top_words = chi_sorted.head(200).index.tolist()

        bert_word_df = compute_bert_word_embeddings(top_words, encoder)
        results["bert_word_coords"] = bert_word_df

        word_proc = procrustes_ca_words_vs_bert_words(
            bary_df=bary_df,
            bert_word_df=bert_word_df,
            out_dir=out_dir,
            n_perm=min(n_perm, 499),
        )
        results["procrustes_word_level"] = word_proc

    # ── Analysis E: 코사인 유사도 비교 ──
    if row_coords_2d is not None:
        print("\n── Analysis E: 코사인 유사도 행렬 비교 ──")
        cos_result = cosine_similarity_comparison(
            ca_row_coords=row_coords_2d,
            bert_centroids_full=all_embeddings,
            abuse_order=abuse_order,
            df_utterances=df_utt,
            out_dir=out_dir,
        )
        results["cosine_comparison"] = cos_result

    # ── 최종 요약 CSV ──
    print("\n── 최종 요약 ──")
    summary_rows = []

    if "procrustes_abuse_level" in results and results["procrustes_abuse_level"] is not None:
        r = results["procrustes_abuse_level"]
        summary_rows.append({
            "analysis": "A. Abuse-level Procrustes (CA vs BERT)",
            "metric": "Procrustes disparity",
            "value": r["procrustes_disparity"],
            "interpretation": "closer to 0 = better alignment",
        })
        summary_rows.append({
            "analysis": "A. Abuse-level Procrustes (CA vs BERT)",
            "metric": "Mantel r (distance correlation)",
            "value": r["mantel_r"],
            "p_value": r["mantel_p"],
            "interpretation": "closer to 1 = better alignment",
        })

    if "triple_comparison" in results:
        for _, row in results["triple_comparison"].iterrows():
            summary_rows.append({
                "analysis": f"C. Triple comparison: {row['pair']}",
                "metric": "Procrustes disparity",
                "value": row["disparity"],
                "interpretation": "closer to 0 = better alignment",
            })
            summary_rows.append({
                "analysis": f"C. Triple comparison: {row['pair']}",
                "metric": "Mantel r",
                "value": row["mantel_r"],
                "p_value": row.get("mantel_p", np.nan),
                "interpretation": "closer to 1 = better alignment",
            })

    if "procrustes_word_level" in results and results["procrustes_word_level"] is not None:
        r = results["procrustes_word_level"]
        summary_rows.append({
            "analysis": "D. Word-level Procrustes (CA bary vs BERT)",
            "metric": "Procrustes disparity",
            "value": r["procrustes_disparity"],
            "interpretation": "closer to 0 = better alignment",
        })
        summary_rows.append({
            "analysis": "D. Word-level Procrustes (CA bary vs BERT)",
            "metric": "Mantel r",
            "value": r["mantel_r"],
            "p_value": r["mantel_p"],
            "interpretation": "closer to 1 = better alignment",
        })

    if "cosine_comparison" in results:
        r = results["cosine_comparison"]
        summary_rows.append({
            "analysis": "E. Cosine similarity matrix correlation",
            "metric": "Pearson r",
            "value": r["pearson_r"],
            "p_value": r["pearson_p"],
            "interpretation": "closer to 1 = same relational structure",
        })
        summary_rows.append({
            "analysis": "E. Cosine similarity matrix correlation",
            "metric": "Spearman r",
            "value": r["spearman_r"],
            "p_value": r["spearman_p"],
            "interpretation": "closer to 1 = same relational structure",
        })

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        summary_path = os.path.join(out_dir, "bert_ca_validation_summary.csv")
        df_summary.to_csv(summary_path, encoding="utf-8-sig", index=False)
        print(f"\n[저장] 전체 요약 → {summary_path}")
        print("\n" + "=" * 72)
        print("[BERT-CA] 검증 결과 요약")
        print("=" * 72)
        print(df_summary.to_string(index=False))

    results["summary"] = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

    print("\n" + "=" * 72)
    print("[BERT-CA] 모든 분석 완료")
    print(f"[BERT-CA] 결과 저장: {out_dir}")
    print("=" * 72)

    return results


# ═══════════════════════════════════════════════════════════════
#  Section 10: 파이프라인 통합 가이드
# ═══════════════════════════════════════════════════════════════

INTEGRATION_GUIDE = """
# ============================================================
# BERT-CA 검증 통합 가이드 (총 3곳 수정)
# ============================================================
#
# ── 수정 1: ca.py ──────────────────────────────────────────
#
# run_abuse_ca_with_prob_bridges() 함수에서 row_coords_2d도 반환.
#
#   (A) early return 2곳 (약 175, 178번째 줄):
#       [수정 전]  return None
#       [수정 후]  return None, None
#
#   (B) 함수 마지막 return (약 588번째 줄):
#       [수정 전]  return bary_df
#       [수정 후]  return bary_df, row_coords_2d
#
# ── 수정 2: pipeline.py ─ CA 호출부 (약 679번째 줄) ────────
#
#   [수정 전]
#       bary_df = run_abuse_ca_with_prob_bridges(
#           ...
#       )
#
#   [수정 후]
#       _ca_result = run_abuse_ca_with_prob_bridges(
#           ...
#       )
#       if _ca_result is not None:
#           bary_df, row_coords_2d = _ca_result
#       else:
#           bary_df, row_coords_2d = None, None
#
# ── 수정 3: pipeline.py ─ 파일 맨 끝 (966번째 줄 이후) ────
#
    # =================================================
    # 9. [REVISION] BERT Contextual Embedding + CA 검증
    # =================================================
    from .contextual_embedding_ca import run_bert_ca_validation

    bert_ca_out = os.path.join(C.OUTPUT_DIR, "bert_ca_validation")
    os.makedirs(bert_ca_out, exist_ok=True)

    bert_results = run_bert_ca_validation(
        json_files=json_files,
        df_abuse_counts=df_abuse_counts if df_abuse_counts is not None and not df_abuse_counts.empty else None,
        abuse_stats_chi=abuse_stats_chi if abuse_stats_chi is not None and not abuse_stats_chi.empty else None,
        bary_df=bary_df,
        row_coords_2d=row_coords_2d,
        out_dir=bert_ca_out,
        model_name="klue/bert-base",
        pooling="mean",
        n_clusters=200,
        batch_size=32,
        n_perm=999,
        abuse_order=C.ABUSE_ORDER,
        allowed_groups={"부정"} if only_negative else None,
    )
"""


if __name__ == "__main__":
    print("=" * 72)
    print("BERT-CA Validation Module")
    print("=" * 72)
    print()
    print("이 모듈은 독립 실행용이 아닙니다.")
    print("파이프라인에서 import하여 사용하세요.")
    print()
    print("=== 사용 예시 ===")
    print("""
    from contextual_embedding_ca import run_bert_ca_validation

    results = run_bert_ca_validation(
        json_files=json_files,
        df_abuse_counts=df_abuse_counts,
        abuse_stats_chi=abuse_stats_chi,
        bary_df=bary_df,
        row_coords_2d=row_coords_2d,
        out_dir="revision_output/bert_ca_validation",
    )
    """)
    print()
    print("=== 파이프라인 통합 가이드 ===")
    print(INTEGRATION_GUIDE)
