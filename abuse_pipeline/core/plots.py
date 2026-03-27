from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # GUI(TkAgg) 비활성화: 저장 전용 백엔드
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
try:
    import squarify
except Exception:
    squarify = None

from matplotlib.lines import Line2D
from matplotlib import font_manager
from matplotlib.patches import Rectangle

from .common import *
from .text import tokenize_korean


def _force_default_font():
    # Cross-platform: prefer commonly available Korean fonts.
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/Library/Fonts/NanumGothic.ttf",
    ]
    for fp in candidates:
        if fp and os.path.exists(fp):
            font_name = fm.FontProperties(fname=fp).get_name()
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False

_force_default_font()



def plot_valence_by_question_radar(pivot_table,
                                   valence_order=VALENCE_ORDER,
                                   valence_colors=VALENCE_COLORS,
                                   title="문항별 정서군(부정·평범·긍정) 평균 문항합계",
                                   out_path=None):

    if pivot_table.empty:
        print("[RADAR] pivot_table 이 비어 있어 레이더 차트를 그리지 않습니다.")
        return

    questions = pivot_table.index.tolist()
    n_axes = len(questions)
    if n_axes == 0:
        print("[RADAR] 문항이 없어 레이더 차트를 그리지 않습니다.")
        return

    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    max_score = pivot_table.values.max()
    max_score = max(1, int(np.ceil(max_score)))

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(questions, fontsize=9)

    ax.set_ylim(0, max_score)
    yticks = np.linspace(0, max_score, 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:.0f}" for v in yticks], fontsize=8)

    legend_elems = []
    for valence in valence_order:
        if valence not in pivot_table.columns:
            continue

        vals = pivot_table[valence].tolist()
        vals += vals[:1]

        color = valence_colors.get(valence, "black")
        ax.plot(angles, vals, color=color, linewidth=2, label=valence)
        ax.fill(angles, vals, color=color, alpha=0.25)

        legend_elems.append(
            Line2D([0], [0], marker='s', linestyle='None',
                   markerfacecolor=color, label=valence, markersize=10)
        )

    ax.set_title(title, fontsize=15, pad=20)

    ax.legend(
        handles=legend_elems,
        loc="upper right",
        bbox_to_anchor=(1.25, 1.05),
        title="정서군",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="gray",
    )

    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=200)
        print(f"[저장] 문항별 레이더차트 -> {out_path}")
    else:
        plt.show()

    plt.close(fig)


def set_korean_font(font_path: str | None = None):
    """
    Ensure Korean glyphs render correctly and avoid minus-sign issues.
    """
    # 1) user provided
    candidates = []
    if font_path:
        candidates.append(font_path)
    plt.rcParams["axes.unicode_minus"] = False

    # 2) common platform fonts (Windows/macOS/Linux)
    candidates += [
        r"C:\Windows\Fonts\malgun.ttf",       # 맑은 고딕
        r"C:\Windows\Fonts\malgunsl.ttf",     # 맑은 고딕 Semilight
        r"C:\Windows\Fonts\NanumGothic.ttf",  # (있으면)
        r"C:\Windows\Fonts\batang.ttc",       # 바탕
        r"C:\Windows\Fonts\gulim.ttc",        # 굴림
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/AppleGothic.ttf",
        "/Library/Fonts/NanumGothic.ttf",
        os.path.expanduser("~/Library/Fonts/NanumGothic.ttf"),       # Homebrew
        os.path.expanduser("~/Library/Fonts/NanumGothicBold.ttf"),   # Homebrew
    ]

    chosen = None
    for fp in candidates:
        if fp and os.path.exists(fp):
            chosen = fp
            break

    if chosen:
        fm.fontManager.addfont(chosen)
        font_name = fm.FontProperties(fname=chosen).get_name()
        plt.rcParams["font.family"] = font_name
        print(f"[FONT] Korean font loaded: {font_name} ({chosen})")
    else:
        available = {f.name for f in fm.fontManager.ttflist}
        for name in ["AppleGothic", "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]:
            if name in available:
                plt.rcParams["font.family"] = name
                print(f"[FONT] Korean font from system: {name}")
                break
        else:
            # fallback: don't crash, but warn
            print("[WARN] Korean font not found. Install a Korean font or provide font_path.")

    plt.rcParams["axes.unicode_minus"] = False


def _get_cmap(cmap_name: str):
    # ✅ deprecation 대응 (matplotlib 최신/구버전 모두)
    try:
        return matplotlib.colormaps.get_cmap(cmap_name)
    except Exception:
        return plt.get_cmap(cmap_name)


def _luminance(rgba):
    r, g, b, a = rgba
    return 0.299 * r + 0.587 * g + 0.114 * b


def plot_treemap_dynamic_v2(
        freq_dict: dict,
        title: str,
        out_path: str,
        font_path: str | None = None,
        cmap_name: str = "Blues",
        max_words: int = 20,
        min_count: float = 1.0,
        color_scale: str = "log",
        label_top_n: int = 50,
        count_top_n: int = 20,
        figsize=(16, 9),
        dpi: int = 220,

        # ✅ 번역 및 표시 옵션
        lang: str = "en",  # "ko" or "en"
        translator: "TokenTranslator | None" = None,
        display_mode: str = "en_only",  # "en_only" or "en_ko"
        translate_title: bool = False,

        out_words_csv: str | None = None,  # CSV 저장 경로

        **kwargs
):
    """
    트리맵을 그리고, 실제로 포함된 단어들의 통계 정보를 CSV로 저장합니다.
    """
    if squarify is None:
        print("[TREEMAP] squarify 미설치 → 트리맵 생성을 건너뜁니다. (`pip install squarify`)")
        return False

    # --- [폰트 설정] ---
    if font_path and os.path.exists(font_path):
        try:
            prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = prop.get_name()
        except Exception as e:
            print(f"[WARN] 폰트 설정 실패: {e}")

    # 1) 데이터 준비 및 필터링
    #    - min_count 이상 필터링
    #    - 빈도순 정렬
    #    - max_words 개수만큼 자르기 (여기가 중요!)
    items = list(freq_dict.items())
    items = [x for x in items if x[1] >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:int(max_words)]

    if not items:
        print(f"[INFO] '{title}'에 대해 표시할 단어가 없습니다 (min_count 미달 등).")
        return False

    labels_ko = [k for k, _ in items]
    sizes = np.array([v for _, v in items], dtype=float)

    # ✅ 표시용 라벨 생성 (번역 적용)
    #    외부 함수 disp_token이 있다고 가정합니다. 없다면 아래 주석을 참고하여 직접 구현하거나 k를 그대로 쓰세요.
    #    labels_disp = [k for k in labels_ko]
    labels_disp = [disp_token(k, lang, translator, mode=display_mode) for k in labels_ko]

    # ------------------------------------------------------------
    # ✅ CSV 저장 로직 (트리맵에 들어가는 단어만 추출)
    # ------------------------------------------------------------
    if out_words_csv is not None:
        try:
            total_count = float(np.sum(sizes)) if len(sizes) > 0 else 0.0
            rows = []
            cumulative_percent = 0.0

            for i, ((raw_word, count), disp_word) in enumerate(zip(items, labels_disp), start=1):
                count = float(count)
                share = (count / total_count) if total_count > 0 else 0.0
                cumulative_percent += share

                rows.append({
                    "Rank": i,
                    "Raw_Word": raw_word,  # 원본 단어
                    "Display_Word": disp_word,  # 번역/가공된 단어
                    "Count": int(count),
                    "Share": round(share, 4),  # 점유율 (0~1)
                    "Share_Percent": round(share * 100, 2),  # 백분율 (%)
                    "Cumulative_Share": round(cumulative_percent, 4),
                    "Is_Label_Shown": (i <= int(label_top_n)),  # 라벨 표시 여부
                    "Is_Count_Shown": (i <= int(count_top_n))  # 숫자 표시 여부
                })

            # DataFrame 생성 및 저장
            df_words = pd.DataFrame(rows)

            # 폴더가 없으면 생성
            os.makedirs(os.path.dirname(out_words_csv), exist_ok=True)

            # 한글 깨짐 방지를 위해 utf-8-sig 사용
            df_words.to_csv(out_words_csv, encoding="utf-8-sig", index=False)
            print(f"[TREEMAP-WORDS] CSV saved -> {out_words_csv} ({len(df_words)} words)")

        except Exception as e:
            print(f"[WARN] 트리맵 단어 CSV 저장 실패: {e}")

    # 2) 캔버스 좌표 계산 (이하 플롯 로직 동일)
    fig_w, fig_h = figsize
    aspect_ratio = fig_w / fig_h
    coord_h = 100.0
    coord_w = 100.0 * aspect_ratio
    total_coord_area = coord_w * coord_h

    # 3) Squarify 계산
    sizes_norm = squarify.normalize_sizes(sizes, coord_w, coord_h)
    rects = squarify.squarify(sizes_norm, 0, 0, coord_w, coord_h)

    # 4) 색상 매핑
    #    log, sqrt, linear 등 스케일 옵션 적용
    if color_scale == "log":
        cvals = np.log1p(sizes)
    elif color_scale == "sqrt":
        cvals = np.sqrt(sizes)
    else:
        cvals = sizes.copy()

    base_cmap = plt.get_cmap(cmap_name)
    # 너무 연한 색은 제외하고 0.25~1.0 구간 사용
    cmap_colors = base_cmap(np.linspace(0.25, 1.0, 256))
    new_cmap = mcolors.LinearSegmentedColormap.from_list("truncated_cmap", cmap_colors)
    norm = mcolors.Normalize(vmin=float(np.min(cvals)), vmax=float(np.max(cvals)) + 1e-9)

    # 5) 그리기
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_axis_off()
    ax.set_xlim(0, coord_w)
    ax.set_ylim(0, coord_h)

    # 제목 번역 처리
    t_title = title
    if lang == "en" and translate_title and translator is not None:
        try:
            t_title = translator.translate(title)
        except:
            pass  # 번역 실패 시 원본 사용

    ax.set_title(f"{t_title}", fontsize=40, pad=40, fontweight='bold')

    for i, r in enumerate(rects):
        x, y, dx, dy = r["x"], r["y"], r["dx"], r["dy"]
        rgba = new_cmap(norm(cvals[i]))

        # 배경색 밝기에 따라 글자색 결정 (흰색/검은색)
        lum = (0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2])
        txt_color = "black" if lum > 0.6 else "white"

        # 사각형 그리기
        ax.add_patch(Rectangle((x, y), dx, dy, facecolor=rgba, edgecolor="white", linewidth=2.0))

        # 너무 작은 사각형은 텍스트 생략
        if (dx * dy) / total_coord_area < 0.003:
            continue
        # 라벨 표시 개수 제한
        if i >= int(label_top_n):
            continue

        disp_word = labels_disp[i]
        cnt = int(round(sizes[i]))

        # 상위 N개만 숫자(빈도) 표시
        display_text = f"{disp_word}\n({cnt:,})" if (i < int(count_top_n)) else disp_word

        # 폰트 크기 자동 조절
        min_side = min(dx, dy)
        fs = max(18, min(100, min_side * 0.7))  # 최소 18, 최대 100

        stroke_color = "white" if txt_color == "black" else "black"

        t = ax.text(
            x + dx / 2, y + dy / 2, display_text,
            ha="center", va="center",
            fontsize=fs,
            color=txt_color,
            fontweight='bold',
            linespacing=1.2
        )
        # 글자 테두리 효과
        t.set_path_effects([pe.withStroke(linewidth=fs / 8, foreground=stroke_color)])

    # 이미지 저장
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    # 번역 캐시 저장
    if translator is not None:
        translator.save()

    return True


def plot_treemap_from_counts(
    word_to_count: dict,
    title: str,
    out_path: str,
    *,
    max_words: int = 40,   # ✅ 기본 40
    min_count: int = 1,
    figsize=(10, 7),
    cmap: str = "tab20",
    show_counts: bool = True,
):
    """
    word_to_count: {"단어": 빈도, ...}  (빈도는 int/float 가능)
    title: 그림 제목
    out_path: 저장 경로 (.png 권장)

    ✅ 동작:
      - min_count 미만 제거
      - 빈도 기준 내림차순 정렬
      - 상위 max_words개만 표시
      - 그 이하는 '기타'로 묶지 않고 완전히 제외
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if squarify is None:
        print("[TREEMAP] squarify 미설치 → 트리맵 생성을 건너뜁니다. (`pip install squarify`)")
        return

    # --- 1) 정리: 유효한 (word, count)만 ---
    items = []
    for w, c in (word_to_count or {}).items():
        if not isinstance(w, str):
            continue
        w = w.strip()
        if not w:
            continue
        try:
            v = float(c)
        except Exception:
            continue
        if np.isfinite(v) and v >= float(min_count):
            items.append((w, v))

    if not items:
        print(f"[TREEMAP] 비어 있음: {title} (min_count={min_count})")
        return

    # --- 2) 정렬 후 상위 N개만 남기고 나머지는 버림(기타 없음) ---
    items.sort(key=lambda x: x[1], reverse=True)
    if max_words is not None and int(max_words) > 0:
        items = items[: int(max_words)]
    else:
        # max_words<=0이면 아무것도 그리지 않게 처리
        print(f"[TREEMAP] max_words가 비정상: {max_words}")
        return

    # --- 3) 라벨/사이즈 ---
    labels, sizes = [], []
    for w, v in items:
        sizes.append(v)
        if show_counts:
            vv = int(round(v))
            labels.append(f"{w}\n{vv:,}")  # ✅ 천단위 콤마
        else:
            labels.append(w)

    sizes = np.asarray(sizes, dtype=float)

    # --- 4) 색상 ---
    cm = plt.get_cmap(cmap)
    colors = [cm(i) for i in np.linspace(0, 1, len(sizes))]

    # --- 5) 플롯 ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    squarify.plot(
        sizes=sizes,
        label=labels,
        color=colors,
        alpha=0.92,
        ax=ax,
        pad=True,
        text_kwargs={"fontsize": 20},
    )

    ax.set_title(f"{title} (Top {len(items)})", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[SAVE] treemap -> {out_path}")


def treemap_from_df_counts(
    df_counts,
    col_name: str,
    title: str,
    out_path: str,
    *,
    max_words: int = 80,
    min_count: int = 1,
):
    """
    df_counts: index=word, columns=[그룹들...] 형태의 카운트 테이블에서
              특정 col_name(그룹) 기준 treemap 저장
    """
    if df_counts is None or df_counts.empty or col_name not in df_counts.columns:
        print(f"[TREEMAP] df_counts 비었거나 col 없음: {col_name}")
        return

    s = df_counts[col_name].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s >= float(min_count)].sort_values(ascending=False).head(max_words)
    plot_treemap_from_counts(
        word_to_count=s.to_dict(),
        title=title,
        out_path=out_path,
        max_words=max_words,
        min_count=min_count,
    )


def run_tfidf_multilogit_no_leak(
    df_text, label_col, label_order, out_dir, label_name="valence"
):
    """
    Leakage-free TF-IDF + Multinomial Logistic Regression with Stratified CV.

    df_text: DataFrame with columns ["ID", label_col, "tfidf_text"]
    label_col: "group" (정서군) or "main_abuse" (학대유형)
    label_order: fixed label order list (e.g., VALENCE_ORDER, ABUSE_ORDER)
    out_dir: output directory
    label_name: tag for filenames (e.g., "valence_ALL", "abuse_NEG_ONLY")
    """
    if not HAS_SKLEARN:
        print(f"[TFIDF-LOGIT] scikit-learn 미설치 → {label_name} 분석은 건너뜁니다.")
        return

    import os
    import numpy as np
    import pandas as pd

    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        cohen_kappa_score,
    )

    os.makedirs(out_dir, exist_ok=True)

    df = df_text.copy()
    if "tfidf_text" not in df.columns:
        print(f"[TFIDF-LOGIT] df_text에 'tfidf_text' 컬럼이 없습니다. ({label_name})")
        return

    # 1) Clean: drop missing + filter labels
    df = df.dropna(subset=[label_col, "tfidf_text"])
    df = df[df[label_col].isin(label_order)]

    if df.empty or df[label_col].nunique() < 2:
        print(f"[TFIDF-LOGIT] {label_name}: 유효한 레이블이 2개 미만이라 분석을 건너뜁니다.")
        return

    texts = df["tfidf_text"].astype(str).tolist()
    y = df[label_col].astype(str).values

    # 2) Determine n_splits safely
    value_counts = df[label_col].value_counts()
    min_count = int(value_counts.min())
    if min_count < 2:
        print(f"[TFIDF-LOGIT] {label_name}: 어떤 레이블은 샘플 수가 2 미만이라 분석을 건너뜁니다.")
        return

    n_splits = min(5, min_count)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 3) Leakage-free pipeline: TF-IDF is fit ONLY on training folds
    pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=str.split,     # assumes whitespace-tokenized text
                    preprocessor=None,
                    token_pattern=None,      # MUST be None when using custom tokenizer
                    ngram_range=(1, 2),      # unigram + bigram
                    min_df=2,
                    max_features=20000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=300,
                ),
            ),
        ]
    )

    all_true, all_pred = [], []
    fold_rows = []

    print(f"[TFIDF-LOGIT] {label_name}: n={len(df)}, 클래스={df[label_col].nunique()}, n_splits={n_splits}")

    # 4) CV loop (fit on train fold only)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        X_train = [texts[i] for i in train_idx]
        y_train = y[train_idx]
        X_test = [texts[i] for i in test_idx]
        y_test = y[test_idx]

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        acc = float((pred == y_test).mean())
        kappa = cohen_kappa_score(y_test, pred, labels=label_order)

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": acc,
                "cohen_kappa": kappa,
            }
        )

        all_true.extend(list(y_test))
        all_pred.extend(list(pred))

    # 5) Fold summary save
    fold_df = pd.DataFrame(fold_rows)
    fold_path = os.path.join(out_dir, f"tfidf_logit_{label_name}_fold_summary.csv")
    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    print(f"[TFIDF-LOGIT] fold summary 저장 -> {fold_path}")

    # 6) Overall report / confusion matrix
    overall_kappa = cohen_kappa_score(all_true, all_pred, labels=label_order)
    print(f"[TFIDF-LOGIT] {label_name} 전체 Cohen's kappa = {overall_kappa:.4f}")

    report_dict = classification_report(
        all_true,
        all_pred,
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    report_path = os.path.join(out_dir, f"tfidf_logit_{label_name}_classification_report.csv")
    report_df.to_csv(report_path, encoding="utf-8-sig")
    print(f"[TFIDF-LOGIT] classification report 저장 -> {report_path}")

    cm = confusion_matrix(all_true, all_pred, labels=label_order)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{l}" for l in label_order],
        columns=[f"pred_{l}" for l in label_order],
    )
    cm_path = os.path.join(out_dir, f"tfidf_logit_{label_name}_confusion_matrix.csv")
    cm_df.to_csv(cm_path, encoding="utf-8-sig")
    print(f"[TFIDF-LOGIT] confusion matrix 저장 -> {cm_path}")

    # 7) Fit final model on ALL data for interpretability (top features)
    pipe_final = Pipeline(steps=pipe.steps)  # clone-ish (simple rebuild)
    pipe_final.fit(texts, y)

    vectorizer = pipe_final.named_steps["tfidf"]
    clf_final = pipe_final.named_steps["clf"]

    feature_names = np.array(vectorizer.get_feature_names_out())

    # Align coef rows with label_order if you want stable ordering
    class_to_row = {c: i for i, c in enumerate(clf_final.classes_)}

    rows_coef = []
    for cls_label in label_order:
        if cls_label not in class_to_row:
            continue
        cls_idx = class_to_row[cls_label]
        coef = clf_final.coef_[cls_idx]
        top_idx = np.argsort(-coef)[:200]
        for rank, f_idx in enumerate(top_idx, start=1):
            rows_coef.append(
                {
                    "class": cls_label,
                    "rank": rank,
                    "feature": feature_names[f_idx],
                    "coef": float(coef[f_idx]),
                }
            )

    coef_df = pd.DataFrame(rows_coef)
    coef_path = os.path.join(out_dir, f"tfidf_logit_{label_name}_top_features.csv")
    coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")
    print(f"[TFIDF-LOGIT] class별 상위 feature 계수 저장 -> {coef_path}")

    return {
        "fold_summary": fold_df,
        "overall_kappa": overall_kappa,
        "classification_report": report_df,
        "confusion_matrix": cm_df,
        "top_features": coef_df,
    }
