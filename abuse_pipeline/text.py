from __future__ import annotations

from .common import *
from .labels import classify_child_group,classify_abuse_main_sub

def extract_child_speech(rec):
    texts = []
    for q in rec.get("list", []):
        for it in q.get("list", []):
            for seg in it.get("audio", []):
                if seg.get("type") == "A":
                    t = seg.get("text")
                    if isinstance(t, str) and t.strip():
                        texts.append(t.strip())
    return texts


def tokenize_korean(text: str):
    """
    한국어 텍스트를 토큰 리스트로 변환.
    - 한글/숫자/공백만 남김
    - Okt 사용 시: 명사/동사/형용사만 사용 (stem 기준)
    - 전역 불용어 + 한 글자 토큰 제거
    - 부정어 처리 (안/못/않다) bigram
    """
    if not isinstance(text, str):
        return []

    text = re.sub(r"[^가-힣0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    stopwords = STOPWORDS_BASE

    if HAS_OKT and okt is not None:
        pos_list = okt.pos(text, stem=True)
        tokens = []
        i = 0
        while i < len(pos_list):
            word, pos = pos_list[i]

            if word in {"안", "못"} and (i + 1) < len(pos_list):
                next_word, next_pos = pos_list[i + 1]
                if next_pos in ["Verb", "Adjective"]:
                    base = next_word
                    if len(base) > 1 and base not in stopwords:
                        combined = f"{word}_{base}"
                        tokens.append(combined)
                    i += 2
                    continue

            if word == "않다":
                if tokens:
                    last = tokens[-1]
                    if not last.startswith("안_"):
                        tokens[-1] = "안_" + last
                i += 1
                continue

            if pos in ["Noun", "Verb", "Adjective"]:
                if len(word) > 1 and word not in stopwords:
                    tokens.append(word)

            i += 1

        return tokens

    else:
        toks = text.split()
        toks = [t for t in toks if len(t) > 1 and t not in stopwords]
        return toks


def normalize_text_for_example(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = re.sub(r"[^가-힣0-9\s]", " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def save_tokenization_examples(json_files, out_path, n_examples=10):
    rows = []
    cnt = 0

    for path in json_files:
        if cnt >= n_examples:
            break
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue

        info = rec.get("info", {})
        child_id = info.get("ID")

        for q in rec.get("list", []):
            for it in q.get("list", []):
                for seg in it.get("audio", []):
                    if seg.get("type") != "A":
                        continue
                    raw = seg.get("text")
                    if not isinstance(raw, str) or not raw.strip():
                        continue

                    norm = normalize_text_for_example(raw)
                    tokens = tokenize_korean(raw)
                    rows.append({
                        "ID": child_id,
                        "raw_text": raw,
                        "normalized_text": norm,
                        "tokens": " ".join(tokens),
                    })
                    cnt += 1
                    if cnt >= n_examples:
                        break
                if cnt >= n_examples:
                    break
            if cnt >= n_examples:
                break

    if not rows:
        print("[INFO] 전처리 예시를 만들 데이터가 없습니다.")
        return

    df_ex = pd.DataFrame(rows)
    df_ex.to_csv(out_path, encoding="utf-8-sig", index=False)
    print(f"[저장] 전처리 before/after 예시 -> {out_path}")


def extract_bridge_utterances_p_z(json_files, bridge_df, out_path,
                                  allowed_groups=None):
    """
    p + z 조건(p_z config)을 모두 통과한 bridge 단어가
    실제로 등장하는 아동 발화(seg['type']=='A')를 추출해서 CSV로 저장하고,
    그 결과를 이용해 추가 통계도 생성한다.

    1) utterance-level CSV (기존)
       - ID, group(정서군), main_abuse_child, bridge_word, 발화(raw/tokens) 등

    2) (NEW) bridge_word 단어 수준 분포 CSV
       - 파일: bridge_prob_p_z_word_distribution_childtokens.csv
       - 각 bridge_word에 대해
         * n_utterances : 해당 단어가 포함된 발화 수
         * n_children   : 해당 단어를 말한 아동 수
         * total_occurrences : 발화 내에서의 총 등장 횟수(토큰 단위)
         * 정서군별 발화 수 (n_utt_group_부정, ...)
         * 학대유형별 발화 수 (n_utt_mainAbuse_방임, ...)

    3) (NEW) co-occurrence CSV
       - 파일: bridge_prob_p_z_cooccurrence_childtokens.csv
       - (bridge_word, co_word, n_utter_cooccur)
       - 한 발화 안에서 bridge_word와 co_word가 함께 나타난
         발화의 개수를 의미(발화 단위 co-occurrence)

    4) (NEW) bigram 패턴 CSV
       - 파일: bridge_prob_p_z_bigram_patterns_childtokens.csv
       - bridge_word 주변 좌/우 이웃 단어 기준 bigram 빈도
       - 컬럼:
         * bridge_word
         * neighbor_word
         * position : "prev" (이웃이 왼쪽) / "next" (이웃이 오른쪽)
         * bigram   : "이웃 bridge_word" 또는 "bridge_word 이웃"
         * count    : 해당 bigram이 나타난 횟수
    """
    if bridge_df is None or bridge_df.empty:
        print("[BRIDGE-UTT] p+z bridge 단어가 없어 발화 추출을 건너뜁니다.")
        return

    # word 기준으로 인덱싱
    bridge_df = bridge_df.set_index("word")
    bridge_words = set(bridge_df.index)

    if not bridge_words:
        print("[BRIDGE-UTT] bridge_df에 word가 없습니다.")
        return

    if allowed_groups is not None and not isinstance(allowed_groups, set):
        allowed_groups = set(allowed_groups)

    rows = []
    n_json = 0
    n_utt = 0
    n_match = 0

    for path in json_files:
        n_json += 1
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception as e:
            print(f"[BRIDGE-UTT] JSON 로드 실패: {path} ({e})")
            continue

        info = rec.get("info", {})
        child_id = info.get("ID")

        # 메타 정보
        group = classify_child_group(rec)  # 정서군 (부정/평범/긍정/None)
        if allowed_groups is not None and group not in allowed_groups:
            continue

        main_abuse, sub_abuses = classify_abuse_main_sub(rec)

        # bridge 정의 자체가 main_abuse 기반이라, main_abuse 없는 경우는 스킵
        if main_abuse not in ABUSE_ORDER:
            continue

        for q in rec.get("list", []):
            qname = q.get("문항")
            for it in q.get("list", []):
                iname = it.get("항목")
                for seg in it.get("audio", []):
                    if seg.get("type") != "A":
                        continue
                    raw = seg.get("text")
                    if not isinstance(raw, str) or not raw.strip():
                        continue

                    n_utt += 1
                    tokens = tokenize_korean(raw)
                    if not tokens:
                        continue

                    token_set = set(tokens)
                    matched = sorted(token_set & bridge_words)
                    if not matched:
                        continue

                    n_match += 1

                    # 각 bridge_word마다 row 하나씩 생성
                    for w in matched:
                        # bridge_df 에서 p, gap, primary/secondary 가져오기
                        try:
                            br = bridge_df.loc[w]
                        except KeyError:
                            continue

                        # 이 발화에서 해당 bridge_word가 등장한 token 횟수
                        cnt_in_utt = sum(1 for t in tokens if t == w)

                        rows.append({
                            "ID": child_id,
                            "group": group,
                            "main_abuse_child": main_abuse,
                            "json_file": os.path.basename(path),
                            "question": qname,
                            "item": iname,
                            "bridge_word": w,
                            "primary_abuse_bridge": br.get("primary_abuse"),
                            "secondary_abuse_bridge": br.get("secondary_abuse"),
                            "p1": br.get("p1"),
                            "p2": br.get("p2"),
                            "gap": br.get("gap"),
                            "utterance_raw": raw,
                            "utterance_tokens": " ".join(tokens),
                            "count_in_utterance": cnt_in_utt,
                        })

    if not rows:
        print("[BRIDGE-UTT] p+z bridge 단어가 포함된 발화 A 를 찾지 못했습니다.")
        return

    df_utt = pd.DataFrame(rows)
    df_utt.to_csv(out_path, encoding="utf-8-sig", index=False)

    print(
        f"[저장] p+z bridge 단어 실제 발화 A -> {out_path} "
        f"(JSON {n_json}개, 발화 {n_utt}개 중 match {n_match}개)"
    )

    # -------------------------------
    # (NEW #1) bridge_word 단어 수준 분포
    # -------------------------------
    word_stats_rows = []
    for w, sub in df_utt.groupby("bridge_word"):
        n_utt_w = len(sub)                 # 이 단어가 들어간 발화 수
        n_child_w = sub["ID"].nunique()    # 이 단어를 말한 아동 수
        total_occ_w = int(sub["count_in_utterance"].sum())

        row = {
            "bridge_word": w,
            "n_utterances": n_utt_w,
            "n_children": n_child_w,
            "total_occurrences": total_occ_w,
        }

        # 정서군별 발화 수
        for g in VALENCE_ORDER:
            col_name = f"n_utt_group_{g}"
            row[col_name] = int((sub["group"] == g).sum())

        # main_abuse(아동 기준)별 발화 수
        for a in ABUSE_ORDER:
            col_name = f"n_utt_mainAbuse_{a}"
            row[col_name] = int((sub["main_abuse_child"] == a).sum())

        word_stats_rows.append(row)

    df_word_stats = pd.DataFrame(word_stats_rows)
    word_dist_path = os.path.join(
        os.path.dirname(out_path),
        "bridge_prob_p_z_word_distribution_childtokens.csv",
    )
    df_word_stats.sort_values("n_utterances", ascending=False, inplace=True)
    df_word_stats.to_csv(word_dist_path, encoding="utf-8-sig", index=False)
    print(f"[저장] p+z bridge 단어 수준 분포 -> {word_dist_path}")

    # -------------------------------
    # (NEW #2) co-occurrence 단어 (발화 단위)
    # -------------------------------
    co_counts = {}

    for _, r in df_utt.iterrows():
        w = r["bridge_word"]
        tokens = r["utterance_tokens"].split()
        token_set = set(tokens)
        # 동일 발화에서 함께 등장하는 단어들 (발화 단위 co-occurrence)
        for t in token_set:
            if t == w:
                continue
            key = (w, t)
            co_counts[key] = co_counts.get(key, 0) + 1

    co_rows = []
    for (w, t), cnt in co_counts.items():
        co_rows.append({
            "bridge_word": w,
            "co_word": t,
            "n_utter_cooccur": int(cnt),
        })

    df_co = pd.DataFrame(co_rows)
    if not df_co.empty:
        df_co.sort_values(
            ["bridge_word", "n_utter_cooccur", "co_word"],
            ascending=[True, False, True],
            inplace=True,
        )
        co_path = os.path.join(
            os.path.dirname(out_path),
            "bridge_prob_p_z_cooccurrence_childtokens.csv",
        )
        df_co.to_csv(co_path, encoding="utf-8-sig", index=False)
        print(f"[저장] p+z bridge 단어 co-occurrence -> {co_path}")
    else:
        print("[BRIDGE-UTT] co-occurrence 결과가 비어 있습니다.")

    # -------------------------------
    # (NEW #3) bigram 패턴 (좌/우 이웃)
    # -------------------------------
    bigram_prev_counts = {}  # (bridge_word, prev_word) -> count
    bigram_next_counts = {}  # (bridge_word, next_word) -> count

    for _, r in df_utt.iterrows():
        w = r["bridge_word"]
        tokens = r["utterance_tokens"].split()
        n_tok = len(tokens)
        for i, tok in enumerate(tokens):
            if tok != w:
                continue
            # 왼쪽 이웃
            if i > 0:
                prev_word = tokens[i - 1]
                key_prev = (w, prev_word)
                bigram_prev_counts[key_prev] = bigram_prev_counts.get(key_prev, 0) + 1
            # 오른쪽 이웃
            if i < n_tok - 1:
                next_word = tokens[i + 1]
                key_next = (w, next_word)
                bigram_next_counts[key_next] = bigram_next_counts.get(key_next, 0) + 1

    bigram_rows = []

    for (w, prev), cnt in bigram_prev_counts.items():
        bigram_rows.append({
            "bridge_word": w,
            "neighbor_word": prev,
            "position": "prev",
            "bigram": f"{prev} {w}",
            "count": int(cnt),
        })

    for (w, nxt), cnt in bigram_next_counts.items():
        bigram_rows.append({
            "bridge_word": w,
            "neighbor_word": nxt,
            "position": "next",
            "bigram": f"{w} {nxt}",
            "count": int(cnt),
        })

    df_bigram = pd.DataFrame(bigram_rows)
    if not df_bigram.empty:
        df_bigram.sort_values(
            ["bridge_word", "count", "position", "neighbor_word"],
            ascending=[True, False, True, True],
            inplace=True,
        )
        bigram_path = os.path.join(
            os.path.dirname(out_path),
            "bridge_prob_p_z_bigram_patterns_childtokens.csv",
        )
        df_bigram.to_csv(bigram_path, encoding="utf-8-sig", index=False)
        print(f"[저장] p+z bridge 단어 bigram 패턴 -> {bigram_path}")
    else:
        print("[BRIDGE-UTT] bigram 패턴 결과가 비어 있습니다.")
