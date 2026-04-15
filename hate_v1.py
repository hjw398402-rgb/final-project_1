import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
import json

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
MODEL_PATH = "C:/workspace/finalproject/results/checkpoint-5628"  # 본인 체크포인트 경로
BERT_THRESHOLD_HIGH = 0.7  # 이 이상이면 BERT 결과만으로 혐오 확정
BERT_THRESHOLD_LOW = 0.3  # 이 이하면 BERT 결과만으로 정상 확정
# 0.3 ~ 0.7 사이 → LLM 2차 검토

LABELS = [
    "여성/가족", "남성", "성소수자", "인종/국적",
    "연령", "지역", "종교", "기타 혐오", "clean"
]

LABEL_COLORS = {
    "여성/가족": "#FF6B6B",
    "남성": "#FFA94D",
    "성소수자": "#CC5DE8",
    "인종/국적": "#FF8787",
    "연령": "#F06595",
    "지역": "#E64980",
    "종교": "#BE4BDB",
    "기타 혐오": "#F03E3E",
    "clean": "#51CF66",
}


# ─────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


# ─────────────────────────────────────────
# BERT 추론
# ─────────────────────────────────────────
def bert_predict(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits).squeeze(0)
    return [{"label": label, "score": round(prob, 4)} for label, prob in zip(LABELS, probs.tolist())]


# ─────────────────────────────────────────
# LLM 2차 검토
# ─────────────────────────────────────────
def llm_check(text, api_key):
    client = OpenAI(api_key=api_key)
    prompt = f"""너는 한국어 혐오표현 탐지 전문가야.
아래 댓글이 혐오표현인지 판단해줘.

판단 기준:
- 직접적인 욕설뿐 아니라 우회표현, 초성, 은어, 맥락적 혐오도 포함
- 예: "수발", "ㅅㅂ", "틀딱", "한남", "김치녀" 등
- 특정 집단을 비하하거나 차별하는 표현 포함

댓글: "{text}"

반드시 아래 JSON 형식으로만 답해. 다른 말 하지마:
{{
  "is_hate": true 또는 false,
  "reason": "판단 이유 한 줄",
  "categories": ["해당 카테고리들"] 
}}

categories 값은 이 중에서만 선택: 여성/가족, 남성, 성소수자, 인종/국적, 연령, 지역, 종교, 기타 혐오, clean"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ─────────────────────────────────────────
# 최종 판정 로직
# ─────────────────────────────────────────
def analyze(text, tokenizer, model, api_key, use_llm):
    bert_results = bert_predict(text, tokenizer, model)

    # 혐오 카테고리만 (clean 제외)
    hate_scores = [r for r in bert_results if r["label"] != "clean"]
    max_score = max(r["score"] for r in hate_scores)

    llm_result = None
    method = ""

    if not use_llm:
        # LLM 미사용: BERT 결과만
        is_hate = max_score >= 0.5
        method = "BERT 단독 판정"
    elif max_score >= BERT_THRESHOLD_HIGH:
        # 명확한 혐오 → BERT만으로 확정
        is_hate = True
        method = "BERT 확정 (고신뢰)"
    elif max_score <= BERT_THRESHOLD_LOW:
        # 명확한 정상 → BERT만으로 확정
        is_hate = False
        method = "BERT 확정 (저신뢰)"
    else:
        # 애매한 구간 → LLM 호출
        try:
            llm_result = llm_check(text, api_key)
            is_hate = llm_result["is_hate"]
            method = "LLM 2차 검토"
        except Exception as e:
            is_hate = max_score >= 0.5
            method = f"LLM 오류 → BERT 대체 ({e})"

    return {
        "is_hate": is_hate,
        "method": method,
        "bert_results": bert_results,
        "llm_result": llm_result,
        "max_score": max_score
    }


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.set_page_config(page_title="혐오표현 탐지기", page_icon="🔍", layout="centered")
st.title("🔍 한국어 혐오표현 탐지기")
st.caption("KLUE-BERT + GPT-4o-mini 2차 검토")
st.divider()

# 모델 로드
with st.spinner("모델 불러오는 중..."):
    try:
        tokenizer, model = load_model()
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        st.stop()

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    use_llm = st.toggle("LLM 2차 검토 사용", value=True)

    if use_llm and not api_key:
        st.warning("API Key를 입력해야 LLM 검토가 활성화됩니다.")
        use_llm = False

    st.divider()
    st.markdown(f"""
    **판정 기준**
    - 🔴 BERT > `{BERT_THRESHOLD_HIGH}` → 혐오 확정
    - 🟡 `{BERT_THRESHOLD_LOW}` ~ `{BERT_THRESHOLD_HIGH}` → LLM 검토
    - 🟢 BERT < `{BERT_THRESHOLD_LOW}` → 정상 확정
    """)

# 입력창
text_input = st.text_area(
    "댓글을 입력하세요",
    placeholder="예) 역시 걔네는 다 그렇지 ㅋㅋ",
    height=120
)
analyze_btn = st.button("분석하기", type="primary", use_container_width=True)

# 분석 실행
if analyze_btn:
    if not text_input.strip():
        st.warning("댓글을 입력해 주세요.")
    else:
        with st.spinner("분석 중..."):
            result = analyze(text_input.strip(), tokenizer, model, api_key, use_llm)

        st.divider()

        # 판정 배너
        if result["is_hate"]:
            st.error("## 🚨 혐오 표현 탐지 → 블라인드 처리 대상")
        else:
            st.success("## ✅ 정상 댓글")

        st.caption(f"판정 방식: **{result['method']}**")

        # LLM 결과 표시
        if result["llm_result"]:
            llm = result["llm_result"]
            with st.expander("💬 LLM 판단 근거 보기"):
                st.write(f"**판단:** {'혐오' if llm['is_hate'] else '정상'}")
                st.write(f"**이유:** {llm['reason']}")
                st.write(f"**카테고리:** {', '.join(llm['categories'])}")

        st.divider()

        # BERT 카테고리별 확률
        st.subheader("📊 BERT 카테고리별 확률")
        for r in result["bert_results"]:
            col1, col2 = st.columns([2, 5])
            color = LABEL_COLORS.get(r["label"], "#aaa")
            score_pct = r["score"] * 100
            with col1:
                st.write(f"**{r['label']}**")
            with col2:
                st.markdown(
                    f"""<div style="background:#eee;border-radius:6px;height:22px;overflow:hidden;">
                        <div style="width:{score_pct:.1f}%;background:{color};height:100%;
                                    border-radius:6px;display:flex;align-items:center;
                                    padding-left:6px;color:white;font-size:12px;font-weight:bold;">
                            {score_pct:.1f}%
                        </div></div>""",
                    unsafe_allow_html=True
                )

st.divider()

# 예시 버튼
st.subheader("예시 문장 테스트")
examples = [
    "오늘 날씨 진짜 좋다",
    "역시 수발 같은 새끼들",
    "걔네는 다 그렇지 ㅋㅋ",
    "저 나라 사람들은 믿으면 안 돼",
    "틀딱들은 왜 저러냐 진짜",
    "여자들은 감정적이라서 리더 못해",
]
cols = st.columns(2)
for i, ex in enumerate(examples):
    if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
        with st.spinner("분석 중..."):
            result = analyze(ex, tokenizer, model, api_key, use_llm)
        st.info(f"**입력:** {ex}")
        if result["is_hate"]:
            st.error(f"🚨 혐오 탐지 | {result['method']}")
            if result["llm_result"]:
                st.caption(f"LLM 이유: {result['llm_result']['reason']}")
        else:
            st.success(f"✅ 정상 | {result['method']}")