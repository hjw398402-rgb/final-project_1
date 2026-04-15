import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
MODEL_PATH = "C:/workspace/finalproject/best_model_v4"


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
# 추론
# ─────────────────────────────────────────
def predict(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze(0)
    hate_score = probs[1].item()  # 1 = 혐오
    clean_score = probs[0].item()  # 0 = 정상
    return hate_score, clean_score


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.set_page_config(page_title="혐오표현 탐지기", page_icon="🔍", layout="centered")
st.title("🔍 한국어 혐오표현 탐지기")
st.caption("KLUE-BERT 기반 혐오표현 탐지 모델")
st.divider()

# 모델 로드
with st.spinner("모델 불러오는 중..."):
    try:
        tokenizer, model = load_model()
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        st.stop()

# 임계값 슬라이더
threshold = st.slider(
    "탐지 임계값",
    min_value=0.1, max_value=0.9,
    value=0.5, step=0.05,
    help="이 값 이상이면 혐오표현으로 판정"
)

# 입력창
text_input = st.text_area(
    "댓글을 입력하세요",
    placeholder="예) 역시 걔네는 다 그렇지 ㅋㅋ",
    height=120
)

analyze_btn = st.button("분석하기", type="primary", use_container_width=True)

if analyze_btn:
    if not text_input.strip():
        st.warning("댓글을 입력해 주세요.")
    else:
        with st.spinner("분석 중..."):
            hate_score, clean_score = predict(text_input.strip(), tokenizer, model)

        st.divider()

        # 판정 배너
        if hate_score >= threshold:
            st.error("## 🚨 혐오 표현 탐지 → 블라인드 처리 대상")
        else:
            st.success("## ✅ 정상 댓글")

        # 확률 표시
        st.subheader("📊 분석 결과")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("혐오 확률", f"{hate_score * 100:.1f}%")
        with col2:
            st.metric("정상 확률", f"{clean_score * 100:.1f}%")

        # 확률 바
        st.markdown("**혐오**")
        st.markdown(
            f"""<div style="background:#eee;border-radius:6px;height:26px;overflow:hidden;">
                <div style="width:{hate_score * 100:.1f}%;background:#FF6B6B;height:100%;
                            border-radius:6px;display:flex;align-items:center;
                            padding-left:8px;color:white;font-size:13px;font-weight:bold;">
                    {hate_score * 100:.1f}%
                </div></div>""",
            unsafe_allow_html=True
        )
        st.markdown("**정상**")
        st.markdown(
            f"""<div style="background:#eee;border-radius:6px;height:26px;overflow:hidden;">
                <div style="width:{clean_score * 100:.1f}%;background:#51CF66;height:100%;
                            border-radius:6px;display:flex;align-items:center;
                            padding-left:8px;color:white;font-size:13px;font-weight:bold;">
                    {clean_score * 100:.1f}%
                </div></div>""",
            unsafe_allow_html=True
        )

st.divider()

# 예시 버튼
st.subheader("예시 문장 테스트")
examples = [
    "오늘 날씨 진짜 좋다",
    "역시 수발 같은 새끼들",
    "ㅅㅂ 왜이래 진짜",
    "저 나라 사람들은 믿으면 안 돼",
    "틀딱들은 왜 저러냐 진짜",
    "감사합니다 도움이 됐어요",
]
cols = st.columns(2)
for i, ex in enumerate(examples):
    if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
        hate_score, clean_score = predict(ex, tokenizer, model)
        st.info(f"**입력:** {ex}")
        if hate_score >= threshold:
            st.error(f"🚨 혐오 탐지 | 혐오 확률: {hate_score * 100:.1f}%")
        else:
            st.success(f"✅ 정상 | 정상 확률: {clean_score * 100:.1f}%")
