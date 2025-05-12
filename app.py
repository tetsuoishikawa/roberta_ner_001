# 以下を「app.py」に書き込み
import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np

# モデルとトークナイザーのロード（保存済みパスに応じて変更）
MODEL_DIR = "./my_ner_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)

# ラベルの取得
id2label = model.config.id2label

# NER 抽出関数
def ner_predict(text):
    tokens = tokenizer.tokenize(text)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs).logits
    predictions = torch.argmax(outputs, dim=2).squeeze().tolist()

    result = []
    for token, pred_id in zip(tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze()), predictions):
        label = id2label.get(pred_id, "O")
        if label != "O":
            result.append((token.replace("▁", ""), label))
    return result

# Streamlit UI
st.title("📚 NER 抽出アプリ")
st.write("ファインチューニング済みモデルによる固有表現抽出")

input_text = st.text_area("テキストを入力してください", height=200)

if st.button("抽出実行"):
    if input_text.strip() == "":
        st.warning("テキストを入力してください。")
    else:
        results = ner_predict(input_text)
        if results:
            st.subheader("🔍 抽出結果")
            for token, label in results:
                st.write(f"**{token}** → `{label}`")
        else:
            st.info("固有表現は検出されませんでした。")
