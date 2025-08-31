
import streamlit as st
from pathlib import Path
import json
from futureself_core import respond_and_record, persona, memory, add_memory_text, load_json, save_json

st.set_page_config(page_title="FutureSelf — The Life Mirror", layout="wide")
st.title("FutureSelf — The Life Mirror ✨")
st.markdown("Personalized, psychology-infused future-self chatbot. No paid APIs required (by default).")

# load persona file if present
PERF = Path("futureself_data/persona.json")
if PERF.exists():
    persona_obj = json.loads(PERF.read_text())
else:
    persona_obj = {"name":"Alex","age":28,"core_values":["curiosity"],"short_bio":"I code."}

if "messages" not in st.session_state:
    st.session_state.messages = []

def add_message(role, text):
    st.session_state.messages.append({"role": role, "text": text})

# sidebar: persona editor
with st.sidebar.form("persona_form"):
    name = st.text_input("Name", value=persona_obj.get("name","Alex"))
    age = st.number_input("Age", min_value=13, max_value=100, value=int(persona_obj.get("age",28)))
    core_values = st.text_area("Core values (comma separated)", ", ".join(persona_obj.get("core_values",[])))
    short_bio = st.text_area("short_bio", persona_obj.get("short_bio",""))
    submitted = st.form_submit_button("Save Persona")
    if submitted:
        newp = {"name":name,"age":age,"core_values":[v.strip() for v in core_values.split(",")],"short_bio":short_bio}
        Path("futureself_data").mkdir(exist_ok=True)
        Path("futureself_data/persona.json").write_text(json.dumps(newp, indent=2))
        st.success("Persona saved. Refresh to load.")

# main chat
chat_col, right_col = st.columns([3,1])
with chat_col:
    st.subheader("Conversation")
    for m in st.session_state.messages:
        if m["role"]=="user":
            st.markdown(f"**You:** {m['text']}")
        else:
            st.markdown(f"**FutureSelf:** {m['text']}")

with st.form("chat"):
    q = st.text_area("Ask your future self...", height=140)
    remember = st.checkbox("Save to memory", value=True)
    send = st.form_submit_button("Send")
    if send and q.strip():
        add_message("user", q)
        # call core respond function
        record = respond_and_record(q, persona_obj, remember)
        add_message("bot", record["reply"])
        st.experimental_rerun()

with right_col:
    st.subheader("Quick Tools")
    if st.button("Show last saved timeline"):
        st.write("Memory size:", len(memory.get("embeddings",[])))
        st.json(memory.get("embeddings", [])[-6:])

st.markdown("---")
st.caption("Note: Running in Streamlit Cloud loads the models on first run and may take time. For faster local testing run in Colab and use the example cell to generate `futureself_last_response.json`.")
