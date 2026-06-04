import json
import time
import httpx
import streamlit as st

BACKEND_URL = "http://localhost:8000"
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

BLUE  = "#002395"
RED   = "#ED2939"
WHITE = "#FFFFFF"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MrWorldwide",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
    }}

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header {{ visibility: hidden; }}

    /* App background */
    .stApp {{ background-color: {WHITE}; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: #f4f6fb;
        border-right: 1px solid #dde3f0;
    }}
    [data-testid="stSidebar"] * {{ font-family: 'Inter', sans-serif !important; }}

    /* Top tricolor stripe */
    .tricolor {{
        display: flex;
        height: 5px;
        width: 100%;
        margin-bottom: 20px;
        border-radius: 3px;
        overflow: hidden;
    }}
    .tc-blue  {{ flex: 1; background: {BLUE}; }}
    .tc-white {{ flex: 1; background: #cdd6f0; }}
    .tc-red   {{ flex: 1; background: {RED}; }}

    /* Login card */
    .login-wrap {{
        display: flex;
        flex-direction: column;
        align-items: center;
        padding-top: 60px;
    }}
    .login-card {{
        background: white;
        border-radius: 16px;
        box-shadow: 0 4px 32px rgba(0,35,149,.10);
        padding: 40px 44px 36px;
        width: 360px;
        text-align: center;
    }}
    .login-flag {{
        display: flex;
        height: 6px;
        border-radius: 3px;
        overflow: hidden;
        margin-bottom: 24px;
    }}

    /* Chat message tweaks */
    [data-testid="stChatMessage"] {{
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 6px;
    }}
    [data-testid="stChatMessage"][data-role="user"] {{
        background: #fff0f1;
        border-left: 4px solid {RED};
    }}
    [data-testid="stChatMessage"][data-role="assistant"] {{
        background: #f0f4ff;
        border-left: 4px solid {BLUE};
    }}

    /* Primary button → red */
    .stButton > button[kind="primary"],
    button[data-testid="baseButton-primary"] {{
        background-color: {RED} !important;
        border-color: {RED} !important;
        color: white !important;
        border-radius: 8px !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: #c4222e !important;
    }}

    /* Chat input */
    [data-testid="stChatInputTextArea"] {{
        border-radius: 12px !important;
    }}
    [data-testid="stChatInputSubmitButton"] {{
        background-color: {BLUE} !important;
        color: white !important;
        border-radius: 8px !important;
    }}

    /* Reasoning panel */
    .reasoning-panel {{
        background: #f0f4ff;
        border-left: 4px solid {BLUE};
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-family: 'Inter', monospace !important;
        font-size: 13px;
        color: #334;
        white-space: pre-wrap;
        margin-top: 4px;
    }}

    /* History items in sidebar */
    .hist-item {{
        background: white;
        border-left: 3px solid {BLUE};
        border-radius: 0 6px 6px 0;
        padding: 8px 10px;
        margin: 4px 0;
        font-size: 12px;
    }}
    .hist-ts {{ color: #888; font-size: 11px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_credentials(username: str, password: str) -> bool:
    try:
        return st.secrets["users"].get(username) == password
    except Exception:
        return False


def _fetch_history(user_id: str) -> list[dict]:
    try:
        r = httpx.get(
            f"{BACKEND_URL}/users/{user_id}/history",
            params={"limit": 20},
            timeout=5.0,
        )
        if r.status_code == 200:
            return r.json().get("interactions", [])
    except Exception:
        pass
    return []


def _set_level_api(user_id: str, level: str) -> bool:
    try:
        r = httpx.post(
            f"{BACKEND_URL}/users/{user_id}/level",
            json={"level": level},
            timeout=5.0,
        )
        return r.status_code == 200
    except Exception:
        return False


# ── Backend SSE stream ────────────────────────────────────────────────────────

class _BackendStream:
    """
    Reads the /chat/stream SSE endpoint and separates visible tokens from metadata.

    Current event types handled:
      {"type": "token",  "text": "..."}            – visible response chunk
      {"type": "done",   "mode": "...",
                         "sources": [...],
                         "reasoning": "...",        – full reasoning trace (debug=true only)
                         "cefr_level": "..."}

    ── TODO for Person B (backend) ──────────────────────────────────────────
    To enable live reasoning streaming, emit a new event type BEFORE "token" events:
      {"type": "reasoning_token", "text": "..."}
    These should carry each chunk of the <reasoning> block as it is generated.
    The frontend will stream them into the "🧠 Thinking…" panel in real-time.
    Once implemented, the "reasoning" field in the "done" event is no longer needed.
    ─────────────────────────────────────────────────────────────────────────
    """

    def __init__(self):
        self.meta: dict = {}
        self._reasoning_chunks: list[str] = []
        self.error: str | None = None
        # Placeholder set by caller to stream reasoning live (future use)
        self.reasoning_placeholder = None

    def token_stream(self, payload: dict, debug: bool = False):
        """Sync generator of visible response tokens for st.write_stream."""
        params = {"debug": "true"} if debug else {}
        try:
            with httpx.Client(timeout=120.0) as client:
                with client.stream(
                    "POST", f"{BACKEND_URL}/chat/stream",
                    params=params, json=payload
                ) as resp:
                    if resp.status_code != 200:
                        self.error = f"Backend error: HTTP {resp.status_code}"
                        return

                    for line in resp.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        try:
                            event = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        etype = event.get("type")

                        if etype == "reasoning_token":
                            # Live reasoning chunk — stream to placeholder when available
                            chunk = event.get("text", "")
                            self._reasoning_chunks.append(chunk)
                            if self.reasoning_placeholder is not None:
                                self.reasoning_placeholder.markdown(
                                    f"<div class='reasoning-panel'>"
                                    f"{''.join(self._reasoning_chunks)}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                        elif etype == "token":
                            yield event.get("text", "")

                        elif etype == "done":
                            self.meta = event
                            if not self._reasoning_chunks and event.get("reasoning"):
                                self._reasoning_chunks = [event["reasoning"]]

        except httpx.ConnectError:
            self.error = (
                "Cannot reach the backend at `localhost:8000`. "
                "Make sure `uvicorn api:app` is running."
            )
        except httpx.TimeoutException:
            self.error = "Response timed out."

    @property
    def reasoning_text(self) -> str:
        return "".join(self._reasoning_chunks)


# ── Login page ────────────────────────────────────────────────────────────────

def show_login():
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown(
            f"""
            <div style="text-align:center;padding:48px 0 24px">
                <div style="font-size:56px;line-height:1">🌍</div>
                <div style="font-size:28px;font-weight:700;color:{BLUE};margin:10px 0 4px">
                    MrWorldwide
                </div>
                <div style="color:#666;font-size:14px;margin-bottom:28px">
                    Socratic French Tutor
                </div>
                <div class="tricolor">
                    <div class="tc-blue"></div>
                    <div class="tc-white"></div>
                    <div class="tc-red"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            username = st.text_input("Student ID", placeholder="e.g. student42")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Log In →", use_container_width=True, type="primary"
            )

        if submitted:
            if not username.strip():
                st.error("Please enter your Student ID.")
            elif _check_credentials(username.strip(), password):
                st.session_state.authenticated = True
                st.session_state.user_id = username.strip()
                st.session_state.cefr_level = "A1"
                st.session_state.messages = []
                st.session_state.history = _fetch_history(username.strip())
                st.rerun()
            else:
                st.error("Invalid Student ID or password.")


# ── Sidebar ───────────────────────────────────────────────────────────────────

def show_sidebar():
    user_id = st.session_state.user_id

    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding:4px 0 12px">
                <div style="font-size:18px;font-weight:700;color:{BLUE}">🌍 MrWorldwide</div>
                <div style="font-size:12px;color:#666;margin-top:2px">
                    Logged in as <b>{user_id}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # CEFR level
        current_level = st.session_state.get("cefr_level", "A1")
        selected = st.selectbox(
            "CEFR Level",
            CEFR_LEVELS,
            index=CEFR_LEVELS.index(current_level),
        )
        if st.button("Update Level", use_container_width=True):
            if _set_level_api(user_id, selected):
                st.session_state.cefr_level = selected
                st.toast(f"✓ Level updated to {selected}")
            else:
                st.error("Could not reach backend.")

        # Debug toggle (key binds directly to session_state)
        st.checkbox("Show Reasoning Trace", key="debug_mode",
                    value=st.session_state.get("debug_mode", False))

        st.divider()

        # Past conversations
        st.markdown(f"**Past Conversations**")
        history: list[dict] = st.session_state.get("history", [])

        if not history:
            st.caption("No past conversations yet.")
        else:
            for item in history:
                ts = time.strftime("%b %d · %H:%M", time.localtime(item.get("ts", 0)))
                preview = (item.get("student_input") or "")
                short = preview[:45] + "…" if len(preview) > 45 else preview

                with st.expander(f"🗒 {ts}"):
                    st.markdown(f"**You:** {preview}")
                    st.markdown(f"**Tutor:** {item.get('response', '')}")
                    if item.get("mode"):
                        st.caption(f"Mode: {item['mode'].upper()} · "
                                   f"Sources: {', '.join(item.get('sources', [])) or 'none'}")

        st.divider()

        if st.button("Log Out", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ── Chat page ─────────────────────────────────────────────────────────────────

def show_chat():
    user_id = st.session_state.user_id
    level   = st.session_state.get("cefr_level", "A1")
    debug   = st.session_state.get("debug_mode", False)

    # Header with tricolor stripe
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:14px;padding:8px 0 4px">
            <span style="font-size:36px">🌍</span>
            <div>
                <div style="font-size:24px;font-weight:700;color:{BLUE};line-height:1.2">
                    MrWorldwide
                </div>
                <div style="font-size:13px;color:#666">
                    Socratic French Tutor &nbsp;·&nbsp; Level <b>{level}</b>
                </div>
            </div>
        </div>
        <div class="tricolor">
            <div class="tc-blue"></div>
            <div class="tc-white"></div>
            <div class="tc-red"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render message history
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("reasoning") and debug:
                with st.expander("🧠 Reasoning Trace", expanded=False):
                    st.markdown(
                        f"<div class='reasoning-panel'>{msg['reasoning']}</div>",
                        unsafe_allow_html=True,
                    )

    # Input
    prompt = st.chat_input("Type a French sentence to practice…")
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    payload = {
        "user_id": user_id,
        "message": prompt,
        "level": level,
        "forced_mode": None,
    }

    streamer = _BackendStream()

    with st.chat_message("assistant"):
        # If debug mode: show reasoning panel placeholder before response streams
        if debug:
            reasoning_expander = st.expander("🧠 Thinking…", expanded=True)
            with reasoning_expander:
                streamer.reasoning_placeholder = st.empty()

        full_response = st.write_stream(
            streamer.token_stream(payload, debug=debug)
        )

    if streamer.error:
        st.error(f"⚠️ {streamer.error}")
        return

    # Metadata footer
    meta = streamer.meta
    mode = meta.get("mode", "")
    sources = meta.get("sources", [])
    footer = (
        f"\n\n---\n*Mode: **{mode.upper()}** | "
        f"Sources: {', '.join(sources) or 'none'} | Level: **{level}***"
    )
    full_response = (full_response or "") + footer

    # Update displayed message with footer (re-render last bubble)
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "reasoning": streamer.reasoning_text,
    })

    # Refresh sidebar history
    st.session_state.history = _fetch_history(user_id)
    st.rerun()


# ── Entry point ───────────────────────────────────────────────────────────────

if not st.session_state.get("authenticated"):
    show_login()
else:
    show_sidebar()
    show_chat()
