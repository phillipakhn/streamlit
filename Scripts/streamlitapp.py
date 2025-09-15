# streamlitapp.py
# PGN → 64×200 paletted (pink-padded) → RGB[0,1] → TF SavedModel (z) → Elo
# Order: 1) Show ELO prediction first, 2) then images (hi-res, then 64×200)
# Adds a minimal empty sidebar for layout aesthetics.

import io, base64
import numpy as np
from PIL import Image
import streamlit as st
import chess.pgn
import tensorflow as tf

# ---- HIDE any JSON/expander debug blocks (kills the [0..13: NULL] UI) ----
st.markdown("""
<style>
  [data-testid="stJson"], [data-testid="stExpander"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# CONFIG (fixed constants)
# =========================
MODEL_DIR = "../Scripts/elo_savedmodel_barcodes2"  # folder with saved_model.pb + variables/

# From your training split:
Y_MEAN = 2163.5744788425636
Y_STD  = 247.24373670930683

DISPLAY_SCALE = 5  # upscale ×5 for presentation; CNN still uses 64×200
T_MAX = 200
H, W = 64, T_MAX

# Palette (must match training exactly)
PALETTE_RGB = [
    (255,255,255),(220,220,220),(0,0,0),
    (255,140,0),(255,215,0),(65,105,225),(60,179,113),(220,20,60),
    (139,69,19),(218,165,32),(25,25,112),(0,100,0),(178,34,34),
    (255,0,255)  # padding pink
]
SYM_TO_IDX = {
    ".":0,"P":1,"p":2,"K":3,"Q":4,"R":5,"B":6,"N":7,
    "k":8,"q":9,"r":10,"b":11,"n":12
}
PAD_IDX = 13

# =========================
# Minimal sidebar (for look & feel; intentionally empty)
# =========================
with st.sidebar:
    st.markdown("&nbsp;", unsafe_allow_html=True)  # non-breaking space to render sidebar
    # (leave empty per your request)

# =========================
# Helpers
# =========================
def _palette_flat():
    pal=[];  [pal.extend([r,g,b]) for (r,g,b) in PALETTE_RGB]
    pal += [0,0,0]*(256-len(PALETTE_RGB))
    return pal

def fen_to_64(board_fen: str):
    out=[]
    for row in board_fen.split("/"):
        for ch in row:
            if ch.isdigit(): out.extend(["."]*int(ch))
            else: out.append(ch)
    return (out+["."]*(64-len(out)))[:64]

def fen_column_indices(fen: str):
    board_part = fen.split()[0]
    return [SYM_TO_IDX.get(s,0) for s in fen_to_64(board_part)]

def pgn_to_fens(pgn_text: str):
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if not game:
        raise ValueError("Could not parse PGN.")
    board = game.board()
    fens=[]
    for mv in game.mainline_moves():
        board.push(mv)
        fens.append(board.fen())
        if len(fens) >= T_MAX:
            break
    return fens

def fens_to_paletted_64x200(fens):
    """Exact training format: 64×200 Paletted PNG with bright-pink right padding."""
    cols = [fen_column_indices(f) for f in fens]
    data=[]; T=len(cols)
    for y in range(H):
        for x in range(W):
            data.append(cols[x][y] if x < T else PAD_IDX)
    img = Image.new("P", (W, H))
    img.putpalette(_palette_flat(), rawmode="RGB")
    img.putdata(data)
    return img  # 64×200 paletted

def paletted_to_rgb01(imgP: Image.Image):
    """RGB float32 [0,1] (64,200,3) — mirrors training decode_png(...,3)/255."""
    return np.asarray(imgP.convert("RGB"), dtype=np.float32) / 255.0

def upscale_nearest(imgP: Image.Image, scale: int):
    """High-res display image using nearest-neighbor so colors/blocks stay crisp."""
    return imgP.resize((imgP.width*scale, imgP.height*scale), resample=Image.NEAREST)

@st.cache_resource
def load_model(model_dir: str):
    return tf.saved_model.load(model_dir)

def _get_signature_io_keys(model):
    sig = getattr(model, "signatures", {}).get("serving_default")
    if not sig: return None, None, None
    _, kwargs = sig.structured_input_signature
    in_key = next(iter(kwargs.keys())) if kwargs else None
    out_keys = list(sig.structured_outputs.keys()) if hasattr(sig, "structured_outputs") else []
    return sig, in_key, out_keys

def predict_z(model, x01_hw3: np.ndarray) -> float:
    """Run model; returns raw z-score (trained on standardized targets)."""
    x = np.expand_dims(x01_hw3, 0).astype(np.float32)  # (1,64,200,3)
    tens = tf.convert_to_tensor(x, dtype=tf.float32)
    sig, in_key, out_keys = _get_signature_io_keys(model)
    if sig and in_key:
        out_map = sig(**{in_key: tens})
        y = out_map[out_keys[0]] if out_keys else next(iter(out_map.values()))
    else:
        y = model(tens, training=False)
    return float(tf.reshape(y, [-1])[0].numpy())

def z_to_elo(z: float) -> float:
    return z * Y_STD + Y_MEAN

# =========================
# UI
# =========================
st.set_page_config(page_title="Chess ELO Predictor", page_icon="♟️", layout="wide")
st.title("♟️ Chess ELO Predictor — PGN → 64×200 barcode → CNN")

DEFAULT_PGN = """[Event "Example"]
[Site "Local"]
[Date "2025.09.15"]
[Round "-"]
[White "White"]
[Black "Black"]
[Result "*"]

1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 *
"""

pgn_text = st.text_area("Paste PGN", height=220, value=DEFAULT_PGN)

if st.button("Predict ELO", type="primary"):
    try:
        fens = pgn_to_fens(pgn_text)
        if not fens:
            st.error("No moves found in the PGN.")
        else:
            # Build CNN input (exact training image)
            pal_64x200 = fens_to_paletted_64x200(fens)

            # ---- Inference FIRST (so ELO shows before images) ----
            x01 = paletted_to_rgb01(pal_64x200)
            model = load_model(MODEL_DIR)
            z = predict_z(model, x01)
            elo = z_to_elo(z)

            st.metric("Predicted Elo", f"{int(round(elo))}")
            st.caption(f"Model output z = {z:.3f}  →  Elo = z*{Y_STD:.1f} + {Y_MEAN:.1f}")
            st.caption(f"Moves parsed: {len(fens)} (capped at {T_MAX})")

            # ---- Then images (hi-res first, then exact 64×200) ----
            hi = upscale_nearest(pal_64x200, DISPLAY_SCALE)
            buf_hi = io.BytesIO(); hi.save(buf_hi, format="PNG", optimize=True)
            st.image(
                f"data:image/png;base64,{base64.b64encode(buf_hi.getvalue()).decode()}",
                caption=f"Display image (scaled ×{DISPLAY_SCALE})",
                use_container_width=True
            )

            buf_small = io.BytesIO(); pal_64x200.save(buf_small, format="PNG", optimize=True)
            st.image(
                f"data:image/png;base64,{base64.b64encode(buf_small.getvalue()).decode()}",
                caption="CNN input image (exact 64×200 paletted)",
                use_container_width=False
            )

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
