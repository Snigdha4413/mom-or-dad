import io
import os
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageOps

# ---------------------------------------------------------------- load model + face detector
model = np.load('model.npz')
detector = cv2.FaceDetectorYN.create('yunet.onnx', '', (320, 320), 0.6)   # finds faces + eye positions

# Where the eyes sit in a TSKinFace 64x64 crop (measured from the dataset's eye annotations):
EYE_MID_X, EYE_MID_Y = 32, 26     # point between the eyes
EYE_DIST = 22.25                  # pixels between the eyes
SQUASH_Y = 0.76                   # the dataset crops are squashed vertically

yy, xx = np.mgrid[0:64, 0:64] / 64
MASK = ((xx - 0.5) / 0.40) ** 2 + ((yy - 0.52) / 0.52) ** 2 <= 1


# ---------------------------------------------------------------- step 1: crop the face like the dataset
def align(file, who):
    img = np.array(ImageOps.exif_transpose(Image.open(file)).convert('RGB'))
    scale = min(1.0, 1024 / max(img.shape[:2]))                  # detect on a smaller copy (faster)
    small = cv2.resize(img, None, fx=scale, fy=scale)
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    if faces is None:
        st.error(f'No face found in the {who} photo. Try a clear, front-facing photo.')
        st.stop()

    face = max(faces, key=lambda f: f[2] * f[3]) / scale          # biggest face, back to full size
    left, right = sorted([face[4:6], face[6:8]], key=lambda p: p[0])

    # YuNet's eye points sit a little higher and wider than the dataset's; correct for that
    axis = right - left
    dist = np.linalg.norm(axis) / 1.068
    axis = axis / np.linalg.norm(axis)
    down = np.array([-axis[1], axis[0]])
    mid = (left + right) / 2 + (0.047 * axis + 0.050 * down) * dist

    # rotate so the eyes are level, scale to the right eye distance, squash, move eyes into place
    angle = -np.arctan2(axis[1], axis[0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) * (EYE_DIST / dist)
    A = np.diag([1, SQUASH_Y]) @ rot
    b = np.array([EYE_MID_X, EYE_MID_Y]) - A @ mid
    return cv2.warpAffine(img, np.hstack([A, b[:, None]]), (64, 64), flags=cv2.INTER_AREA,
                          borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- step 2: same cleaning as the notebook
def prep(img):
    g = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)).astype(float)
    g[~MASK] = 128                                                # background -> plain grey
    return g.reshape(-1)


def region(x, box):
    top, bottom, left, right = box
    return x.reshape(64, 64)[top:bottom, left:right].reshape(-1)


# ---------------------------------------------------------------- step 3: face space + verdict
def to_space(x, name):
    return (x - model[name + '_mean']) @ model[name + '_W']       # whitened PCA + removals, in one matrix


def cos(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


def dad_percentile(zf, zm, zc, ref):
    g = cos(zc, zf) - cos(zc, zm)                                 # > 0 means more like dad
    return np.mean(ref < g)                                       # rank among same-sex children


def big(img):
    return cv2.resize(img, (192, 192), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------- page: Method -> Photos -> Analysis
S = st.session_state
S.setdefault('step', 'mode'); S.setdefault('kids', []); S.setdefault('mode', 'upload')
EXAMPLE = ['examples/father.jpg', 'examples/mother.jpg', 'examples/child.jpg']
TYPES = ['jpg', 'jpeg', 'png']


def pick(mode): S.update(mode=mode, step='photos')
def reset(): S.update(step='mode', kids=[], data=None, ex=False)
def rd(x): return x.getvalue() if hasattr(x, 'getvalue') else open(x, 'rb').read()


def stepper(n):
    tags = ''.join(f'<span class="{"on" if i <= n else ""}">{i + 1} · {t}</span>' for i, t in enumerate(['Method', 'Photos', 'Analysis']))
    st.markdown(f'<div class="steps">{tags}</div>', unsafe_allow_html=True)


def analyse(f, m, c, son):
    """Whole-face verdict, feature map and relatedness for one child. Returns dad percentile (or None)."""
    sex, kids = ('son', 'sons') if son else ('daughter', 'daughters')
    zf, zm, zc = to_space(prep(f), 'face'), to_space(prep(m), 'face'), to_space(prep(c), 'face')
    if cos(zf, zm) > 0.99 or cos(zc, zf) > 0.99 or cos(zc, zm) > 0.99:
        st.warning('Two of the photos look identical. Use three different people for a real result.')
        return None
    p = dad_percentile(zf, zm, zc, model[f'g_{sex}'])
    st.image([big(f), big(m), big(c)], caption=['Father', 'Mother', 'Child'], width=150)
    st.subheader('Verdict')
    if 0.4 <= p <= 0.6:
        st.info(f'**An even mix.** More like dad than {p:.0%} of {kids} in our data.')
    elif p > 0.6:
        st.success(f'**Looks more like DAD.** More like dad than {p:.0%} of {kids} in our data.')
    else:
        st.success(f'**Looks more like MUM.** More like mum than {1 - p:.0%} of {kids} in our data.')
    st.progress(float(p), text=f'Mum ◀ {1 - p:.0%} | {p:.0%} ▶ Dad')

    st.subheader('Feature by feature')
    canvas, lines = cv2.resize(c, (256, 256), interpolation=cv2.INTER_CUBIC), []
    for i, (name, box) in enumerate(zip(model['region_names'], model['region_boxes'])):
        key = f'r{i}'
        zf, zm, zc = (to_space(region(prep(x), box), key) for x in (f, m, c))
        q = dad_percentile(zf, zm, zc, model[f'{key}_g_{sex}'])
        colour = np.array([42, 120, 214]) if q > 0.5 else np.array([232, 123, 164])   # blue = dad, pink = mum
        strength = min(1.0, abs(q - 0.5) * 2.4)
        top, bottom, left, right = (int(v) * 4 for v in box)
        patch = canvas[top:bottom, left:right].astype(float)
        canvas[top:bottom, left:right] = (patch * (1 - 0.6 * strength) + colour * 0.6 * strength).astype(np.uint8)
        lines.append(f'**{name}**: like {"dad" if q > 0.5 else "mum"} (more than {max(q, 1 - q):.0%} of {kids})')
    a, b = st.columns([1, 1.3])
    a.image(canvas, caption='Blue = dad, pink = mum, stronger colour = stronger lean', width=256)
    b.markdown('\n\n'.join(lines))

    st.subheader('Do they look related at all?')
    zc, zp = to_space(prep(c), 'face'), (to_space(prep(f), 'face') + to_space(prep(m), 'face')) / 2
    s = cos(zc, zp)
    st.write(f"Looks more like this couple than **{np.mean(model['sim_stranger'] < s):.0%}** of random couples, "
             f"and more than {np.mean(model['sim_own'] < s):.0%} of real children resemble their own parents.")
    return p


# ---- step 1: how do you want to add photos?
if S['step'] == 'mode':
    stepper(0)
    st.markdown('<span class="mode-marker"></span>', unsafe_allow_html=True)   # tells the page to show the carousel
    st.title('How would you like to add your photos?')
    st.caption('Pick a method to begin.')
    a, b = st.columns(2)
    with a.container(border=True):
        st.subheader('Upload photos'); st.write('Choose pictures from your device. You can add several siblings at once.')
        st.button('Upload from device', on_click=pick, args=('upload',), use_container_width=True)
    with b.container(border=True):
        st.subheader('Live camera'); st.write('Take the photos right now with your webcam or phone camera.')
        st.button('Open camera', on_click=pick, args=('camera',), use_container_width=True)

# ---- step 2: the photos
elif S['step'] == 'photos':
    stepper(1)
    cam = S['mode'] == 'camera'
    st.title('Add the family')
    st.caption('Tip: clear, front-facing photos with good light work best.')
    one = (lambda label, key: st.camera_input(label, key=key)) if cam else (lambda label, key: st.file_uploader(label, type=TYPES, key=key))
    c1, c2 = st.columns(2)
    with c1: f = one('Father', 'f')
    with c2: m = one('Mother', 'm')

    st.subheader('Children')
    st.caption('Add one child or several siblings. Each one gets their own result.')
    kids = []
    if cam:
        shot = st.camera_input('Take a child photo', key=f'cam{len(S["kids"])}')
        if shot and st.button('Add this child'):
            S['kids'].append(shot.getvalue()); st.rerun()
        kids += [io.BytesIO(k) for k in S['kids']]
    kids += st.file_uploader('Upload children' if not cam else 'Or upload from device', type=TYPES, accept_multiple_files=True, key='ku')

    if all(os.path.exists(p) for p in EXAMPLE):
        st.button('Try an example family', on_click=lambda: S.update(ex=True))
    if S.get('ex') and not (f or m or kids):
        f, m, kids = EXAMPLE[0], EXAMPLE[1], [EXAMPLE[2]]

    sexes = []
    for i, k in enumerate(kids):
        x, y = st.columns([1, 5])
        x.image(k, width=80)
        sexes.append(y.radio(f'Child {i + 1} is a', ['Son', 'Daughter'], horizontal=True, key=f'sex{i}') == 'Son')

    back, go = st.columns(2)
    back.button('← Back', on_click=lambda: S.update(step='mode'), use_container_width=True)
    ready = bool(f and m and kids)
    if go.button('Analyze family', type='primary', disabled=not ready, use_container_width=True):
        S['data'] = {'f': rd(f), 'm': rd(m), 'kids': [(rd(k), s) for k, s in zip(kids, sexes)]}
        S['step'] = 'results'; st.rerun()
    if not ready:
        st.caption('Add a father, a mother and at least one child to continue.')

# ---- step 3: analysis
else:
    stepper(2)
    st.title('Your results')
    d = S['data']
    f, m = align(io.BytesIO(d['f']), 'father'), align(io.BytesIO(d['m']), 'mother')
    summary, ps = st.container(), []
    tabs = st.tabs([f'Child {i + 1}' for i in range(len(d['kids']))])
    for i, (tab, (raw, son)) in enumerate(zip(tabs, d['kids'])):
        with tab:
            p = analyse(f, m, align(io.BytesIO(raw), f'child {i + 1}'), son)
            if p is not None: ps.append(p)
    if len(ps) > 1:
        avg = float(np.mean(ps))
        who = 'dad' if avg > 0.5 else 'mum'
        summary.info(f'**Family overview:** across {len(ps)} children the lean is toward **{who}** '
                     f'(average {max(avg, 1 - avg):.0%} rank). Open each tab for the individual result.')
    st.button('Start over', on_click=reset, use_container_width=True)
    st.caption('Demo, not a genetic test. Trained on TSKinFace (776 families). Each verdict is ranked against '
               'children of the same sex, so a son is not called "like dad" just for being a boy.')
