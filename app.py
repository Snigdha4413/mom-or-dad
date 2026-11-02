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


# ---------------------------------------------------------------- page
st.title('Mom or Dad?')
st.write('Upload the father, the mother and their child. Each face is aligned, cleaned and projected into a PCA '
         'face space with the shared-photo effects removed.')

# example family shipped with the app (photos we have permission to use)
EXAMPLE = ['examples/father.jpg', 'examples/mother.jpg', 'examples/child.jpg']
EXAMPLE_IS_SON = True                                             # change if the example child is a daughter
if all(os.path.exists(p) for p in EXAMPLE) and st.button('Try an example family'):
    st.session_state['example'] = True

c1, c2, c3 = st.columns(3)
f_file = c1.file_uploader('Father', type=['jpg', 'jpeg', 'png'])
m_file = c2.file_uploader('Mother', type=['jpg', 'jpeg', 'png'])
c_file = c3.file_uploader('Child', type=['jpg', 'jpeg', 'png'])
son = st.radio('The child is a', ['Son', 'Daughter'], horizontal=True) == 'Son'

if st.session_state.get('example') and not (f_file or m_file or c_file):  # uploading anything replaces the example
    f_file, m_file, c_file = EXAMPLE
    son = EXAMPLE_IS_SON
    st.caption('Showing the example family. Upload your own photos to replace it.')
kids = 'sons' if son else 'daughters'
sex = 'son' if son else 'daughter'

if f_file and m_file and c_file:
    f, m, c = align(f_file, 'father'), align(m_file, 'mother'), align(c_file, 'child')
    st.image([big(f), big(m), big(c)], caption=['Father', 'Mother', 'Child'], width=150)

    # whole-face verdict
    zf, zm, zc = to_space(prep(f), 'face'), to_space(prep(m), 'face'), to_space(prep(c), 'face')
    p = dad_percentile(zf, zm, zc, model[f'g_{sex}'])
    st.subheader('Verdict')
    if 0.4 <= p <= 0.6:
        st.info(f'**An even mix.** More like dad than {p:.0%} of {kids} in our data.')
    elif p > 0.6:
        st.success(f'**Looks more like DAD.** More like dad than {p:.0%} of {kids} in our data.')
    else:
        st.success(f'**Looks more like MUM.** More like mum than {1 - p:.0%} of {kids} in our data.')
    st.progress(float(p), text=f'Mum ◀ {1 - p:.0%} | {p:.0%} ▶ Dad')

    # region map
    st.subheader('Feature by feature')
    canvas = cv2.resize(c, (256, 256), interpolation=cv2.INTER_CUBIC)
    lines = []
    for i, (name, box) in enumerate(zip(model['region_names'], model['region_boxes'])):
        key = f'r{i}'
        zf, zm, zc = (to_space(region(prep(x), box), key) for x in (f, m, c))
        q = dad_percentile(zf, zm, zc, model[f'{key}_g_{sex}'])
        colour = np.array([42, 120, 214]) if q > 0.5 else np.array([232, 123, 164])   # blue = dad, pink = mum
        strength = min(1.0, abs(q - 0.5) * 2.4)
        top, bottom, left, right = (int(v) * 4 for v in box)
        patch = canvas[top:bottom, left:right].astype(float)
        canvas[top:bottom, left:right] = (patch * (1 - 0.6 * strength) + colour * 0.6 * strength).astype(np.uint8)
        who = 'dad' if q > 0.5 else 'mum'
        lines.append(f'**{name}**: like {who} (more than {max(q, 1 - q):.0%} of {kids})')
    left_col, right_col = st.columns([1, 1.3])
    left_col.image(canvas, caption='Blue = dad, pink = mum, stronger colour = stronger lean', width=256)
    right_col.markdown('\n\n'.join(lines))

    # do they look related?
    st.subheader('Do they look related at all?')
    zc, zp = to_space(prep(c), 'face'), (to_space(prep(f), 'face') + to_space(prep(m), 'face')) / 2
    s = cos(zc, zp)
    st.write(f"The child looks more like this couple than like **{np.mean(model['sim_stranger'] < s):.0%}** of "
             f"random couples, and more than {np.mean(model['sim_own'] < s):.0%} of real children resemble their own parents.")

    st.caption('Demo, not a genetic test. Trained on TSKinFace (787 families). The verdict is ranked against '
               'children of the same sex, so a son is not called "like dad" just for being a boy.')
