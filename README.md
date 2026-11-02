# Mom or Dad? Family resemblance with PCA

Upload a father, a mother and their child. The app says which parent the child looks more like, overall and feature by feature. Everything is built on PCA (eigenfaces).

**Live demo:** open the GitHub Pages site for this repo. The Streamlit app runs entirely in your browser (via [stlite](https://github.com/whitphx/stlite) and Pyodide), so no server is needed and photos never leave your computer. The first load takes about 20–40 seconds. The same `app.py` also runs on Streamlit Cloud: https://mom-or-dad.streamlit.app/

**Example family:** put three photos you have permission to use in `examples/` as `father.jpg`, `mother.jpg` and `child.jpg`. A "Try an example family" button then appears. If the child is a daughter, set `EXAMPLE_IS_SON = False` in `app.py`. TSKinFace images can't be used here, because the dataset can't be redistributed.

## The main idea

TSKinFace cuts every family member out of **one family photo**. So a model can "recognise" a family just from the shared lighting, colour and background, without looking at the faces at all. We show this, then use PCA to remove it.

- **Control test:** a model that sees only the background around the face (no face) still picks the right parents 70% of the time.
- **Spouse test:** husband and wife share the photo but are not related. How well they "match" measures photo leakage. A good model scores high on child vs parents and close to 50% on spouses.

## Results (one 80/20 split by family, 209 test children; chance = 50%)

| Model | Child vs parents | Spouse (photo) | Margin |
|---|---|---|---|
| Background only, no face | 69.9% | 72.0% | −2.1 |
| Baseline PCA | 75.2% | 68.0% | +7.2 |
| + face mask | 77.6% | 68.7% | +8.9 |
| + whitening + cosine | 79.5% | 65.1% | +14.4 |
| **+ remove photo directions (final)** | **74.8%** | **57.5%** | **+17.3** |
| Colour PCA (rejected) | 82.2% | 74.0% | +8.2 |

A different split moves these numbers by about 2–3 points. Averaged over 5 folds, the final model scores **72.7%** (spouse 60.0%).

We also tried removing a sex direction (male − female) and an age direction (adult − child). Over 5 folds it made no difference (72.4% vs 72.7%), so it was left out.

- **Most inherited regions:** the eyes (about 70%), then the nose (63%).
- **Fair verdict (5-fold, all 1,015 children):** the baseline calls 51% of sons "like dad" but only 41% of daughters. The final model, ranking each child against same-sex children, gives 49% vs 49%.

## How it works

1. **Clean**: grayscale, histogram equalisation, and an elliptical face mask (background set to grey).
2. **PCA(0.95, whiten=True)**: whitening gives every eigenface an equal vote, so lighting and pose stop dominating.
3. **Remove photo directions** with a projection matrix `P = I − QQᵀ`. The directions are the top 20 eigenvectors of the husband–wife cross-covariance, i.e. what unrelated people in the same photo share.
4. **Verdict**: g = cos(child, father) − cos(child, mother), reported as a percentile among children of the same sex.
5. **Regions**: the same steps on six patches (forehead, eyes, nose, mouth, chin) cut from the cleaned face, removing 5 photo directions each.

Every step is linear, so each face space is one matrix: `z = (x − mean) @ W`. The app only needs NumPy and OpenCV.

**App alignment:** OpenCV's YuNet detector finds the eyes, and the photo is warped so the eyes land where they are in the TSKinFace crops (eye midpoint at (32, 26), 22.25 px apart, squashed vertically by 0.76).

## Files

| File | What it is |
|---|---|
| `mom_or_dad_simple.ipynb` | The whole project step by step, with explanations |
| `index.html` | GitHub Pages entry point: loads stlite and runs `app.py` in the browser |
| `app.py` | Streamlit app (same file for GitHub Pages and Streamlit Cloud) |
| `examples/` | Optional example family photos (add your own) |
| `model.npz` | Saved face spaces and calibration data (made by the notebook) |
| `yunet.onnx` | OpenCV face + eye detector (MIT licence, opencv_zoo) |

## Run

**GitHub Pages:** in the repo, go to Settings → Pages → Deploy from a branch → `main`, folder `/ (root)`. Pages serves `index.html`, which starts the app.

**Locally:**
```
pip install -r requirements.txt
streamlit run app.py
```

To retrain, put the dataset at `TSKinFace_Data/TSKinFace_cropped/{FMSD,FMS,FMD}` and run the notebook (about 3 minutes).

## Limitations

- There is no ground truth for "which parent", so it is tested indirectly (verification and the bias check).
- Photo removal reduces the shared-photo effect but does not remove it completely.
- Faces are 64×64 grayscale, from one dataset of mostly East Asian families.
- Works best with front-facing, evenly lit photos.

## Dataset

The TSKinFace images are not included (redistribution is not allowed). Cite: Qin, X., Tan, X., Chen, S. "Tri-Subject Kinship Verification: Understanding the Core of A Family." IEEE Transactions on Multimedia, 2015.
