# 🚦 Lagos Traffic Congestion Predictor

**3MTT NextGen Capstone — AI-16 | Williams Ikechukwu**
AI & Machine Learning NextGen Cohort, Federal Capital Territory

A machine learning system that predicts traffic congestion levels
(Low / Normal / High / Heavy) on major Lagos road segments, built to
reflect Nigerian urban traffic realities — okada/keke traffic,
market days, school hours, fuel scarcity, and rainfall — rather than
generic Western traffic patterns.

This repo contains the **deployed web app** for the model built and
evaluated in `Model.ipynb`. The app is a thin interface over the exact
trained pipeline (`model.pkl`) — same features, same Random Forest, same
hyperparameters, same F1-macro ≈ 0.947 test performance reported in the
notebook.

---

## Repository structure

```
.
├── Traffic.py                          # Streamlit web app
├── reproduce_model.py              # Reproduces model.pkl from the notebook's pipeline
├── lagos_synthetic_traffic_dataset.csv   # Synthetic training dataset
├── segment_metadata.csv            # Road segment lookup (Road_Type / Lanes per segment)
├── model.pkl                       # Trained Random Forest pipeline (from Model.ipynb)
├── requirements.txt
└── README.md
```

## Model summary

- **Algorithm**: Random Forest (`class_weight="balanced"`), tuned via
  `RandomizedSearchCV` (5-fold CV, `scoring="f1_macro"`)
- **Best params**: `n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=2`
- **Features**: Day of week, segment, road type, lanes, vehicle counts
  (Car/Keke-Okada/Bus/Truck), market day, school hours, rainfall
  intensity, fuel scarcity, day of month, month, weekend flag, minute,
  and cyclically-encoded hour (`hour_sin`/`hour_cos`)
- **Target**: `Traffic_Situation`, ordinal 1=Low, 2=Normal, 3=High, 4=Heavy
- **Performance**: CV F1-macro 0.948, Train F1-macro 0.996, Test F1-macro 0.947
- **Evaluation metric rationale**: F1-macro was used instead of accuracy
  because the classes are imbalanced (~49% Low, 37% Normal, 10% High,
  4% Heavy); accuracy would let the model ignore the rare-but-critical
  "Heavy" class while still scoring well, since it weighs every
  prediction equally rather than every class equally.
- **Confusion matrix finding**: the model never confuses the two
  extremes (Low↔Heavy) — all misclassifications occur between adjacent
  severity levels, which matters for real-world usability.

## Known limitation worth noting

Vehicle counts (`CarCount`, `KekeOkadaCount`, `BusCount`, `TruckCount`)
are the model's strongest predictors, and the app currently asks the
user to provide them manually (defaulted to reasonable starting values).
In a future iteration with live sensor or API-fed traffic volume data,
these would be filled in automatically rather than user-estimated — this
is flagged directly in the app's footer for transparency.

---

## Running locally

```bash
pip install -r requirements.txt
streamlit run Traffic.py
```

`model.pkl` and `segment_metadata.csv` are already included in this
repo, so no retraining is needed to run the app. If you want to
regenerate them from scratch (e.g. after changing the dataset), run:

```bash
python reproduce_model.py
```
(Note: this reruns the full `RandomizedSearchCV` hyperparameter search —
50 model fits, some up to 500 trees each — and can take several minutes.)

## Deploying to Streamlit Community Cloud

1. **Push this folder to a public GitHub repository**:

   ```bash
   git init
   git add .
   git commit -m "Lagos traffic congestion predictor"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

   > `model.pkl` is ~40MB (compressed) — under GitHub's 100MB file
   > limit, so a plain `git add`/`push` works without needing Git LFS.

2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign
   in with GitHub.
3. Click **"New app"**, select your repository, branch (`main`), and
   set the main file path to `Traffic.py`.
4. Click **Deploy**. Streamlit Cloud installs everything from
   `requirements.txt` and launches the app — you'll get a shareable
   `*.streamlit.app` link.

## Author

Williams Ikechukwu — 3MTT NextGen Fellow (AI-16), Omni Digital Media
Institute, Federal Capital Territory
