# Project Documentation

**Lagos Traffic Congestion Predictor**
3MTT NextGen Capstone — AI-16 | Williams Ikechukwu
Fellow ID: FE/23/55951287
AI & Machine Learning NextGen Cohort, Federal Capital Territory

This document covers the full methodology, findings, and analysis behind
the project. For setup/run/deployment instructions, see `README.md`.

---

## 1. Problem Statement

No publicly available, labeled traffic congestion dataset exists for
Nigerian cities. Existing traffic prediction research and datasets are
built around Western urban patterns — car-dominant traffic, standardized
road infrastructure, and exogenous factors (weather, holidays) that
don't reflect Nigerian realities such as okada/keke-dominated informal
transport, market-day traffic surges, school-run patterns, and periodic
fuel scarcity.

This project addresses that gap in two parts:
1. Generate a realistic **synthetic Lagos traffic dataset**, structured
   around a real-world reference schema but extended with Nigeria-specific
   signals.
2. Train, evaluate, and deploy a classification model that predicts
   congestion level from time, weather, and contextual features.

## 2. Data Source & Generation Methodology

No ready-made Nigerian traffic dataset exists publicly (verified via
search across Kaggle, HDX, and academic sources). The closest usable
reference was Kaggle's `hasibullahaman/traffic-prediction-dataset` — a
single-junction dataset with 15-minute vehicle counts by type (Car,
Bike, Bus, Truck) and a 4-class congestion label. This schema was
extended for the Nigerian context and used to synthetically generate:

- **46,080 rows**: 8 named Lagos road segments × 60 days × 15-minute intervals
- **8 segments**, each with realistic capacity, centrality, lane count,
  and road type (highway/arterial), based on real Lagos roads (Third
  Mainland Bridge, Lekki-Epe Expressway, Ikorodu Road, Agege Motor Road,
  Apapa-Oshodi Expressway, Ozumba Mbadiwe, Ajah-Sangotedo Road,
  Berger-Ojota Corridor)
- **Vehicle types**: `CarCount`, `KekeOkadaCount` (replacing the
  original dataset's generic "Bike" category to reflect Nigeria's
  keke/okada-dominant informal transport), `BusCount`, `TruckCount`
- **Nigeria-specific exogenous features**: `Is_Market_Day`,
  `Is_School_Hours`, `Is_Raining`, `Rainfall_Intensity`,
  `Is_Fuel_Scarcity`

**Congestion label derivation**: `Traffic_Situation` (1=Low, 2=Normal,
3=High, 4=Heavy) was computed from a load ratio — total vehicle volume
relative to segment capacity — adjusted upward for rainfall and fuel
scarcity, then bucketed into the four ordinal classes. Time-of-day and
day-of-week traffic curves were calibrated to reflect realistic Lagos
rush-hour patterns (sharp weekday AM/PM peaks, flatter weekends), and
rainfall seasonality followed Lagos's actual wet-season pattern
(heaviest April–July).

See `Data_Generation.ipynb` for the full generation code and rationale.

## 3. Feature Engineering

- **Date/Time**: decomposed into `Day_of_month`, `Month`,
  `Is_weekend`, `Minute`, and cyclically-encoded hour
  (`hour_sin`/`hour_cos`) so that adjacent hours (e.g. 23:00 and 00:00)
  remain numerically close — a plain integer hour would incorrectly
  treat them as maximally distant.
- **Day of week**: one-hot encoded (nominal, no true order) rather than
  label-encoded, to avoid imposing a false ordinal relationship between
  unrelated days.
- **Target (`Traffic_Situation`)**: explicitly mapped to an ordinal
  integer scale (1=Low → 4=Heavy) rather than left as arbitrary
  category labels, since the classes have a genuine severity order
  that both modeling and evaluation should respect.
- **Dropped columns**: `Total` (a near-direct sum of the four vehicle
  count columns and a direct input to the label formula — keeping it
  would let the model trivially reconstruct the label rather than
  learn generalizable patterns) and `Is_Raining` (redundant with
  `Rainfall_Intensity`, which is 0 when not raining and otherwise
  carries strictly more information).

## 4. Model Development & Evaluation Methodology

- **Baseline**: Decision Tree, tuned via `GridSearchCV` (5-fold CV,
  `scoring="f1_macro"`) — established a reference performance level
  and served as a pipeline sanity check before investing in a more
  complex model.
- **Final model**: Random Forest (`class_weight="balanced"`), tuned via
  `RandomizedSearchCV` (5-fold CV, 10 sampled candidate combinations,
  `scoring="f1_macro"`) over `n_estimators`, `max_depth`,
  `min_samples_split`, `min_samples_leaf`.
- **Train/test split**: 80/20, stratified by target class, to preserve
  class proportions given the imbalance described below.
- **Evaluation metric — why F1-macro, not accuracy**: the target
  classes are imbalanced (~49% Low, 37% Normal, 10% High, 4% Heavy).
  Accuracy weighs every *prediction* equally, which means a model can
  score deceptively high by performing well only on the majority
  classes (Low, Normal) while getting the rare-but-consequential
  classes (High, Heavy) wrong. F1-macro instead weighs every *class*
  equally, computing F1 independently per class and averaging without
  weighting by frequency — ensuring the model is genuinely evaluated on
  its ability to detect severe congestion, not just common conditions.

## 5. Results

| Metric | Decision Tree (baseline) | Random Forest (final) |
|---|---|---|
| CV (validation) F1-macro | 0.941 | 0.948 |
| Train F1-macro | 0.994 | 0.996 |
| Test F1-macro | 0.950 | 0.947 |

Both models perform comparably (within noise of each other), which is
expected given the label was generated by a fairly deterministic
formula — both a single well-tuned tree and an ensemble can learn it
almost perfectly. **Random Forest was selected as the final model** for
its lower variance and greater robustness to noise, which matters more
once the model is eventually validated against real-world data.

**Per-class performance (Random Forest, test set):**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Low | 0.99 | 0.99 | 0.99 | 4,505 |
| Normal | 0.97 | 0.97 | 0.97 | 3,440 |
| High | 0.89 | 0.93 | 0.91 | 941 |
| Heavy | 0.92 | 0.92 | 0.92 | 330 |

**Confusion matrix finding**: reviewing the full confusion matrix
(Low: 4,449 correct / 56 → Normal; Normal: 3,329 correct / 29 → Low, 82
→ High; High: 872 correct / 44 → Normal, 25 → Heavy; Heavy: 302 correct
/ 28 → High), the two extreme corners — Low predicted as Heavy, and
Heavy predicted as Low — are **exactly zero**. Every misclassification
occurs between *adjacent* severity levels only. This matters for
real-world usability: a model that occasionally confuses "High" and
"Heavy" is a minor inconvenience, but one that could tell a commuter
"Low" traffic when conditions are actually "Heavy" would be a genuine
safety/reliability failure. The model structurally avoids that failure
mode.

The class with the highest relative error rate is **Heavy** (8.5%,
mostly confused with High) — consistent with it being the rarest class
(~4% of the data), giving the model the least training signal to learn
its precise boundary.

## 6. Feature Importance Analysis

The trained Random Forest's feature importances (see `Model.ipynb` for
the full chart) show a clear two-tier structure:

**Dominant features (~0.08–0.23 importance each)**: `KekeOkadaCount`,
`CarCount`, `BusCount`, `TruckCount`, `hour_cos`, `hour_sin`,
`TruckCount` — together accounting for the large majority of the
model's decision-making.

**Minor features (~0.01–0.03 each)**: `Lanes`, `Road_Type`,
`Is_School_Hours`, individual `Segment_*` dummies, `Is_weekend`,
`Day_of_month`, `Is_Market_Day`. Notably, `Rainfall_Intensity` and
`Is_Fuel_Scarcity` rank even lower — near the bottom of the full
importance list, alongside individual `Day_of_week_*` dummies.

**Interpretation**: this is a direct consequence of how the target was
generated — vehicle counts are the model's strongest predictors because
they are the near-literal arithmetic input to the label formula
(`Total ≈ CarCount + KekeOkadaCount + BusCount + TruckCount`, and
`Total` was a direct driver of the load-ratio calculation). This
confirms the model learned a coherent, plausible signal, but also
means the individually-low importance scores for Nigeria-specific
factors (market day, school hours, fuel scarcity, rainfall) understate
how *reliably* they were generated by the synthetic label formula in
the first place — the formula weighted raw volume more heavily than
these exogenous factors.

## 7. Testing Nigeria-Specific Context Feature Influence

A natural follow-up question: **do time and Nigeria-specific conditions
(rain, market day, fuel scarcity) actually affect the model's
predictions**, or are they effectively ignored in favor of vehicle
counts? This was tested directly (not just inferred from importance
scores) by holding vehicle counts fixed and varying time/conditions.

**Finding 1 — at low, unambiguous vehicle counts**: with counts fixed
at a level the model confidently classifies as "Low" (dry conditions:
96.1% Low confidence), introducing heavy rain (intensity 1.0) shifted
confidence to 81% Low / 18.6% Normal — a real, measurable effect, but
not large enough to flip the predicted class.

**Finding 2 — near a decision boundary**: with vehicle counts fixed at
a level near the High/Heavy boundary, stacking rain + fuel scarcity
together shifted the Heavy-class probability from 2.4% to 19.8% — an
~8x increase. It did not flip the predicted label in this specific
test, but demonstrates that near class boundaries, these contextual
factors exert a meaningfully larger — and potentially decisive —
influence.

**Conclusion**: vehicle volume dominates the model's predictions
overall, but time-of-day and Nigeria-specific contextual factors exert
a real, secondary influence, most impactful near class decision
boundaries rather than as independent, strong drivers on their own.
This is an honest and defensible finding about what the synthetic
label-generation formula emphasized, and is a legitimate basis for
future work: rebalancing the label formula to give exogenous factors
more predictive weight if a stronger standalone contextual signal is
desired, or validating this relationship against real observed data.

## 8. Example Inputs by Class

For testing the deployed app or recording demo material, the following
inputs were verified directly against the deployed model
(`Third Mainland Bridge` segment, `Rainfall_Intensity=0`, no market
day/school hours/fuel scarcity unless noted):

| Target class | Day | Hour | Car | Keke/Okada | Bus | Truck | Model output |
|---|---|---|---|---|---|---|---|
| **Low** | Sunday | 3:00 AM | 20 | 15 | 8 | 3 | Low — 100.0% |
| **Normal** | Wednesday | 12:00 PM | 210 | 165 | 90 | 35 | Normal — 99.0% |
| **High** | Friday | 8:00 AM | 294 | 231 | 126 | 49 | High — 94.0% |
| **Heavy** | Friday | 8:00 AM | 380 | 300 | 160 | 65 | Heavy — 98.3% |

These confirm the model responds correctly across the full class range,
and that the class boundary on this segment sits roughly between
~300 total vehicles (Low/Normal) and ~800+ total vehicles (Heavy) per
15-minute window — consistent with the segment's configured capacity.
Note that this boundary is segment-specific: smaller-capacity roads
(e.g. Agege Motor Road) reach "Heavy" at proportionally lower raw
counts.

## 9. Deployment

The trained model is deployed as an interactive Streamlit web app
(`Traffic.py`), hosted on Streamlit Community Cloud. See `README.md`
for full deployment instructions.

**Note on encoder implementation**: `Model.ipynb` (the research/
evaluation notebook) uses `category_encoders.OneHotEncoder`. The
deployed model instead uses scikit-learn's own `OneHotEncoder` inside a
`ColumnTransformer` — same features, same Random Forest
hyperparameters, same target, functionally equivalent performance
(Test F1-macro 0.947 in both cases). This change was made after
`category_encoders` caused a cross-environment pickle compatibility
error (`AttributeError: 'OrdinalEncoder' object has no attribute
'index_start'`) when the model was unpickled in Streamlit Cloud's
freshly-built environment, despite an identical version pin.
`category_encoders`'s internal object structure proved less reliable
for cross-environment deployment than scikit-learn's, which is far more
rigorously maintained and tested for this exact use case. This is
documented here as a deliberate, evidence-based engineering decision —
not an inconsistency — between the research notebook and the deployed
application.

## 10. Known Limitations

- **Synthetic data**: the entire dataset is generated, not observed.
  While calibrated against realistic Lagos traffic patterns and cross-
  referenced against academic junction studies (Ibadan, Abuja, Port
  Harcourt, Ado-Ekiti) for plausibility, it has not been validated
  against continuous real-world sensor or API-collected data.
- **Deterministic label formula**: because `Traffic_Situation` was
  computed via a formula rather than observed, the high model
  performance (F1-macro ~0.95) partly reflects how learnable that
  formula is, not necessarily how hard real-world congestion
  prediction would be with the same features.
- **Vehicle counts required as live app inputs**: the deployed model's
  strongest predictors are real-time vehicle counts, which a live user
  cannot know in advance — the app currently requires the user to
  estimate them. In a future version with live sensor or API-fed
  traffic volume data, these would be filled in automatically.
- **Geographic scope**: covers Lagos only; other Nigerian states/cities
  have not yet been modeled.
- **Nigeria-specific contextual features have secondary influence**:
  as detailed in Section 7, market day, school hours, fuel scarcity,
  and rainfall exert real but secondary effects relative to vehicle
  volume — a finding worth further investigation with real data.

## 11. Future Work

- Validate the model against real observed traffic data (starting with
  small-scale academic junction studies already identified: Ibadan,
  Abuja, Port Harcourt, Ado-Ekiti)
- Incorporate HDX Nigerian road network shapefiles for richer segment-
  level features (connectivity, centrality) beyond the current
  manually-specified values
- Expand coverage to additional Nigerian cities/states
- Explore live API-based vehicle count estimation (e.g. via Google
  Maps/TomTom traffic APIs) to remove the need for manual vehicle count
  input in the deployed app
- Rebalance the synthetic label-generation formula to give exogenous
  Nigeria-specific factors greater independent predictive weight, and
  re-test whether this changes the feature importance hierarchy

## 12. Conclusion

This project demonstrates an end-to-end machine learning pipeline —
from identifying a real data gap, through synthetic data generation,
feature engineering, model development and rigorous evaluation, to a
deployed, usable web application — tailored specifically to Nigerian
urban traffic realities. The honest reporting of both strong results
(F1-macro ~0.95, zero extreme-class confusion) and genuine limitations
(vehicle-count dependency, secondary influence of contextual features)
reflects the project's actual findings rather than an idealized
account, and provides a clear, evidence-based foundation for future
work toward real-world validation.
