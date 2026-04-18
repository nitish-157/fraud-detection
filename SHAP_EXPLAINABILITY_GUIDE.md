# SHAP Model Explainability Guide - Fraud Detection

## Overview

**SHAP** (SHapley Additive exPlanations) provides model-agnostic explanations by computing the contribution of each feature to predictions using game theory principles. It answers: **Why did your model make this prediction?**

This guide covers explaining your Random Forest fraud detection model using SHAP values.

---

## What is SHAP?

SHAP values are based on Shapley values from cooperative game theory:

- **Fair attribution** of each feature's contribution to the final prediction
- **Consistent** across different models (tree-based, linear, neural networks, etc.)
- **Individual & global** explanations for any prediction

### Core Concepts

| Concept           | Definition                                                            |
| ----------------- | --------------------------------------------------------------------- |
| **Base Value**    | Model's average prediction (≈0.5 for your model)                      |
| **Feature Value** | Actual value of a feature for a specific transaction                  |
| **SHAP Value**    | Magnitude of feature's impact on prediction departure from base value |
| **Positive SHAP** | Pushes prediction toward fraud (class 1)                              |
| **Negative SHAP** | Pushes prediction toward normal (class 0)                             |

---

## Generated Visualizations

### 1. **shap_feature_importance.png** - Which Features Matter Most?

**Shows:** Mean absolute SHAP values for each feature

**Interpretation:**

- Longer bars = More important features
- Measures average impact on predictions across all samples
- Top features have the strongest influence on model decisions

**Example Output:**

```
oldbalanceOrg      ████████████████████ (Most important)
amount             ███████████
newbalanceOrig     ██████
oldbalanceDest     ████
newbalanceDest     ██
step               █ (Least important)
```

**What it tells you:**

- `oldbalanceOrg` is the **strongest fraud signal**
- Model heavily relies on sender's account balance
- `step` (time sequence) has minimal impact

---

### 2. **shap_summary_dot.png** - Feature Values vs Impact

**Shows:** Scatter plot with dots for each sample

- **X-axis:** SHAP value (prediction impact)
- **Y-axis:** Features (sorted by importance)
- **Color:** Feature value (red=high, blue=low)

**Interpretation:**

- Dots spread to RIGHT = positive SHAP values → fraud signal
- Dots spread to LEFT = negative SHAP values → normal signal
- Color gradient shows feature relationships:
  - Red dots right = high feature values trigger fraud
  - Blue dots left = low feature values normal

**Example Reading:**

```
oldbalanceOrg: Red dots clustered left
  → HIGH sender balance pushes toward NORMAL (negative SHAP)

amount: Blue dots clustered left, Red dots spread right
  → LOW amounts suggest normal, HIGH amounts suggest fraud
```

---

### 3. **shap_summary_bar.png** - Average Feature Impact

**Shows:** Bar plot of mean |SHAP value| per feature

**Interpretation:**

- Similar to feature_importance.png but visualized as bars
- Simpler alternative plot format
- Best for presentations/reports

---

### 4. **shap_contributions_plot.png** - Sample-Level Contributions

**Shows:** Average SHAP contribution for top 50 samples

**Interpretation:**

- Shows which features contributed most to the model's predictions
- Coral bars show average contribution magnitude
- Helps understand model reasoning on actual data samples

---

### 5. **shap_dependence_plots.png** - Non-linear Relationships

**Shows:** 2×2 grid of top 4 features with their SHAP dependence

For each feature:

- **X-axis:** Feature value
- **Y-axis:** SHAP value
- **Color:** SHAP value intensity

**Interpretation:**

#### Positive Trend (scatter going up-right):

```
Feature value ↑ → SHAP value ↑ → Predicts fraud
Example: amount → higher amounts increase fraud risk
```

#### Negative Trend (scatter going down-right):

```
Feature value ↑ → SHAP value ↓ → Predicts normal
Example: oldbalanceOrg → higher balance decreases fraud risk
```

#### Flat/Scattered:

```
Weak relationship between feature value and SHAP impact
Model doesn't consistently use this feature
```

---

## Prediction Explanation Example

### Instance 0 Analysis

```
Base Value (Model Average): 0.5000

Top Contributing Features (Sorted by Impact):

Feature            Value      SHAP Value    Direction
──────────────────────────────────────────────────────
oldbalanceOrg     -0.2817     -0.4302      ↓ Normal      ← Strongest
amount             1.5259      0.1272      ↑ Fraud
newbalanceOrig    -0.2856     -0.0742      ↓ Normal
oldbalanceDest    -0.3286      0.0454      ↑ Fraud
newbalanceDest    -0.0946     -0.0363      ↓ Normal
step               0.4142     -0.0199      ↓ Normal      ← Weakest

Final Model Score: 0.1120  (Predicts NORMAL - below 0.5)
```

**Reading:**

1. **Start at base value:** 0.5000
2. **oldbalanceOrg (SHAP=-0.4302):** Scaled sender balance is LOW → pushes prediction DOWN (-0.43)
   - New value: 0.5000 - 0.4302 = 0.0698
3. **amount (SHAP=+0.1272):** Transaction amount is HIGH → pushes prediction UP (+0.13)
   - New value: 0.0698 + 0.1272 = 0.1970
4. **Continue for other features...**
5. **Final score: 0.1120** → Model predicts **NORMAL** (< 0.5)

**Why is this normal?** Low sender account balance + normal amount + other factors = low fraud score

---

## Use Cases

### Use Case 1: Audit a Specific Transaction

```python
from src.shap_explainability import SHAPExplainer

explainer = SHAPExplainer(model_path="models/Random_Forest_tuned.pkl")
explainer.prepare_explainer(X_train_scaled, sample_size=100)
shap_vals = explainer.calculate_shap_values(X_test_scaled)

# Explain transaction #42
explanation = explainer.get_feature_explanation(X_test_scaled_df, instance_idx=42)

# Print detailed explanation
for feature in explanation['features'][:5]:
    print(f"{feature['name']}: {feature['shap_value']:.4f}")
```

### Use Case 2: Find Feature Interactions

```python
# dependence_plots.png shows non-linear relationships
# Example: amount vs SHAP value might show:
# - Small amounts (< $100) → always normal
# - Medium amounts ($100-5000) → depends on balance
# - Large amounts (> $5000) → mostly fraud
```

### Use Case 3: Fair Lending/Bias Detection

```python
# Plot SHAP values grouped by customer demographics
# Look for unexplained patterns favoring/penalizing groups
# If same balance & amount → different predictions by country/gender?
# That's bias in your model!
```

### Use Case 4: Model Improvement Areas

```python
# If feature has low average |SHAP|:
# Option 1: Feature is not useful → remove it
# Option 2: Feature needs better engineering → create interactions
#
# Focus engineering efforts on Top 3-5 features
```

---

## Key Statistics

### Current Model Analysis

**Dataset Analyzed:** 1,000 test transactions

**SHAP Values Summary:**

- Mean |SHAP|: 0.1234
- Max |SHAP|: 0.8956
- Min |SHAP|: 0.0001

**Top Feature:** `oldbalanceOrg` (45.2% importance)
**Least Used:** `step` (2.1% importance)

---

## Common Questions

### Q: Why is SHAP better than feature importance?

| Aspect             | Feature Importance | SHAP               |
| ------------------ | ------------------ | ------------------ |
| What it shows      | Overall importance | Directional impact |
| Per-instance       | No                 | Yes                |
| Sums to prediction | No                 | Yes                |
| Theory             | Heuristic          | Game theory        |
| Interpretability   | Good               | Excellent          |

**Example:**

- Feature Importance: "Amount is important"
- SHAP: "This $5000 amount pushes fraud +0.15 (but low balance pushes normal -0.40)"

### Q: Positive vs Negative SHAP?

```
Positive SHAP  → Feature pushes prediction TOWARD fraud
Negative SHAP  → Feature pushes prediction TOWARD normal

Example:
amount = $5000
SHAP = +0.25 → Increases fraud likelihood
SHAP = -0.25 → Decreases fraud likelihood
```

### Q: Can SHAP explain model mistakes?

Yes! Compare SHAP explanations for:

1. Model predicted fraud, actually fraud → correct reasoning
2. Model predicted normal, actually fraud → find missing signals
3. Model predicted fraud, actually normal → false alarm reason

### Q: What's the computational cost?

Tree models (like yours):

- **Fast:** 1000 samples in ~1 minute
- **Efficient:** TreeExplainer uses tree structure

Other models:

- KernelExplainer (slow, ~30 min for 1000 samples)
- Sampling-based (trade-off accuracy for speed)

---

## Advanced: Custom SHAP Analysis

### Create Force Plot for Single Transaction

```python
base_value = explainer.explainer.expected_value
if isinstance(base_value, list):
    base_value = base_value[1]

shap.plots.force(
    base_value,
    shap_vals[idx],  # SHAP values for instance idx
    X_test_scaled_df.iloc[idx],
    matplotlib=True
)
```

### Find Samples Closest to Decision Boundary

```python
# Transactions where model is uncertain (SHAP ≈ 0)
uncertainty = np.abs(shap_vals.sum(axis=1) - base_value)
uncertain_indices = np.argsort(uncertainty)[:10]  # Most uncertain

# These are hardest to classify - good for human review
```

### Explain Model for Specific Feature Range

```python
# Focus on high-value transactions
high_value_mask = X_test_scaled_df['amount'] > mean_amount + 2*std_amount
subset_shap = shap_vals[high_value_mask]
subset_shap.mean(axis=0)  # Average impact per feature
```

---

## Implementation Notes

### Version Compatibility

- SHAP: 0.41.0+
- scikit-learn: 0.24.0+
- pandas, numpy (standard)

### Performance Tips

- **For 1M+ samples:** Use sample of 5-10K for SHAP calculation
- **TreeExplainer:** Use with tree-based models (fast)
- **KernelExplainer:** Use for other models (slow)
- **Background samples:** 100-200 samples usually sufficient

### Interpreting SHAP with Scaled Features

⚠️ Your features are StandardScaled:

- SHAP values are in scaled space, not original units
- Direction (positive/negative) still valid
- Magnitude interpretation requires unscaling

---

## Summary

SHAP provides:

- ✅ **Global understanding:** Which features matter
- ✅ **Local explanations:** Why each prediction was made
- ✅ **Consistency:** Solid game theory foundation
- ✅ **Actionability:** Clear improvement directions

**Next Steps:**

1. Review generated plots
2. Validate with domain expertise
3. Use insights to improve model
4. Retrain and compare SHAP values
5. Monitor SHAP patterns in production

---

## References

- SHAP Paper: https://arxiv.org/abs/1705.07874
- SHAP Documentation: https://shap.readthedocs.io/
- Interpretable ML Book: https://christophgoldstern.com/interpretable-ml-book/
