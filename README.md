# Customer Segmentation (K-Means) & Bike Rental EDA

- **Customer Segmentation**: Load data, clean, engineer features, run EDA, train K-Means, and serve predictions via a Streamlit app.  
  Based on: [Customer Segmentation Using Machine Learning - Data Science with Onur](https://youtu.be/-LGwdrajMZ0)
- **Bike Rental EDA**: Exploratory analysis and visualizations for bike-sharing hourly/daily datasets (UCI).

---

## Clone and run (e.g. from GitHub)

```bash
git clone https://github.com/YOUR_USERNAME/customer-segmentation.git
cd customer-segmentation
pip install -r requirements.txt
```

Then open `Analysis_Model.ipynb`, choose your Python kernel, **Run All**, and finally:

```bash
streamlit run segmentation.py
```

*(Replace the clone URL with your repo. Model files `*.pickle` are not in the repo; the notebook creates them when you run it.)*

### Bike Rental EDA (optional)

1. Download the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) and place `hour.csv` and `day.csv` in a `data/` folder in the project root.
2. Run: `python bike_rental_eda.py`  
   Plots will open in sequence; close each window to continue.

---

## Run in VS Code

1. **Open the project folder**  
   `File → Open Folder` → choose the `Customer Segmentation` folder.

2. **Select the Python interpreter**  
   `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) → **Python: Select Interpreter** → pick the environment where you installed the dependencies (e.g. a venv or Conda env).

3. **Install dependencies** (once)  
   Open the integrated terminal (`Ctrl+`` ` or **Terminal → New Terminal**). The terminal will start in the project folder. Run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model (notebook)**  
   - Open `Analysis_Model.ipynb`.  
   - In the notebook toolbar, choose the same Python kernel you selected above.  
   - Use **Run All** (or run cells from top to bottom).  
   - When it finishes, `kmeans_model.pickle` and `scaler.pickle` will appear in the project folder.

5. **Run the Streamlit app**  
   In the same terminal (project folder):

   ```bash
   streamlit run segmentation.py
   ```

   Then open http://localhost:8501 in your browser.

If the model files are missing, the app will show a clear message asking you to run the notebook first.

---

## Setup (summary)

| Step | Action |
|------|--------|
| 1 | Open folder in VS Code |
| 2 | Select Python interpreter (venv/Conda) |
| 3 | `pip install -r requirements.txt` |
| 4 | Open `Analysis_Model.ipynb` → select kernel → **Run All** |
| 5 | `streamlit run segmentation.py` |

Ensure `customer_segmentation.csv` is in this folder (it is included).

---

## Files

| File | Purpose |
|------|---------|
| `customer_segmentation.csv` | Dataset (Kaggle customer personality) |
| `Analysis_Model.ipynb` | EDA, feature engineering, K-Means, PCA, export model |
| `segmentation.py` | Streamlit app for segment prediction |
| `bike_rental_eda.py` | Bike-sharing EDA script (needs `data/hour.csv`, `data/day.csv`) |
| `kmeans_model.pickle` | Trained K-Means model (created after running notebook) |
| `scaler.pickle` | Fitted StandardScaler (created after running notebook) |
| `data/` | Optional folder for bike rental CSVs (`hour.csv`, `day.csv`) |
| `.vscode/` | VS Code settings and recommended extensions |

---

## Dataset

[Kaggle: Customer Segmentation Clustering](https://www.kaggle.com/datasets/vishakhdapat/customer-segmentation-clustering).  
Place `customer_segmentation.csv` in the project root (or it may already be there if included in the repo).

---

## License

MIT — see [LICENSE](LICENSE).
"# Customer-Segmentation-Streamlit-app-" 
