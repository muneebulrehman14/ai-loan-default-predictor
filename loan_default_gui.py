
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')


class LoanDefaultPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Loan Default Prediction System - Rural Pakistan Microfinance")
        self.root.geometry("1000x900")
        self.root.configure(bg='#f0f4f8')

        # Colors - Light professional theme
        self.bg_dark = '#f0f4f8'
        self.bg_panel = '#ffffff'
        self.accent_blue = '#1d6db5'
        self.accent_pink = '#c0446b'
        self.accent_green = '#1a7f5a'
        self.text_light = '#1e293b'
        self.text_gray = '#4a5568'
        self.border_color = '#cbd5e1'
        self.btn_fg = '#ffffff'

        # Models and data
        self.models = {}
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None
        self.df = None
        self.training_limit = None   # set by startup dialog

        # Pakistani microfinance encoders
        self.le_occupation = LabelEncoder()
        self.le_region = LabelEncoder()
        self.le_loan_purpose = LabelEncoder()

        # Show startup config dialog BEFORE loading data
        self.root.withdraw()           # hide main window until ready
        self._show_startup_dialog()    # blocks until user clicks Start

    # ─────────────────────────────────────────────────────────────────────
    def _show_startup_dialog(self):
        """Splash / config dialog: let user choose training data limit."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Configure Training Data")
        dialog.geometry("520x540")
        dialog.resizable(False, True)
        dialog.configure(bg='#f0f4f8')
        dialog.grab_set()

        # Center dialog on screen
        dialog.update_idletasks()
        sw = dialog.winfo_screenwidth()
        sh = dialog.winfo_screenheight()
        x  = (sw - 520) // 2
        y  = (sh - 540) // 2
        dialog.geometry(f"520x540+{x}+{y}")

        # ── Header ───────────────────────────────────────────────────
        hdr = tk.Frame(dialog, bg='#1d6db5')
        hdr.pack(fill='x')
        tk.Label(hdr, text="🏦  Loan Default Prediction System",
                 font=('Segoe UI', 15, 'bold'),
                 bg='#1d6db5', fg='#ffffff').pack(pady=(14, 2))
        tk.Label(hdr, text="Microfinance Borrowers in Rural Pakistan",
                 font=('Segoe UI', 10), bg='#1d6db5', fg='#d0e8ff').pack(pady=(0, 14))

        # ── Bottom bar — always visible ───────────────────────────────
        # (packed before body so side='bottom' works correctly in tkinter)

        # ── Body ─────────────────────────────────────────────────────
        body = tk.Frame(dialog, bg='#f0f4f8')
        body.pack(fill='both', expand=True, padx=24, pady=(16, 8))

        tk.Label(body, text="Select Training Data Size",
                 font=('Segoe UI', 12, 'bold'),
                 bg='#f0f4f8', fg='#1e293b').pack(anchor='w')
        tk.Label(body,
                 text="More records = higher accuracy but slower startup.\n"
                      "Your dataset contains 255,347 records.",
                 font=('Segoe UI', 9), bg='#f0f4f8', fg='#4a5568',
                 justify='left').pack(anchor='w', pady=(2, 12))

        # Radio options
        OPTIONS = [
            ("10,000  records  — Fast startup  (~10 sec)",    10_000),
            ("20,000  records  — Balanced  (default)",        20_000),
            ("50,000  records  — Better accuracy  (~45 sec)", 50_000),
            ("100,000 records  — High accuracy  (~90 sec)",  100_000),
            ("Full dataset  (255,347)  — Best accuracy  (~3 min)", None),
            ("Custom …",                                       -1),
        ]
        self._limit_var = tk.IntVar(value=20_000)

        radio_frame = tk.Frame(body, bg='#ffffff', relief='groove', bd=1)
        radio_frame.pack(fill='x', pady=(0, 10))

        for i, (label, val) in enumerate(OPTIONS):
            int_val = val if val is not None else 0   # 0 = full dataset sentinel
            if val == -1:
                int_val = -1
            r = tk.Radiobutton(radio_frame, text=label,
                               variable=self._limit_var, value=int_val,
                               font=('Segoe UI', 10),
                               bg='#ffffff', fg='#1e293b',
                               activebackground='#e0eeff',
                               selectcolor='#dbeafe',
                               anchor='w')
            r.pack(fill='x', padx=16, pady=3)

        # Custom entry row
        custom_row = tk.Frame(body, bg='#f0f4f8')
        custom_row.pack(fill='x', pady=(0, 4))
        tk.Label(custom_row, text="Custom value:",
                 font=('Segoe UI', 9), bg='#f0f4f8', fg='#4a5568').pack(side='left')
        self._custom_entry = tk.Entry(custom_row, font=('Segoe UI', 10),
                                      bg='#f1f5f9', fg='#1e293b',
                                      relief='solid', bd=1, width=14)
        self._custom_entry.pack(side='left', padx=8)
        tk.Label(custom_row, text="(1 – 255,347)",
                 font=('Segoe UI', 9), bg='#f0f4f8', fg='#94a3b8').pack(side='left')

        # ── Status + Start button — pinned at bottom ──────────────────
        self._dlg_status = tk.Label(body, text="",
                                    font=('Segoe UI', 9, 'italic'),
                                    bg='#f0f4f8', fg='#c0392b')
        self._dlg_status.pack(anchor='w', pady=(8, 4))

        # ── Start action ─────────────────────────────────────────────
        def on_start():
            chosen = self._limit_var.get()
            if chosen == -1:          # custom
                raw = self._custom_entry.get().strip()
                try:
                    chosen = int(raw.replace(',', ''))
                    if not (1 <= chosen <= 255_347):
                        raise ValueError
                except ValueError:
                    self._dlg_status.config(
                        text="⚠  Enter a whole number between 1 and 255,347")
                    return
            self.training_limit = chosen if chosen != 0 else None   # 0 = full
            dialog.destroy()
            self._launch_app()

        tk.Button(body, text="▶   Start System",
                  font=('Segoe UI', 11, 'bold'),
                  bg='#1d6db5', fg='#ffffff',
                  activebackground='#155fa0', activeforeground='#ffffff',
                  relief='flat', cursor='hand2',
                  padx=24, pady=10,
                  command=on_start).pack(fill='x', pady=(4, 12))

        # If user closes the dialog window, quit the app
        dialog.protocol("WM_DELETE_WINDOW", self.root.destroy)
        dialog.wait_window()

    # ─────────────────────────────────────────────────────────────────────
    def _launch_app(self):
        """Called after startup dialog closes — load data, train, build UI."""
        self.root.deiconify()

        # Show loading screen while training
        loading = tk.Toplevel(self.root)
        loading.title("Loading…")
        loading.geometry("360x140")
        loading.resizable(False, False)
        loading.configure(bg='#ffffff')
        loading.grab_set()
        loading.update_idletasks()
        sw = loading.winfo_screenwidth()
        sh = loading.winfo_screenheight()
        loading.geometry(f"360x140+{(sw-360)//2}+{(sh-140)//2}")

        limit_text = (f"{self.training_limit:,} records"
                      if self.training_limit else "all 255,347 records")
        tk.Label(loading, text="⏳  Training Models…",
                 font=('Segoe UI', 13, 'bold'),
                 bg='#ffffff', fg='#1d6db5').pack(pady=(22, 6))
        status_lbl = tk.Label(loading,
                              text=f"Loading {limit_text} — please wait",
                              font=('Segoe UI', 9), bg='#ffffff', fg='#4a5568')
        status_lbl.pack()
        prog = ttk.Progressbar(loading, mode='indeterminate', length=300)
        prog.pack(pady=14)
        prog.start(12)
        loading.update()

        # Load & train
        self.load_data()
        self.train_models()

        prog.stop()
        loading.destroy()
        self.create_widgets()
    
    def load_data(self):
        """Load and transform CSV to Pakistani microfinance format"""
        try:
            csv_path = r'd:\airespap\Loan_default.csv\Loan_default.csv'
            df = pd.read_csv(csv_path)
            print(f"Loaded dataset: {len(df)} records")
            print(f"Columns: {list(df.columns)}")
            
            # Check if already in Pakistani format (lowercase columns)
            pakistani_columns = ['age', 'monthly_income', 'loan_amount', 'occupation', 'region', 'loan_purpose', 'default']
            if all(col in df.columns for col in pakistani_columns):
                print("CSV already in Pakistani format - using directly")
                self.df = df.copy()
                # Add LoanID if not present
                if 'LoanID' not in self.df.columns:
                    self.df.insert(0, 'LoanID', [f'PK{i:06d}' for i in range(len(self.df))])
            else:
                print("Converting to Pakistani microfinance format...")
                np.random.seed(42)
                n = len(df)
                
                # Transform to Pakistani format
                self.df = pd.DataFrame({
                    'LoanID': df['LoanID'] if 'LoanID' in df.columns else [f'PK{i:06d}' for i in range(n)],
                    'age': df['Age'].clip(18, 75).values if 'Age' in df.columns else np.random.randint(18, 75, n),
                    'monthly_income': (df['Income'] / 12).clip(8000, 200000).astype(int).values if 'Income' in df.columns else np.random.randint(15000, 150000, n),
                    'loan_amount': df['LoanAmount'].clip(10000, 600000).astype(int).values if 'LoanAmount' in df.columns else np.random.randint(10000, 500000, n),
                    'loan_term_months': df['LoanTerm'].clip(12, 36).values if 'LoanTerm' in df.columns else np.random.choice([12, 18, 24, 36], n),
                    'num_dependents': np.random.poisson(3, n).clip(0, 8),
                    'distance_to_bank_km': np.random.exponential(8, n).clip(0.5, 50).round(1),
                    'previous_loans': df['NumCreditLines'].clip(0, 5).values if 'NumCreditLines' in df.columns else np.random.poisson(1, n).clip(0, 5),
                    'credit_score': df['CreditScore'].clip(300, 850).values if 'CreditScore' in df.columns else np.random.normal(550, 80, n).clip(300, 850).astype(int),
                    'education_years': np.random.choice([0, 5, 8, 10, 12, 14, 16], n, p=[0.15, 0.15, 0.2, 0.15, 0.2, 0.1, 0.05]),
                    'has_collateral': np.random.choice([0, 1], n, p=[0.7, 0.3]),
                    'occupation': np.random.choice(['Farmer', 'Shopkeeper', 'Laborer', 'Artisan', 'Teacher', 'No job', 'Pensioner', 'Student'], n, p=[0.23, 0.18, 0.18, 0.14, 0.09, 0.05, 0.05, 0.08]),
                    'region': np.random.choice(['Punjab', 'Sindh', 'KPK', 'Balochistan'], n, p=[0.5, 0.3, 0.15, 0.05]),
                    'loan_purpose': np.random.choice(['Agriculture', 'Business', 'Education', 'Home Repair', 'Medical'], n),
                })
                
                # Apply occupation-specific income adjustments
                self.df.loc[self.df['occupation'] == 'No job', 'monthly_income'] = np.random.exponential(6000, (self.df['occupation'] == 'No job').sum()).clip(0, 12000).astype(int)
                self.df.loc[self.df['occupation'] == 'Pensioner', 'monthly_income'] = np.random.normal(18000, 4000, (self.df['occupation'] == 'Pensioner').sum()).clip(8000, 35000).astype(int)
                self.df.loc[self.df['occupation'] == 'Student', 'monthly_income'] = np.random.exponential(8000, (self.df['occupation'] == 'Student').sum()).clip(0, 20000).astype(int)
                
                # Calculate default probability
                default_prob = 0.15 - 0.0001 * (self.df['monthly_income'] - 30000) / 1000
                default_prob[self.df['credit_score'] < 400] += 0.12
                default_prob[self.df['credit_score'] > 700] -= 0.08
                default_prob[self.df['has_collateral'] == 1] -= 0.05
                default_prob[self.df['previous_loans'] >= 3] += 0.08  # 3+ loans = high risk
                default_prob[self.df['previous_loans'] >= 2] += 0.04  # 2+ loans = medium risk
                default_prob[self.df['previous_loans'] == 0] -= 0.02  # No history = slightly risky
                default_prob[self.df['education_years'] < 5] += 0.05
                
                # Age risk - retirement in Pakistan is 50-60
                default_prob[self.df['age'] >= 50] += 0.03  # Near retirement
                default_prob[self.df['age'] >= 55] += 0.05  # Retirement age
                default_prob[self.df['age'] >= 60] += 0.08  # Post-retirement
                default_prob[self.df['age'] >= 65] += 0.12  # High age risk
                
                # Age + No job = extremely high risk (no income source)
                default_prob[(self.df['age'] >= 55) & (self.df['occupation'] == 'No job')] += 0.15
                default_prob[(self.df['age'] >= 60) & (self.df['occupation'] == 'No job')] += 0.25
                
                # Age + Occupation specific risks (physical jobs and retirement)
                # Laborers - physical work, harder with age
                default_prob[(self.df['age'] >= 50) & (self.df['occupation'] == 'Laborer')] += 0.08
                default_prob[(self.df['age'] >= 55) & (self.df['occupation'] == 'Laborer')] += 0.12
                
                # Farmers - physical work, land dependent
                default_prob[(self.df['age'] >= 55) & (self.df['occupation'] == 'Farmer')] += 0.06
                default_prob[(self.df['age'] >= 62) & (self.df['occupation'] == 'Farmer')] += 0.10
                
                # Artisans - skill based but physical
                default_prob[(self.df['age'] >= 58) & (self.df['occupation'] == 'Artisan')] += 0.05
                
                # Shopkeepers - less physical, can work longer
                default_prob[(self.df['age'] >= 60) & (self.df['occupation'] == 'Shopkeeper')] += 0.03
                
                # Teachers - less physical but retire
                default_prob[(self.df['age'] >= 60) & (self.df['occupation'] == 'Teacher')] += 0.04
                
                # Students - older students (30+) are riskier
                default_prob[(self.df['age'] >= 30) & (self.df['occupation'] == 'Student')] += 0.08
                
                # Regional risk
                for reg, risk in {'Punjab': -0.02, 'Sindh': 0.01, 'KPK': 0.03, 'Balochistan': 0.05}.items():
                    default_prob[self.df['region'] == reg] += risk
                
                # Purpose risk
                for purp, risk in {'Agriculture': 0.02, 'Business': -0.01, 'Education': -0.03, 'Home Repair': 0.01, 'Medical': 0.03}.items():
                    default_prob[self.df['loan_purpose'] == purp] += risk
                
                # Occupation base risk
                for occ, risk in {'Farmer': 0.01, 'Shopkeeper': -0.02, 'Laborer': 0.02, 'Artisan': 0.0, 'Teacher': -0.03, 'No job': 0.15, 'Pensioner': 0.05, 'Student': 0.10}.items():
                    default_prob[self.df['occupation'] == occ] += risk
                
                default_prob = default_prob.clip(0.05, 0.95)
                self.df['default'] = (np.random.rand(n) < default_prob).astype(int)
                
                # Save back to CSV
                self.df.to_csv(csv_path, index=False)
                print(f"Saved transformed CSV with {len(self.df)} records")
            
            # Apply user-chosen training data limit
            if self.training_limit is not None:
                n_samples = min(self.training_limit, len(self.df))
                if n_samples < len(self.df):
                    self.df = self.df.sample(n=n_samples, random_state=42).reset_index(drop=True)
            # else: training_limit is None → use full dataset (no sampling)
            
            print(f"Training on: {len(self.df)} records")
            print(f"No Default: {(self.df['default']==0).sum()}, Default: {(self.df['default']==1).sum()}")
            print(f"Default rate: {self.df['default'].mean():.1%}")
            
        except Exception as e:
            print(f"Failed to load CSV: {str(e)}")
            self.df = self._create_fallback_data()
    
    def _create_fallback_data(self):
        """Create Pakistani microfinance synthetic data if CSV fails"""
        np.random.seed(42)
        N = 5000
        occupations = ['Farmer', 'Shopkeeper', 'Laborer', 'Artisan', 'Teacher', 'No job', 'Pensioner', 'Student']
        regions = ['Punjab', 'Sindh', 'KPK', 'Balochistan']
        loan_purposes = ['Agriculture', 'Business', 'Education', 'Home Repair', 'Medical']
        
        df = pd.DataFrame({
            'LoanID': [f'PK{i:06d}' for i in range(N)],
            'age': np.random.randint(18, 65, N),
            'monthly_income': np.random.randint(8000, 80000, N),
            'loan_amount': np.random.randint(5000, 200000, N),
            'loan_term_months': np.random.choice([6, 12, 18, 24, 36], N),
            'num_dependents': np.random.randint(0, 8, N),
            'distance_to_bank_km': np.random.randint(1, 100, N),
            'previous_loans': np.random.randint(0, 5, N),
            'credit_score': np.random.randint(300, 850, N),
            'education_years': np.random.randint(0, 16, N),
            'has_collateral': np.random.choice([0, 1], N, p=[0.6, 0.4]),
            'occupation': np.random.choice(occupations, N),
            'region': np.random.choice(regions, N),
            'loan_purpose': np.random.choice(loan_purposes, N),
        })
        
        # Realistic default logic
        default_prob = (
            0.35
            - 0.0004 * df['monthly_income'] / 1000
            - 0.0003 * df['credit_score']
            + 0.001 * df['loan_amount'] / 1000
            + 0.025 * df['num_dependents']
            - 0.08 * df['has_collateral']
            + 0.002 * df['distance_to_bank_km']
            + 0.03 * (df['previous_loans'] >= 2)  # 2+ previous loans = higher risk
            + 0.06 * (df['previous_loans'] >= 3)  # 3+ previous loans = much higher risk
            + 0.03 * (df['age'] >= 50)  # Near retirement
            + 0.05 * (df['age'] >= 55)  # Retirement age
            + 0.08 * (df['age'] >= 60)  # Post-retirement
            + 0.15 * ((df['age'] >= 55) & (df['occupation'] == 'No job'))  # Old + no job = very risky
            + 0.08 * ((df['age'] >= 50) & (df['occupation'] == 'Laborer'))  # Old laborer
            + 0.06 * ((df['age'] >= 55) & (df['occupation'] == 'Farmer'))  # Old farmer
            + 0.05 * ((df['age'] >= 58) & (df['occupation'] == 'Artisan'))  # Old artisan
            + 0.08 * ((df['age'] >= 30) & (df['occupation'] == 'Student'))  # Older student
            - 0.01 * df['education_years']
        )
        
        # Add occupation, region, purpose risk
        for occ, risk in {'Farmer': 0.05, 'Shopkeeper': -0.02, 'Laborer': 0.03, 'Artisan': -0.01, 'Teacher': -0.05, 'No job': 0.15, 'Pensioner': 0.05, 'Student': 0.10}.items():
            default_prob[df['occupation'] == occ] += risk
        for reg, risk in {'Punjab': -0.02, 'Sindh': 0.01, 'KPK': 0.03, 'Balochistan': 0.04}.items():
            default_prob[df['region'] == reg] += risk
        for purp, risk in {'Agriculture': 0.02, 'Business': -0.01, 'Education': -0.03, 'Home Repair': 0.01, 'Medical': 0.03}.items():
            default_prob[df['loan_purpose'] == purp] += risk
        
        default_prob = default_prob.clip(0.05, 0.95)
        df['default'] = (np.random.rand(N) < default_prob).astype(int)
        
        # Add debt-to-income ratio
        df['debt_to_income'] = df['loan_amount'] / df['monthly_income'].replace(0, 1)
        df['debt_to_income'] = df['debt_to_income'].clip(0, 100)
        
        return df
    
    def train_models(self):
        """Train models on transformed Pakistani microfinance data"""
        # Use the transformed dataframe from load_data()
        df_train = self.df.drop('LoanID', axis=1) if 'LoanID' in self.df.columns else self.df.copy()
        
        print(f"Training on {len(df_train)} records")
        print(f"Class distribution - No Default: {(df_train['default']==0).sum()}, Default: {(df_train['default']==1).sum()}")
        
        # Separate features and target
        X = df_train.drop('default', axis=1)
        y = df_train['default']
        
        # Encode categorical variables (Pakistani fields)
        for col, encoder in [('occupation', self.le_occupation), 
                             ('region', self.le_region), 
                             ('loan_purpose', self.le_loan_purpose)]:
            X[col] = encoder.fit_transform(X[col])
        
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Apply SMOTE for class balancing (used by RF and XGBoost)
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        
        print(f"Training on {len(X_train_bal)} samples after SMOTE")
        
        # Scale features for Logistic Regression
        # FIX: Scale original (unbalanced) training data for LR — LR uses class_weight='balanced'
        self.scaler = StandardScaler()
        X_train_sc = self.scaler.fit_transform(X_train)  # FIXED: use X_train not X_train_bal
        
        # Train models (reduced n_estimators for speed)
        print("Training Random Forest...")
        self.models['Random Forest'] = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        self.models['Random Forest'].fit(X_train_bal, y_train_bal)
        
        # FIX: Use class_weight='balanced' for LR instead of SMOTE-balanced data.
        # Training LR on SMOTE data caused it to heavily over-predict DEFAULT (only 58.6% accuracy).
        print("Training Logistic Regression...")
        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=500, random_state=42, class_weight='balanced')  # FIXED
        self.models['Logistic Regression'].fit(X_train_sc, y_train)  # FIXED: use y_train
        
        print("Training XGBoost...")
        # Calculate scale_pos_weight = (negative_samples / positive_samples)
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
        self.models['XGBoost'] = XGBClassifier(n_estimators=50, random_state=42,
                                                scale_pos_weight=scale_pos_weight,
                                                eval_metric='logloss', verbosity=0)
        self.models['XGBoost'].fit(X_train_bal, y_train_bal)
        
        print("Models trained successfully!")
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Header
        header_frame = tk.Frame(self.root, bg='#1d6db5')
        header_frame.pack(fill='x', padx=0, pady=0)

        
        title = tk.Label(header_frame,
                        text="Loan Default Prediction System",
                        font=('Segoe UI', 20, 'bold'),
                        bg='#1d6db5', fg='#ffffff')
        title.pack(pady=(16, 2))

        subtitle = tk.Label(header_frame,
                           text="Microfinance Borrowers in Rural Pakistan",
                           font=('Segoe UI', 11),
                           bg='#1d6db5', fg='#d0e8ff')
        subtitle.pack()

        limit_info = (f"Training on {len(self.df):,} records"
                      if self.training_limit else f"Training on full dataset ({len(self.df):,} records)")
        data_source = tk.Label(header_frame,
                           text=limit_info,
                           font=('Segoe UI', 9, 'italic'),
                           bg='#1d6db5', fg='#a3d4ff')
        data_source.pack(pady=(2, 16))
        
        # ── Notebook (tab container) ────────────────────────────────────────────
        nb_style = ttk.Style()
        nb_style.theme_use('default')
        nb_style.configure('TNotebook', background='#f0f4f8', borderwidth=0)
        nb_style.configure('TNotebook.Tab', font=('Segoe UI', 10, 'bold'),
                        padding=[14, 6], background='#dde6f0', foreground='#4a5568')
        nb_style.map('TNotebook.Tab',
                  background=[('selected', '#1d6db5')],
                  foreground=[('selected', '#ffffff')])

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=0, pady=0)

        tab_predict = tk.Frame(notebook, bg='#f0f4f8')
        notebook.add(tab_predict, text='  🏦  Loan Predictor  ')

        tab_compare = tk.Frame(notebook, bg='#f0f4f8')
        notebook.add(tab_compare, text='  📊  Region × Occupation Analysis  ')

        self._build_predictor_tab(tab_predict)
        self._build_compare_tab(tab_compare)

        footer = tk.Label(self.root,
                         text="CSC 412 - Artificial Intelligence  |  Bahria University  |  Spring 2026",
                         font=('Segoe UI', 9),
                         bg='#dde6f0', fg='#4a5568',
                         pady=6)
        footer.pack(fill='x', pady=0)

    # ─────────────────────────────────────────────────────────────────────
    def _build_predictor_tab(self, parent):
        """Original predictor UI inside Tab 1"""
        # Main content frame with scrollable left panel
        main_frame = tk.Frame(parent, bg='#f0f4f8')
        main_frame.pack(fill='both', expand=True, padx=15, pady=12)

        left_container = tk.Frame(main_frame, bg='#f0f4f8')
        left_container.pack(side='left', fill='both', expand=True, padx=(0, 8))

        canvas = tk.Canvas(left_container, bg='#ffffff', highlightthickness=1,
                           highlightbackground='#cbd5e1')
        scrollbar = ttk.Scrollbar(left_container, orient='vertical', command=canvas.yview)
        input_frame = tk.LabelFrame(canvas, text="  Borrower Information  ",
                                   font=('Segoe UI', 12, 'bold'),
                                   bg='#ffffff', fg='#1d6db5',
                                   padx=15, pady=15,
                                   relief='groove', bd=2)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        canvas_window = canvas.create_window((0, 0), window=input_frame, anchor='nw', tags="input_frame")
        
        # Create input fields (Pakistani Microfinance themed)
        self.inputs = {}
        row = 0
        
        # Personal Info
        self.create_input_field(input_frame, "Age (18-65):", row, 'age', 18, 65, 35)
        row += 1
        self.create_input_field(input_frame, "Monthly Income (PKR):", row, 'monthly_income', 8000, 80000, 25000)
        row += 1
        self.create_input_field(input_frame, "Loan Amount (PKR):", row, 'loan_amount', 5000, 200000, 50000)
        row += 1
        self.create_combobox(input_frame, "Loan Term:", row, 'loan_term_months', 
                           ['6', '12', '18', '24', '36'], '12')
        row += 1
        
        # Family & Location
        self.create_input_field(input_frame, "Number of Dependents:", row, 'num_dependents', 0, 8, 2)
        row += 1
        self.create_input_field(input_frame, "Distance to Bank (km):", row, 'distance_to_bank_km', 1, 100, 10)
        row += 1
        
        # Credit History
        self.create_input_field(input_frame, "Previous Loans:", row, 'previous_loans', 0, 5, 0)
        row += 1
        self.create_input_field(input_frame, "Credit Score (300-850):", row, 'credit_score', 300, 850, 550)
        row += 1
        
        # Categorical Pakistani fields
        occupations = ['Farmer', 'Shopkeeper', 'Laborer', 'Artisan', 'Teacher', 'No job', 'Pensioner', 'Student']
        self.le_occupation.fit(occupations)
        self.create_combobox(input_frame, "Occupation:", row, 'occupation', occupations, occupations[0])
        row += 1
        
        regions = ['Punjab', 'Sindh', 'KPK', 'Balochistan']
        self.le_region.fit(regions)
        self.create_combobox(input_frame, "Region:", row, 'region', regions, regions[0])
        row += 1
        
        loan_purposes = ['Agriculture', 'Business', 'Education', 'Home Repair', 'Medical']
        self.le_loan_purpose.fit(loan_purposes)
        self.create_combobox(input_frame, "Loan Purpose:", row, 'loan_purpose', loan_purposes, loan_purposes[0])
        row += 1
        
        # Education & Collateral
        self.create_input_field(input_frame, "Education Years (0-16):", row, 'education_years', 0, 16, 10)
        row += 1
        self.create_combobox(input_frame, "Has Collateral:", row, 'has_collateral', 
                           ['No', 'Yes'], 'No')
        row += 1
        
        # Update scroll region
        def configure_canvas(event):
            canvas.configure(scrollregion=canvas.bbox('all'))
            canvas.itemconfig(canvas_window, width=event.width)
        input_frame.bind('<Configure>', configure_canvas)
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(canvas_window, width=e.width))
        
        # Predict Button
        predict_btn = tk.Button(left_container, text="⚡  Predict Default Risk",
                               font=('Segoe UI', 12, 'bold'),
                               bg='#1d6db5', fg='#ffffff',
                               activebackground='#155fa0', activeforeground='#ffffff',
                               relief='flat', cursor='hand2',
                               padx=20, pady=10,
                               command=self.predict)
        predict_btn.pack(pady=12)
        
        # Right panel - Results
        result_frame = tk.LabelFrame(main_frame, text="  Prediction Results  ",
                                  font=('Segoe UI', 12, 'bold'),
                                  bg='#ffffff', fg='#1d6db5',
                                  padx=15, pady=15,
                                  relief='groove', bd=2)
        result_frame.pack(side='right', fill='both', expand=True, padx=(8, 0))

        self.result_label = tk.Label(result_frame,
                                    text="Enter borrower details and click 'Predict'",
                                    font=('Segoe UI', 11),
                                    bg='#ffffff', fg='#4a5568',
                                    wraplength=350, justify='center')
        self.result_label.pack(pady=15)

        self.model_frame = tk.Frame(result_frame, bg='#f8fafc', relief='sunken', bd=1)
        self.model_frame.pack(fill='both', expand=True, pady=8, padx=5)
        
        self.model_labels = {}
        for model_name, color in [('Random Forest', '#1d6db5'),
                                  ('Logistic Regression', '#c0446b'),
                                  ('XGBoost', '#1a7f5a')]:
            frame = tk.Frame(self.model_frame, bg='#f8fafc')
            frame.pack(fill='x', pady=4, padx=8)

            name_label = tk.Label(frame, text=f"{model_name}:",
                                 font=('Segoe UI', 10, 'bold'),
                                 bg='#f8fafc', fg=color,
                                 width=20, anchor='w')
            name_label.pack(side='left')

            result_text = tk.Label(frame, text="Awaiting input...",
                                  font=('Segoe UI', 10),
                                  bg='#f8fafc', fg='#4a5568',
                                  anchor='w')
            result_text.pack(side='left', padx=10)
            self.model_labels[model_name] = result_text
        
        # Recommendation frame
        self.recommendation_frame = tk.LabelFrame(result_frame, text="  Loan Officer Recommendation  ",
                                                 font=('Segoe UI', 11, 'bold'),
                                                 bg='#ffffff', fg='#1d6db5',
                                                 padx=10, pady=10,
                                                 relief='groove', bd=2)
        self.recommendation_frame.pack(fill='x', pady=12)

        self.recommendation_label = tk.Label(self.recommendation_frame,
                                            text="Complete prediction to see recommendation",
                                            font=('Segoe UI', 10),
                                            bg='#ffffff', fg='#4a5568',
                                            wraplength=350, justify='left')
        self.recommendation_label.pack()
        
        # Risk meter canvas
        self.risk_canvas = tk.Canvas(result_frame, width=350, height=45,
                                     bg='#ffffff', highlightthickness=0)
        self.risk_canvas.pack(pady=8)
        
        # Download results button
        self.download_btn = tk.Button(result_frame, text="📊  Download Results (Graphs)",
                                     font=('Segoe UI', 10, 'bold'),
                                     bg='#1a7f5a', fg='#ffffff',
                                     activebackground='#146b4a', activeforeground='#ffffff',
                                     relief='flat', cursor='hand2',
                                     padx=15, pady=8,
                                     command=self.download_results,
                                     state='disabled')
        self.download_btn.pack(pady=10)
        
        # Store last prediction for download
        self.last_prediction = None
        
        # Store done
    
    def create_input_field(self, parent, label_text, row, field_name, min_val, max_val, default=None):
        """Create a labeled input field with validation"""
        label = tk.Label(parent, text=label_text,
                        font=('Segoe UI', 10),
                        bg='#ffffff', fg='#1e293b',
                        anchor='e', width=20)
        label.grid(row=row, column=0, sticky='e', pady=5, padx=5)
        
        entry = tk.Entry(parent, font=('Segoe UI', 10),
                        bg='#f1f5f9', fg='#1e293b',
                        insertbackground='#1e293b',
                        relief='solid', bd=1, width=25)
        entry.grid(row=row, column=1, sticky='w', pady=5, padx=5)
        
        if default is not None:
            entry.insert(0, str(default))
        else:
            entry.insert(0, str(min_val))
        
        self.inputs[field_name] = {'widget': entry, 'type': 'number', 
                                  'min': min_val, 'max': max_val}
    
    def create_combobox(self, parent, label_text, row, field_name, values, default):
        """Create a labeled combobox"""
        label = tk.Label(parent, text=label_text,
                        font=('Segoe UI', 10),
                        bg='#ffffff', fg='#1e293b',
                        anchor='e', width=20)
        label.grid(row=row, column=0, sticky='e', pady=5, padx=5)
        
        combo = ttk.Combobox(parent, values=values, state='readonly',
                            font=('Segoe UI', 10), width=23)
        combo.set(default)
        combo.grid(row=row, column=1, sticky='w', pady=5, padx=5)
        
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TCombobox', fieldbackground='#f1f5f9',
                       background='#ffffff', foreground='#1e293b',
                       selectbackground='#1d6db5', selectforeground='#ffffff')
        
        self.inputs[field_name] = {'widget': combo, 'type': 'combo', 'values': values}
    
    def validate_inputs(self):
        """Validate all input fields"""
        data = {}
        for field_name, field_info in self.inputs.items():
            if field_info['type'] == 'number':
                try:
                    # Handle float vs int
                    if '.' in str(field_info['min']) or '.' in str(field_info['max']):
                        value = float(field_info['widget'].get())
                    else:
                        value = int(float(field_info['widget'].get()))
                    min_val = field_info['min']
                    max_val = field_info['max']
                    if not (min_val <= value <= max_val):
                        messagebox.showerror("Invalid Input",
                                           f"{field_name} must be between {min_val} and {max_val}")
                        return None
                    data[field_name] = value
                except ValueError:
                    messagebox.showerror("Invalid Input", 
                                       f"{field_name} must be a valid number")
                    return None
            else:
                value = field_info['widget'].get()
                data[field_name] = value
        return data
    
    def encode_categorical(self, data):
        """Encode Pakistani microfinance features for prediction - matches training exactly"""
        encoded = {}
        
        # Numeric features (exact match with interface)
        encoded['age'] = data['age']
        encoded['monthly_income'] = data['monthly_income']
        encoded['loan_amount'] = data['loan_amount']
        encoded['loan_term_months'] = int(data['loan_term_months'])  # FIX: combobox returns str, cast to int
        encoded['num_dependents'] = data['num_dependents']
        encoded['distance_to_bank_km'] = data['distance_to_bank_km']
        encoded['previous_loans'] = data['previous_loans']
        encoded['credit_score'] = data['credit_score']
        encoded['education_years'] = data['education_years']
        encoded['has_collateral'] = 1 if data['has_collateral'] == 'Yes' else 0
        
        # Calculate debt-to-income ratio
        encoded['debt_to_income'] = data['loan_amount'] / data['monthly_income']
        encoded['debt_to_income'] = min(encoded['debt_to_income'], 100)  # Cap at 100
        
        # Categorical features (encode using fitted encoders)
        occupation = data['occupation']
        encoded['occupation'] = self.le_occupation.transform([occupation])[0] if occupation in self.le_occupation.classes_ else 0
        
        region = data['region']
        encoded['region'] = self.le_region.transform([region])[0] if region in self.le_region.classes_ else 0
        
        purpose = data['loan_purpose']
        encoded['loan_purpose'] = self.le_loan_purpose.transform([purpose])[0] if purpose in self.le_loan_purpose.classes_ else 0
        
        return encoded
    
    def predict(self):
        """Make predictions using all models"""
        # Validate inputs
        data = self.validate_inputs()
        if data is None:
            return
        
        # Encode features (Pakistani microfinance -> CSV structure)
        encoded_data = self.encode_categorical(data)
        
        # Prepare feature vector (ensure float32 for XGBoost compatibility)
        features = np.array([[encoded_data[col] for col in self.feature_columns]], dtype=np.float32)
        
        # Get predictions from all models
        results = {}
        
        for name, model in self.models.items():
            if name == 'Logistic Regression':
                features_scaled = self.scaler.transform(features)
                prob = model.predict_proba(features_scaled)[0, 1]
            else:
                prob = model.predict_proba(features)[0, 1]
            
            pred = 1 if prob >= 0.35 else 0
            results[name] = {'prediction': pred, 'probability': prob}
        
        # Update display
        self.update_results(results, data)
        self.update_risk_meter(results)
        
        # Store for download and enable button
        self.last_prediction = {'results': results, 'data': data}
        self.download_btn.config(state='normal')
    
    def update_results(self, results, data):
        """Update the result display"""
        # FIX: Use weighted ensemble — RF and XGBoost outperform LR (~77% vs 58%),
        # so they get higher weight. Simple average was dragging accuracy down.
        model_weights = {'Random Forest': 0.40, 'Logistic Regression': 0.20, 'XGBoost': 0.40}
        avg_prob = sum(
            results[m]['probability'] * model_weights[m] for m in results
        )  # FIXED: weighted average
        ensemble_pred = 1 if avg_prob >= 0.35 else 0
        
        # Update model labels
        for name, result in results.items():
            prob_pct = result['probability'] * 100
            pred_text = "DEFAULT" if result['prediction'] == 1 else "NO DEFAULT"
            color = '#ef4444' if result['prediction'] == 1 else '#22c55e'
            
            self.model_labels[name].config(
                text=f"{pred_text} ({prob_pct:.1f}% risk)",
                fg=color
            )
        
        # Update main result
        if ensemble_pred == 1:
            result_text = "HIGH DEFAULT RISK DETECTED"
            result_color = '#ef4444'
            recommendation = (
                "⚠️ HIGH RISK RECOMMENDATION:\n\n"
                "• Reject loan application OR\n"
                "• Require additional collateral\n"
                "• Request guarantor/co-signer\n"
                "• Reduce loan amount significantly\n"
                "• Increase interest rate to offset risk"
            )
        else:
            result_text = "LOW DEFAULT RISK"
            result_color = '#22c55e'
            recommendation = (
                "✅ LOW RISK RECOMMENDATION:\n\n"
                "• Approve loan application\n"
                "• Standard interest rates apply\n"
                "• Regular repayment schedule\n"
                "• Monitor quarterly for any changes"
            )
        
        self.result_label.config(
            text=f"{result_text}\n(Average Risk: {avg_prob*100:.1f}%)",
            fg=result_color,
            font=('Segoe UI', 14, 'bold')
        )
        
        self.recommendation_label.config(
            text=recommendation,
            fg='#1e293b',
            font=('Segoe UI', 10)
        )
    
    def update_risk_meter(self, results):
        """Update the visual risk meter"""
        self.risk_canvas.delete('all')
        
        model_weights = {'Random Forest': 0.40, 'Logistic Regression': 0.20, 'XGBoost': 0.40}
        avg_prob = sum(results[m]['probability'] * model_weights[m] for m in results)  # FIXED: weighted
        
        # Draw background bar
        self.risk_canvas.create_rectangle(10, 10, 340, 30,
                                         fill='#e2e8f0', outline='#cbd5e1')
        
        # Draw risk level
        width = int(330 * avg_prob)
        color = '#22c55e' if avg_prob < 0.3 else '#eab308' if avg_prob < 0.6 else '#ef4444'
        
        if width > 0:
            self.risk_canvas.create_rectangle(10, 10, 10 + width, 30,
                                             fill=color, outline='')
        
        # Draw labels
        self.risk_canvas.create_text(175, 38, text=f"Risk Level: {avg_prob*100:.1f}%",
                                    fill='#1e293b', font=('Segoe UI', 9, 'bold'))
    
    def download_results(self):
        """Download prediction results as graphs"""
        if self.last_prediction is None:
            messagebox.showwarning("No Data", "Make a prediction first!")
            return
        
        # Ask for save location
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"Loan_Prediction_{timestamp}.png"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile=default_name,
            title="Save Prediction Results"
        )
        
        if not filepath:
            return
        
        try:
            results = self.last_prediction['results']
            data = self.last_prediction['data']
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Loan Default Prediction Results - Pakistani Microfinance', fontsize=14, fontweight='bold')
            
            # 1. Model Risk Comparison (Bar Chart)
            ax1 = axes[0, 0]
            models = list(results.keys())
            risks = [results[m]['probability'] * 100 for m in models]
            colors = ['#22c55e' if r < 50 else '#ef4444' for r in risks]
            bars = ax1.bar(models, risks, color=colors, edgecolor='black')
            ax1.set_ylabel('Default Risk (%)')
            ax1.set_title('Risk by Model')
            ax1.set_ylim(0, 100)
            ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Threshold')
            for bar, risk in zip(bars, risks):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                        f'{risk:.1f}%', ha='center', fontsize=9)
            
            # 2. Risk Gauge (Pie Chart)
            ax2 = axes[0, 1]
            avg_risk = sum(risks) / len(risks)
            if avg_risk < 50:
                labels = ['Safe', 'Risk']
                sizes = [100 - avg_risk, avg_risk]
                pie_colors = ['#22c55e', '#e5e7eb']
            else:
                labels = ['Risk', 'Safe']
                sizes = [avg_risk, 100 - avg_risk]
                pie_colors = ['#ef4444', '#e5e7eb']
            ax2.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'Overall Risk: {avg_risk:.1f}%')
            
            # 3. Borrower Profile (Horizontal Bar)
            ax3 = axes[1, 0]
            profile_data = {
                'Age': data['age'],
                'Income (K)': data['monthly_income'] / 1000,
                'Loan (K)': data['loan_amount'] / 1000,
                'Credit Score': data['credit_score'] / 8.5,  # Scale to 100
                'Dependents': data['num_dependents'] * 10   # Scale to match
            }
            ax3.barh(list(profile_data.keys()), list(profile_data.values()), color='#38bdf8')
            ax3.set_xlabel('Value (scaled)')
            ax3.set_title('Borrower Profile')
            
            # 4. Recommendation Summary (Text)
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            if avg_risk >= 50:
                rec_text = "⚠️ HIGH RISK\n\nRecommendation:\n• Reject or reduce loan\n• Require collateral\n• Increase interest rate"
                rec_color = '#ef4444'
            else:
                rec_text = "✅ LOW RISK\n\nRecommendation:\n• Approve loan\n• Standard rates\n• Regular monitoring"
                rec_color = '#22c55e'
            
            ax4.text(0.5, 0.5, rec_text, fontsize=11, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor=rec_color, alpha=0.3),
                    transform=ax4.transAxes)
            
            # Borrower details
            details = f"""
Borrower Details:
• Occupation: {data['occupation']}
• Region: {data['region']}
• Purpose: {data['loan_purpose']}
• Term: {data['loan_term_months']} months
• Education: {data['education_years']} years
• Previous Loans: {data['previous_loans']}
            """
            ax4.text(0.5, 0.1, details, fontsize=9, ha='center', va='bottom',
                    transform=ax4.transAxes, family='monospace')
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            messagebox.showinfo("Success", f"Results saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")


    # ─────────────────────────────────────────────────────────────────────
    def _build_compare_tab(self, parent):
        """Build the Region × Occupation comparison tab."""
        REGIONS     = ['Punjab', 'Sindh', 'KPK', 'Balochistan']
        OCCUPATIONS = ['Farmer', 'Shopkeeper', 'Laborer', 'Artisan',
                       'Teacher', 'No job', 'Pensioner', 'Student']
        PURPOSES    = ['All', 'Agriculture', 'Business', 'Education',
                       'Home Repair', 'Medical']

        # ── Top controls bar ───────────────────────────────────────────
        ctrl_frame = tk.LabelFrame(parent, text="  Filter & Compare Options  ",
                                   font=('Segoe UI', 10, 'bold'),
                                   bg='#ffffff', fg='#1d6db5',
                                   relief='groove', bd=2, padx=12, pady=8)
        ctrl_frame.pack(fill='x', padx=15, pady=(12, 6))

        # Row 1: regions checkboxes
        tk.Label(ctrl_frame, text="Regions:", font=('Segoe UI', 9, 'bold'),
                 bg='#ffffff', fg='#1e293b').grid(row=0, column=0, sticky='w', padx=(0,8))
        self._cmp_region_vars = {}
        for i, r in enumerate(REGIONS):
            v = tk.BooleanVar(value=True)
            self._cmp_region_vars[r] = v
            tk.Checkbutton(ctrl_frame, text=r, variable=v,
                           font=('Segoe UI', 9), bg='#ffffff',
                           fg='#1e293b', selectcolor='#e0eeff',
                           activebackground='#ffffff').grid(row=0, column=i+1, sticky='w', padx=4)

        # Row 2: occupations checkboxes
        tk.Label(ctrl_frame, text="Occupations:", font=('Segoe UI', 9, 'bold'),
                 bg='#ffffff', fg='#1e293b').grid(row=1, column=0, sticky='w', padx=(0,8), pady=(6,0))
        self._cmp_occ_vars = {}
        for i, o in enumerate(OCCUPATIONS):
            v = tk.BooleanVar(value=True)
            self._cmp_occ_vars[o] = v
            tk.Checkbutton(ctrl_frame, text=o, variable=v,
                           font=('Segoe UI', 9), bg='#ffffff',
                           fg='#1e293b', selectcolor='#e0eeff',
                           activebackground='#ffffff').grid(row=1, column=i+1, sticky='w', padx=4, pady=(6,0))

        # Row 3: loan purpose filter + chart type + run button
        tk.Label(ctrl_frame, text="Loan Purpose:", font=('Segoe UI', 9, 'bold'),
                 bg='#ffffff', fg='#1e293b').grid(row=2, column=0, sticky='w', pady=(8,0))
        self._cmp_purpose_var = tk.StringVar(value='All')
        purpose_combo = ttk.Combobox(ctrl_frame, textvariable=self._cmp_purpose_var,
                                     values=PURPOSES, state='readonly',
                                     font=('Segoe UI', 9), width=14)
        purpose_combo.grid(row=2, column=1, columnspan=2, sticky='w', padx=4, pady=(8,0))

        tk.Label(ctrl_frame, text="Chart Type:", font=('Segoe UI', 9, 'bold'),
                 bg='#ffffff', fg='#1e293b').grid(row=2, column=3, sticky='w', padx=(16,4), pady=(8,0))
        self._cmp_chart_var = tk.StringVar(value='Grouped Bar')
        chart_combo = ttk.Combobox(ctrl_frame, textvariable=self._cmp_chart_var,
                                   values=['Grouped Bar', 'Heatmap', 'Line Trend', 'Stacked Bar'],
                                   state='readonly', font=('Segoe UI', 9), width=14)
        chart_combo.grid(row=2, column=4, columnspan=2, sticky='w', padx=4, pady=(8,0))

        run_btn = tk.Button(ctrl_frame, text="▶  Run Comparison",
                            font=('Segoe UI', 10, 'bold'),
                            bg='#1d6db5', fg='#ffffff',
                            activebackground='#155fa0', activeforeground='#ffffff',
                            relief='flat', cursor='hand2', padx=14, pady=6,
                            command=self._run_comparison)
        run_btn.grid(row=2, column=7, columnspan=2, padx=(20, 0), pady=(8, 0))

        # ── Summary stats strip ────────────────────────────────────────
        self._cmp_summary_frame = tk.Frame(parent, bg='#f0f4f8')
        self._cmp_summary_frame.pack(fill='x', padx=15, pady=(4, 6))

        # ── Chart canvas (matplotlib embedded) ────────────────────────
        chart_outer = tk.LabelFrame(parent, text="  Comparison Chart  ",
                                    font=('Segoe UI', 10, 'bold'),
                                    bg='#ffffff', fg='#1d6db5',
                                    relief='groove', bd=2)
        chart_outer.pack(fill='both', expand=True, padx=15, pady=(0, 8))

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.pyplot as plt
        self._cmp_fig, self._cmp_ax = plt.subplots(figsize=(11, 5))
        self._cmp_fig.patch.set_facecolor('#ffffff')
        self._cmp_canvas_widget = FigureCanvasTkAgg(self._cmp_fig, master=chart_outer)
        self._cmp_canvas_widget.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

        # ── Data table below chart ─────────────────────────────────────
        tbl_frame = tk.LabelFrame(parent, text="  Default Rate Table (%)  ",
                                  font=('Segoe UI', 10, 'bold'),
                                  bg='#ffffff', fg='#1d6db5',
                                  relief='groove', bd=2)
        tbl_frame.pack(fill='x', padx=15, pady=(0, 12))

        self._cmp_tree = ttk.Treeview(tbl_frame, height=5)
        tbl_scroll = ttk.Scrollbar(tbl_frame, orient='horizontal',
                                   command=self._cmp_tree.xview)
        self._cmp_tree.configure(xscrollcommand=tbl_scroll.set)
        tbl_style = ttk.Style()
        tbl_style.configure('Treeview', font=('Segoe UI', 9),
                            rowheight=22, background='#ffffff',
                            fieldbackground='#ffffff', foreground='#1e293b')
        tbl_style.configure('Treeview.Heading', font=('Segoe UI', 9, 'bold'),
                            background='#1d6db5', foreground='#ffffff')
        self._cmp_tree.pack(fill='x', padx=5, pady=(5, 0))
        tbl_scroll.pack(fill='x', padx=5, pady=(0, 5))

        # Export button
        export_btn = tk.Button(parent, text="💾  Export Chart as PNG",
                               font=('Segoe UI', 9, 'bold'),
                               bg='#1a7f5a', fg='#ffffff',
                               activebackground='#146b4a', activeforeground='#ffffff',
                               relief='flat', cursor='hand2', padx=12, pady=5,
                               command=self._export_comparison_chart)
        export_btn.pack(pady=(0, 10))

        # Draw default chart on load
        self.root.after(800, self._run_comparison)

    # ─────────────────────────────────────────────────────────────────────
    def _compute_default_rates(self):
        """Return a dict {region: {occupation: default_rate%}} from self.df."""
        regions     = [r for r, v in self._cmp_region_vars.items() if v.get()]
        occupations = [o for o, v in self._cmp_occ_vars.items()  if v.get()]
        purpose     = self._cmp_purpose_var.get()

        if not regions or not occupations:
            return {}, [], []

        df = self.df.copy()
        if purpose != 'All':
            df = df[df['loan_purpose'] == purpose]

        rates = {}
        for reg in regions:
            rates[reg] = {}
            for occ in occupations:
                mask = (df['region'] == reg) & (df['occupation'] == occ)
                sub  = df[mask]
                rates[reg][occ] = (sub['default'].mean() * 100) if len(sub) > 0 else 0.0
        return rates, regions, occupations

    # ─────────────────────────────────────────────────────────────────────
    def _run_comparison(self):
        """Compute default rates and draw the selected chart type."""
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np

        rates, regions, occupations = self._compute_default_rates()
        if not rates:
            messagebox.showwarning("Nothing selected",
                                   "Please select at least one Region and one Occupation.")
            return

        chart_type = self._cmp_chart_var.get()
        ax = self._cmp_ax
        ax.clear()
        self._cmp_fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8fafc')
        ax.tick_params(colors='#1e293b')
        for spine in ax.spines.values():
            spine.set_edgecolor('#cbd5e1')

        # Build matrix
        matrix = np.array([[rates[r].get(o, 0) for o in occupations] for r in regions])

        # Colour palette (one colour per region)
        palette = ['#1d6db5', '#c0446b', '#1a7f5a', '#e07b1a',
                   '#7c3aed', '#0891b2', '#b45309', '#be123c']
        reg_colors = {r: palette[i % len(palette)] for i, r in enumerate(regions)}

        purpose_label = self._cmp_purpose_var.get()
        title_suffix  = f" — Loan Purpose: {purpose_label}" if purpose_label != 'All' else ''

        if chart_type == 'Grouped Bar':
            x      = np.arange(len(occupations))
            n      = len(regions)
            width  = 0.8 / n
            for i, reg in enumerate(regions):
                vals = [rates[reg].get(o, 0) for o in occupations]
                bars = ax.bar(x + i * width - (n - 1) * width / 2, vals,
                              width=width * 0.9, label=reg,
                              color=reg_colors[reg], alpha=0.85)
                for b, v in zip(bars, vals):
                    if v > 0:
                        ax.text(b.get_x() + b.get_width() / 2,
                                b.get_height() + 0.4, f'{v:.1f}',
                                ha='center', va='bottom',
                                fontsize=6.5, color='#1e293b')
            ax.set_xticks(x)
            ax.set_xticklabels(occupations, rotation=25, ha='right', fontsize=9)
            ax.set_ylabel('Default Rate (%)', fontsize=10, color='#1e293b')
            ax.set_title(f'Default Rate by Occupation & Region{title_suffix}',
                         fontsize=12, fontweight='bold', color='#1e293b')
            ax.legend(title='Region', fontsize=8, title_fontsize=8)
            ax.axhline(y=np.mean(matrix), color='#ef4444', linestyle='--',
                       linewidth=1, alpha=0.7, label='Overall avg')
            ax.set_ylim(0, min(100, matrix.max() * 1.3 + 5))

        elif chart_type == 'Heatmap':
            im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto',
                           vmin=0, vmax=min(100, matrix.max() + 5))
            ax.set_xticks(range(len(occupations)))
            ax.set_xticklabels(occupations, rotation=30, ha='right', fontsize=9)
            ax.set_yticks(range(len(regions)))
            ax.set_yticklabels(regions, fontsize=9)
            ax.set_title(f'Default Rate Heatmap (%){title_suffix}',
                         fontsize=12, fontweight='bold', color='#1e293b')
            for i in range(len(regions)):
                for j in range(len(occupations)):
                    val = matrix[i, j]
                    ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                            fontsize=8, fontweight='bold',
                            color='white' if val > matrix.max() * 0.6 else '#1e293b')
            self._cmp_fig.colorbar(im, ax=ax, label='Default Rate (%)', shrink=0.8)

        elif chart_type == 'Line Trend':
            for reg in regions:
                vals = [rates[reg].get(o, 0) for o in occupations]
                ax.plot(range(len(occupations)), vals,
                        marker='o', linewidth=2, markersize=6,
                        label=reg, color=reg_colors[reg])
                for xi, yi in enumerate(vals):
                    ax.annotate(f'{yi:.1f}%', (xi, yi),
                                textcoords='offset points', xytext=(0, 6),
                                ha='center', fontsize=7, color=reg_colors[reg])
            ax.set_xticks(range(len(occupations)))
            ax.set_xticklabels(occupations, rotation=25, ha='right', fontsize=9)
            ax.set_ylabel('Default Rate (%)', fontsize=10, color='#1e293b')
            ax.set_title(f'Default Rate Trend by Occupation{title_suffix}',
                         fontsize=12, fontweight='bold', color='#1e293b')
            ax.legend(title='Region', fontsize=8, title_fontsize=8)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.set_ylim(0, min(100, matrix.max() * 1.35 + 3))

        elif chart_type == 'Stacked Bar':
            bottom = np.zeros(len(occupations))
            x = np.arange(len(occupations))
            for i, reg in enumerate(regions):
                vals = np.array([rates[reg].get(o, 0) for o in occupations])
                ax.bar(x, vals, bottom=bottom, label=reg,
                       color=reg_colors[reg], alpha=0.85)
                for xi, (v, b) in enumerate(zip(vals, bottom)):
                    if v > 1:
                        ax.text(xi, b + v / 2, f'{v:.1f}',
                                ha='center', va='center',
                                fontsize=6.5, color='white', fontweight='bold')
                bottom += vals
            ax.set_xticks(x)
            ax.set_xticklabels(occupations, rotation=25, ha='right', fontsize=9)
            ax.set_ylabel('Cumulative Default Rate (%)', fontsize=10, color='#1e293b')
            ax.set_title(f'Stacked Default Rate by Occupation{title_suffix}',
                         fontsize=12, fontweight='bold', color='#1e293b')
            ax.legend(title='Region', fontsize=8, title_fontsize=8)

        self._cmp_fig.tight_layout()
        self._cmp_canvas_widget.draw()

        # ── Update summary strip ───────────────────────────────────────
        for w in self._cmp_summary_frame.winfo_children():
            w.destroy()

        overall   = matrix.mean()
        max_val   = matrix.max()
        min_val   = matrix[matrix > 0].min() if matrix.max() > 0 else 0
        ri, ci    = np.unravel_index(matrix.argmax(), matrix.shape)
        ri2, ci2  = np.unravel_index(matrix.argmin(), matrix.shape)

        stats = [
            ("📊 Overall Avg", f"{overall:.1f}%", '#1d6db5'),
            ("🔴 Highest Risk", f"{regions[ri]} / {occupations[ci]}\n{max_val:.1f}%", '#c0392b'),
            ("🟢 Lowest Risk",  f"{regions[ri2]} / {occupations[ci2]}\n{min_val:.1f}%", '#1a7f5a'),
            ("📋 Records",      f"{len(self.df[self.df['region'].isin(regions) & self.df['occupation'].isin(occupations)]):,}", '#7c3aed'),
        ]
        for label, value, color in stats:
            card = tk.Frame(self._cmp_summary_frame, bg='#ffffff',
                            relief='groove', bd=1)
            card.pack(side='left', padx=6, pady=4, ipadx=10, ipady=5)
            tk.Label(card, text=label, font=('Segoe UI', 8, 'bold'),
                     bg='#ffffff', fg='#4a5568').pack()
            tk.Label(card, text=value, font=('Segoe UI', 10, 'bold'),
                     bg='#ffffff', fg=color).pack()

        # ── Update data table ──────────────────────────────────────────
        tree = self._cmp_tree
        tree.delete(*tree.get_children())
        tree['columns'] = ['Region'] + occupations
        tree.column('#0', width=0, stretch=False)
        tree.column('Region', width=100, anchor='w')
        tree.heading('Region', text='Region ▼')
        for occ in occupations:
            tree.column(occ, width=90, anchor='center')
            tree.heading(occ, text=occ)
        for i, reg in enumerate(regions):
            row_vals = [f"{rates[reg].get(o, 0):.1f}%" for o in occupations]
            tag = 'even' if i % 2 == 0 else 'odd'
            tree.insert('', 'end', values=[reg] + row_vals, tags=(tag,))
        tree.tag_configure('even', background='#f0f7ff')
        tree.tag_configure('odd',  background='#ffffff')

    # ─────────────────────────────────────────────────────────────────────
    def _export_comparison_chart(self):
        """Save the comparison chart as a PNG file."""
        from datetime import datetime
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"Comparison_Chart_{timestamp}.png"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile=default_name,
            title="Save Comparison Chart"
        )
        if not filepath:
            return
        try:
            self._cmp_fig.savefig(filepath, dpi=150, bbox_inches='tight',
                                  facecolor='white')
            messagebox.showinfo("Saved", f"Chart saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")


def main():
    root = tk.Tk()
    app = LoanDefaultPredictor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
