import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# CUSTOM CSS STYLING
# ===========================
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: bold;
    }
    
    /* Headers */
    h1 {
        color: white;
        text-align: center;
        padding: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    h2, h3 {
        color: white;
    }
    
    /* Cards */
    .css-1r6slb0 {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 15px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ===========================
# SESSION STATE INITIALIZATION
# ===========================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# ===========================
# HELPER FUNCTIONS
# ===========================

@st.cache_data
def load_data(file):
    """Load and clean the customer churn dataset"""
    try:
        df = pd.read_csv(file)
        
        # Clean TotalCharges column
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
        # Convert Churn to binary
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        # Create tenure groups
        df['TenureGroup'] = pd.cut(
            df['tenure'], 
            bins=[0, 12, 24, 48, 72],
            labels=['0-12 months', '13-24 months', '25-48 months', '49+ months']
        )
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def train_model(df):
    """Train logistic regression model"""
    try:
        # Select features
        categorical_features = ['Contract', 'PaymentMethod', 'InternetService']
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        # Create copy for modeling
        df_model = df.copy()
        label_encoders = {}
        
        # Encode categorical variables
        for col in categorical_features:
            le = LabelEncoder()
            df_model[f'{col}_encoded'] = le.fit_transform(df_model[col].astype(str))
            label_encoders[col] = le
        
        # Prepare features and target
        feature_columns = [f'{col}_encoded' for col in categorical_features] + numerical_features
        X = df_model[feature_columns]
        y = df_model['Churn']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return model, scaler, label_encoders, metrics, feature_columns
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None, None

# ===========================
# SIDEBAR NAVIGATION
# ===========================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/graph.png", width=80)
    st.title("🎯 Navigation")
    
    page = st.radio(
        "Select a page:",
        ["🏠 Home", "📤 Upload Data", "📊 Dashboard", "📈 Analysis", "🤖 ML Model", "🎯 Predict", "💡 Insights"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if st.session_state.data_loaded:
        st.success("✅ Data Loaded")
    else:
        st.warning("⚠️ No Data Loaded")
    
    if st.session_state.model_trained:
        st.success("✅ Model Trained")
    else:
        st.info("ℹ️ Model Not Trained")
    
    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("[📖 Documentation](https://github.com)")
    st.markdown("[💾 Download Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)")

# ===========================
# PAGE: HOME
# ===========================
if page == "🏠 Home":
    st.markdown("<h1>🎯 Customer Churn Prediction System</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); 
                    border-radius: 20px; padding: 30px; text-align: center;'>
            <h2 style='color: white;'>Welcome! 👋</h2>
            <p style='color: white; font-size: 18px;'>
                Predict customer churn using machine learning and take action to retain your customers.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.15); padding: 20px; border-radius: 15px; text-align: center;'>
            <h1 style='font-size: 48px;'>📤</h1>
            <h3 style='color: white;'>Upload Data</h3>
            <p style='color: white;'>Load your customer CSV file</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.15); padding: 20px; border-radius: 15px; text-align: center;'>
            <h1 style='font-size: 48px;'>📊</h1>
            <h3 style='color: white;'>Analyze</h3>
            <p style='color: white;'>Explore churn patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.15); padding: 20px; border-radius: 15px; text-align: center;'>
            <h1 style='font-size: 48px;'>🤖</h1>
            <h3 style='color: white;'>Predict</h3>
            <p style='color: white;'>ML-powered predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.15); padding: 20px; border-radius: 15px; text-align: center;'>
            <h1 style='font-size: 48px;'>💡</h1>
            <h3 style='color: white;'>Insights</h3>
            <p style='color: white;'>Actionable recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Getting started
    with st.expander("🚀 Getting Started", expanded=True):
        st.markdown("""
        ### Quick Start Guide:
        
        1. **Upload Data** - Go to the Upload Data page and load your CSV file
        2. **Explore Dashboard** - View key metrics and visualizations
        3. **Analyze Patterns** - Discover what drives customer churn
        4. **Train Model** - Build your prediction model
        5. **Make Predictions** - Predict churn for individual customers
        6. **Get Insights** - View business recommendations
        
        ### Dataset Requirements:
        - CSV format
        - Columns: customerID, tenure, MonthlyCharges, Contract, PaymentMethod, Churn, etc.
        - Download sample: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
        """)

# ===========================
# PAGE: UPLOAD DATA
# ===========================
elif page == "📤 Upload Data":
    st.markdown("<h1>📤 Upload Your Dataset</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload the Telco Customer Churn dataset or similar format"
        )
        
        if uploaded_file is not None:
            with st.spinner("🔄 Loading and processing data..."):
                df = load_data(uploaded_file)
                
                if df is not None:
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    
                    st.success(f"✅ Successfully loaded {len(df):,} customer records!")
                    
                    # Quick stats
                    col_a, col_b, col_c, col_d = st.columns(4)
                    
                    with col_a:
                        st.metric("📊 Total Rows", f"{len(df):,}")
                    with col_b:
                        st.metric("📋 Columns", len(df.columns))
                    with col_c:
                        st.metric("📉 Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
                    with col_d:
                        st.metric("✅ Complete Data", f"{(1-df.isnull().sum().sum()/(len(df)*len(df.columns)))*100:.1f}%")
                    
                    # Data preview
                    st.markdown("### 👀 Data Preview")
                    st.dataframe(df.head(10), use_container_width=True, height=400)
                    
                    # Data info
                    with st.expander("📊 Dataset Information"):
                        col_x, col_y = st.columns(2)
                        with col_x:
                            st.markdown("**Numerical Columns:**")
                            st.write(df.select_dtypes(include=[np.number]).columns.tolist())
                        with col_y:
                            st.markdown("**Categorical Columns:**")
                            st.write(df.select_dtypes(include=['object']).columns.tolist())
    
    with col2:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.15); padding: 20px; border-radius: 15px;'>
            <h3 style='color: white;'>📚 Dataset Info</h3>
            <p style='color: white;'>
                Upload a CSV file containing customer data with the following columns:
            </p>
            <ul style='color: white;'>
                <li>customerID</li>
                <li>tenure</li>
                <li>MonthlyCharges</li>
                <li>TotalCharges</li>
                <li>Contract</li>
                <li>PaymentMethod</li>
                <li>InternetService</li>
                <li>Churn (Yes/No)</li>
            </ul>
            <br>
            <p style='color: white;'>
                <strong>📥 Get Sample Data:</strong><br>
                <a href='https://www.kaggle.com/datasets/blastchar/telco-customer-churn' 
                   style='color: #ffd700;'>Kaggle: Telco Churn</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

# ===========================
# PAGE: DASHBOARD
# ===========================
elif page == "📊 Dashboard":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.df
        
        st.markdown("<h1>📊 Executive Dashboard</h1>", unsafe_allow_html=True)
        
        # KPI Metrics
        total_customers = len(df)
        churned = df['Churn'].sum()
        retained = total_customers - churned
        churn_rate = (churned / total_customers) * 100
        avg_charge = df['MonthlyCharges'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "👥 Total Customers",
                f"{total_customers:,}",
                help="Total number of customers"
            )
        
        with col2:
            st.metric(
                "📉 Churned",
                f"{churned:,}",
                delta=f"-{churn_rate:.1f}%",
                delta_color="inverse",
                help="Customers who left"
            )
        
        with col3:
            st.metric(
                "✅ Retained",
                f"{retained:,}",
                delta=f"+{100-churn_rate:.1f}%",
                help="Customers who stayed"
            )
        
        with col4:
            st.metric(
                "💰 Avg Monthly",
                f"${avg_charge:.2f}",
                help="Average monthly charges"
            )
        
        st.markdown("---")
        
        # Charts
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### 🥧 Churn Distribution")
            fig = px.pie(
                names=['Retained', 'Churned'],
                values=[retained, churned],
                color_discrete_sequence=['#10b981', '#ef4444'],
                hole=0.5
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=16
            )
            fig.update_layout(
                height=400,
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=14)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.markdown("### 📊 Monthly Charges by Churn")
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=df[df['Churn']==0]['MonthlyCharges'],
                name='Retained',
                marker_color='#10b981',
                boxmean='sd'
            ))
            fig.add_trace(go.Box(
                y=df[df['Churn']==1]['MonthlyCharges'],
                name='Churned',
                marker_color='#ef4444',
                boxmean='sd'
            ))
            fig.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.2)')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional charts
        st.markdown("### 📈 Churn Trends")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            tenure_churn = df.groupby('TenureGroup')['Churn'].mean() * 100
            fig = px.bar(
                x=tenure_churn.index,
                y=tenure_churn.values,
                labels={'x': 'Tenure Group', 'y': 'Churn Rate (%)'},
                color=tenure_churn.values,
                color_continuous_scale='Reds',
                text=tenure_churn.values.round(1)
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False,
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.2)')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_b:
            contract_churn = df.groupby('Contract')['Churn'].mean() * 100
            fig = px.bar(
                x=contract_churn.index,
                y=contract_churn.values,
                labels={'x': 'Contract Type', 'y': 'Churn Rate (%)'},
                color=contract_churn.values,
                color_continuous_scale='Oranges',
                text=contract_churn.values.round(1)
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False,
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.2)')
            )
            st.plotly_chart(fig, use_container_width=True)

# ===========================
# PAGE: ANALYSIS
# ===========================
elif page == "📈 Analysis":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.df
        
        st.markdown("<h1>📈 Deep Dive Analysis</h1>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📅 Tenure Analysis", "📋 Contract Analysis", "💳 Payment Analysis"])
        
        with tab1:
            st.markdown("### Churn by Customer Tenure")
            
            tenure_data = df.groupby('TenureGroup').agg({
                'Churn': ['sum', 'count', 'mean']
            }).round(3)
            tenure_data.columns = ['Churned', 'Total', 'Rate']
            tenure_data['Rate'] = tenure_data['Rate'] * 100
            tenure_data['Retained'] = tenure_data['Total'] - tenure_data['Churned']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=tenure_data.index,
                    y=tenure_data['Retained'],
                    name='Retained',
                    marker_color='#10b981'
                ))
                fig.add_trace(go.Bar(
                    x=tenure_data.index,
                    y=tenure_data['Churned'],
                    name='Churned',
                    marker_color='#ef4444'
                ))
                fig.update_layout(
                    barmode='stack',
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.2)')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Churn Rates by Tenure:**")
                for idx, row in tenure_data.iterrows():
                    st.metric(
                        str(idx),
                        f"{row['Rate']:.1f}%",
                        f"{int(row['Churned'])} churned"
                    )
            
            st.info("💡 **Insight:** Early tenure customers (0-12 months) show the highest churn risk. Focus retention efforts here!")
        
        with tab2:
            st.markdown("### Churn by Contract Type")
            
            contract_data = df.groupby(['Contract', 'Churn']).size().unstack(fill_value=0)
            contract_data.columns = ['Retained', 'Churned']
            contract_data['Total'] = contract_data.sum(axis=1)
            contract_data['ChurnRate'] = (contract_data['Churned'] / contract_data['Total'] * 100).round(1)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    contract_data.reset_index(),
                    x='Contract',
                    y=['Retained', 'Churned'],
                    barmode='group',
                    color_discrete_sequence=['#10b981', '#ef4444'],
                    labels={'value': 'Number of Customers', 'variable': 'Status'}
                )
                fig.update_layout(
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.2)')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Churn Rates:**")
                for idx, row in contract_data.iterrows():
                    st.metric(
                        idx,
                        f"{row['ChurnRate']:.1f}%",
                        f"{int(row['Churned'])} / {int(row['Total'])}"
                    )
            
            st.warning("⚠️ **Alert:** Month-to-month contracts have 3-4x higher churn! Consider offering contract conversion incentives.")
        
        with tab3:
            st.markdown("### Churn by Payment Method")
            
            payment_churn = df.groupby('PaymentMethod')['Churn'].mean() * 100
            payment_counts = df.groupby('PaymentMethod').size()
            
            fig = px.bar(
                x=payment_churn.index,
                y=payment_churn.values,
                color=payment_churn.values,
                color_continuous_scale='Plasma',
                labels={'x': 'Payment Method', 'y': 'Churn Rate (%)'},
                text=payment_churn.values.round(1)
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False,
                yaxis=dict(gridcolor='rgba(255,255,255,0.2)')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Payment stats
            col1, col2, col3, col4 = st.columns(4)
            for i, (method, rate) in enumerate(payment_churn.items()):
                with [col1, col2, col3, col4][i]:
                    st.metric(
                        method.split('(')[0].strip(),
                        f"{rate:.1f}%",
                        f"{payment_counts[method]} customers"
                    )

# ===========================
# PAGE: ML MODEL
# ===========================
elif page == "🤖 ML Model":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.df
        
        st.markdown("<h1>🤖 Machine Learning Model</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px;'>
            <h3 style='color: white;'>Model Information</h3>
            <p style='color: white;'>
                This system uses <strong>Logistic Regression</strong> to predict customer churn.
                The model analyzes patterns in customer behavior, contract details, and usage to 
                identify high-risk customers.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🚀 Train Model", type="primary", use_container_width=True):
                with st.spinner("🔄 Training model... This may take a moment..."):
                    model, scaler, encoders, metrics, features = train_model(df)
                    
                    if model is not None:
                        st.session_state.model = model
                        st.session_state.scaler = scaler
                        st.session_state.encoders = encoders
                        st.session_state.metrics = metrics
                        st.session_state.features = features
                        st.session_state.model_trained = True
                        
                        st.success("✅ Model trained successfully!")
                        st.balloons()
        
        with col2:
            if st.session_state.model_trained:
                st.success("✅ Model Ready")
            else:
                st.info("ℹ️ Click Train Model to begin")
        
        if st.session_state.model_trained:
            metrics = st.session_state.metrics
            
            st.markdown("---")
            st.markdown("### 📊 Model Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🎯 Accuracy",
                    f"{metrics['accuracy']*100:.2f}%",
                    help="Overall prediction accuracy"
                )
            
            with col2:
                st.metric(
                    "🔍 Precision",
                    f"{metrics['precision']*100:.2f}%",
                    help="Accuracy of churn predictions"
                )
            
            with col3:
                st.metric(
                    "📡 Recall",
                    f"{metrics['recall']*100:.2f}%",
                    help="% of churners correctly identified"
                )
            
            st.markdown("---")
            
            # Confusion Matrix
            col_a, col_b = st.columns([1, 1])
            
            with col_a:
                st.markdown("### Confusion Matrix")
                cm = metrics['confusion_matrix']
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted: No Churn', 'Predicted: Churn'],
                    y=['Actual: No Churn', 'Actual: Churn'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 20},
                    showscale=False
                ))
                fig.update_layout(
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', size=14)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_b:
                st.markdown("### Model Insights")
                st.markdown(f"""
                <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;'>
                    <p style='color: white;'><strong>True Negatives:</strong> {cm[0][0]:,} ✅</p>
                    <p style='color: white;'><strong>False Positives:</strong> {cm[0][1]:,} ⚠️</p>
                    <p style='color: white;'><strong>False Negatives:</strong> {cm[1][0]:,} ⚠️</p>
                    <p style='color: white;'><strong>True Positives:</strong> {cm[1][1]:,} ✅</p>
                    <br>
                    <p style='color: #10b981;'>The model correctly identifies <strong>{metrics['recall']*100:.1f}%</strong> of customers who will churn!</p>
                </div>
                """, unsafe_allow_html=True)

# ===========================
# PAGE: PREDICT
# ===========================
elif page == "🎯 Predict":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first!")
    elif not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first!")
    else:
        st.markdown("<h1>🎯 Predict Customer Churn</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
            <p style='color: white; font-size: 16px;'>
                Enter customer details below to predict their likelihood of churning.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Customer Information")
            tenure = st.slider("Tenure (months)", 0, 72, 12, help="How long has the customer been with us?")
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=5.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges), step=100.0)
        
        with col2:
            st.markdown("### 📋 Service Details")
            contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
            payment = st.selectbox(
                "Payment Method", 
                ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
            )
            internet = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
            model = st.session_state.model
            scaler = st.session_state.scaler
            encoders = st.session_state.encoders
            
            # Encode inputs
            contract_enc = encoders['Contract'].transform([contract])[0]
            payment_enc = encoders['PaymentMethod'].transform([payment])[0]
            internet_enc = encoders['InternetService'].transform([internet])[0]
            
            # Create feature array
            features = np.array([[contract_enc, payment_enc, internet_enc, tenure, monthly_charges, total_charges]])
            features_scaled = scaler.transform(features)
            
            # Predict
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
            
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            
            with col_b:
                # Risk gauge
                if probability > 0.7:
                    risk_color = "#dc2626"
                    risk_level = "🔴 CRITICAL RISK"
                    risk_text = "Immediate action required!"
                elif probability > 0.5:
                    risk_color = "#f59e0b"
                    risk_level = "🟠 HIGH RISK"
                    risk_text = "Customer likely to churn"
                elif probability > 0.3:
                    risk_color = "#fbbf24"
                    risk_level = "🟡 MEDIUM RISK"
                    risk_text = "Monitor closely"
                else:
                    risk_color = "#10b981"
                    risk_level = "🟢 LOW RISK"
                    risk_text = "Customer likely to stay"
                
                st.markdown(f"""
                <div style='background: {risk_color}; padding: 30px; border-radius: 20px; text-align: center;'>
                    <h1 style='color: white; font-size: 48px; margin: 0;'>{probability*100:.1f}%</h1>
                    <h3 style='color: white; margin: 10px 0;'>{risk_level}</h3>
                    <p style='color: white; font-size: 18px;'>{risk_text}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Progress bar
            st.progress(probability)
            
            st.markdown("---")
            
            # Recommendations
            st.markdown("### 💡 Recommended Actions")
            
            if probability > 0.5:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div style='background: rgba(220, 38, 38, 0.2); padding: 20px; border-radius: 10px; border-left: 5px solid #dc2626;'>
                        <h4 style='color: white;'>🚨 Immediate Actions</h4>
                        <ul style='color: white;'>
                            <li>Contact within 24 hours</li>
                            <li>Offer 10-15% retention discount</li>
                            <li>Schedule customer success call</li>
                            <li>Review service satisfaction</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div style='background: rgba(245, 158, 11, 0.2); padding: 20px; border-radius: 10px; border-left: 5px solid #f59e0b;'>
                        <h4 style='color: white;'>📋 Long-term Strategy</h4>
                        <ul style='color: white;'>
                            <li>Suggest contract upgrade</li>
                            <li>Highlight unused features</li>
                            <li>Assign dedicated support</li>
                            <li>Add to VIP monitoring list</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: rgba(16, 185, 129, 0.2); padding: 20px; border-radius: 10px; border-left: 5px solid #10b981;'>
                    <h4 style='color: white;'>✅ Maintenance Actions</h4>
                    <ul style='color: white;'>
                        <li>Continue regular check-ins</li>
                        <li>Send quarterly satisfaction surveys</li>
                        <li>Offer upsell opportunities for premium features</li>
                        <li>Recognize loyalty with rewards program</li>
                        <li>Keep as reference for best practices</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Customer profile summary
            st.markdown("---")
            st.markdown("### 👤 Customer Profile Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Tenure", f"{tenure} months")
                st.metric("Contract", contract)
            
            with col2:
                st.metric("Monthly Charges", f"${monthly_charges:.2f}")
                st.metric("Payment Method", payment.split('(')[0])
            
            with col3:
                st.metric("Total Charges", f"${total_charges:.2f}")
                st.metric("Internet Service", internet)

# ===========================
# PAGE: INSIGHTS
# ===========================
elif page == "💡 Insights":
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.df
        
        st.markdown("<h1>💡 Business Insights & Recommendations</h1>", unsafe_allow_html=True)
        
        # Calculate metrics
        total_customers = len(df)
        churned = df['Churn'].sum()
        churn_rate = (churned / total_customers) * 100
        avg_monthly = df['MonthlyCharges'].mean()
        annual_loss = churned * avg_monthly * 12
        
        # Financial Impact
        st.markdown("### 💰 Financial Impact Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Annual Revenue Loss", f"${annual_loss:,.0f}")
        
        with col2:
            st.metric("10% Reduction Saves", f"${annual_loss*0.1:,.0f}", delta="10%")
        
        with col3:
            st.metric("20% Reduction Saves", f"${annual_loss*0.2:,.0f}", delta="20%")
        
        with col4:
            st.metric("Avg Customer Value", f"${avg_monthly*12:.0f}/year")
        
        st.markdown("---")
        
        # Key Findings
        st.markdown("### 🔍 Key Findings")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("""
            <div style='background: rgba(220, 38, 38, 0.2); padding: 20px; border-radius: 15px; border-left: 5px solid #dc2626;'>
                <h4 style='color: white;'>⚠️ Critical Risk Factors</h4>
                <ul style='color: white; line-height: 1.8;'>
                    <li><strong>Early Tenure Crisis:</strong> First 12 months show 3-4x higher churn</li>
                    <li><strong>Contract Type:</strong> Month-to-month customers churn 4x more</li>
                    <li><strong>Price Sensitivity:</strong> Churned customers pay $15-20 more/month</li>
                    <li><strong>Payment Method:</strong> Electronic check users show higher churn</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown("""
            <div style='background: rgba(16, 185, 129, 0.2); padding: 20px; border-radius: 15px; border-left: 5px solid #10b981;'>
                <h4 style='color: white;'>✅ Retention Opportunities</h4>
                <ul style='color: white; line-height: 1.8;'>
                    <li><strong>Contract Conversion:</strong> Incentivize long-term commitments</li>
                    <li><strong>Onboarding Excellence:</strong> 90-day success programs</li>
                    <li><strong>Price Optimization:</strong> Loyalty discounts for high-value customers</li>
                    <li><strong>Payment Experience:</strong> Streamline electronic payments</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Action Plan
        st.markdown("### 🎯 Recommended Action Plan")
        
        recommendations = [
            {
                "priority": "🔴 Priority 1",
                "title": "Deploy Early Warning System",
                "description": "Use ML model to identify at-risk customers and trigger automated retention campaigns",
                "impact": "Prevent 30% of predicted churns",
                "timeline": "Immediate",
                "color": "#dc2626"
            },
            {
                "priority": "🟠 Priority 2",
                "title": "Contract Conversion Campaign",
                "description": "Offer 15% discount for customers switching from month-to-month to annual contracts",
                "impact": "Convert 25% of MTM customers, reduce churn by 15%",
                "timeline": "1-2 months",
                "color": "#f59e0b"
            },
            {
                "priority": "🟡 Priority 3",
                "title": "Enhanced Onboarding Program",
                "description": "Implement 90-day customer success program with weekly check-ins for new customers",
                "impact": "20% reduction in early-tenure churn",
                "timeline": "2-3 months",
                "color": "#fbbf24"
            },
            {
                "priority": "🟢 Priority 4",
                "title": "Pricing & Value Optimization",
                "description": "Introduce loyalty discounts and better communicate value proposition to high-paying customers",
                "impact": "10% churn reduction in high-charge segment",
                "timeline": "3-4 months",
                "color": "#10b981"
            }
        ]
        
        for rec in recommendations:
            with st.expander(f"{rec['priority']}: {rec['title']}", expanded=False):
                st.markdown(f"""
                <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border-left: 5px solid {rec["color"]};'>
                    <p style='color: white;'><strong>📝 Description:</strong> {rec['description']}</p>
                    <p style='color: white;'><strong>📊 Expected Impact:</strong> {rec['impact']}</p>
                    <p style='color: white;'><strong>⏰ Timeline:</strong> {rec['timeline']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ROI Calculator
        st.markdown("### 💵 ROI Calculator")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            reduction_percent = st.slider("Target Churn Reduction (%)", 0, 50, 15, 5)
        
        with col2:
            customers_saved = int(churned * (reduction_percent / 100))
            revenue_saved = annual_loss * (reduction_percent / 100)
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                        padding: 25px; border-radius: 15px; text-align: center;'>
                <h2 style='color: white; margin: 0;'>Potential Annual Savings</h2>
                <h1 style='color: white; font-size: 48px; margin: 10px 0;'>${revenue_saved:,.0f}</h1>
                <p style='color: white; font-size: 18px;'>By retaining {customers_saved:,} customers</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Summary table
        scenarios = []
        for pct in [5, 10, 15, 20, 25]:
            scenarios.append({
                'Reduction': f'{pct}%',
                'Customers Saved': f'{int(churned * pct/100):,}',
                'Annual Savings': f'${annual_loss * pct/100:,.0f}'
            })
        
        st.markdown("#### 📊 Scenario Analysis")
        st.dataframe(pd.DataFrame(scenarios), use_container_width=True, hide_index=True)

# ===========================
# FOOTER
# ===========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='color: white; font-size: 14px;'>
        🔍 Customer Churn Analysis & Prediction System | Built with Streamlit & Scikit-learn
    </p>
    <p style='color: white; font-size: 12px;'>
        Made with ❤️ for better customer retention
    </p>
</div>
""", unsafe_allow_html=True)