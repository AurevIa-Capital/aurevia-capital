import streamlit as st


def load_css():
    """Load and apply custom CSS styles."""
    st.markdown(
        """
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #1E3A8A;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .card {
            border-radius: 5px;
            padding: 1.5rem;
            background-color: #f8f9fa;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #1E3A8A;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #64748b;
        }
        .profit {
            color: #10b981;
        }
        .loss {
            color: #ef4444;
        }
        .highlight {
            background-color: #f1f5f9;
            padding: 0.5rem;
            border-left: 4px solid #1E3A8A;
            margin-bottom: 1rem;
        }
        .watch-card {
            display: flex;
            align-items: center;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 15px;
            background-color: #ffffff;
        }
        .watch-image {
            width: 80px;
            height: 80px;
            object-fit: contain;
            margin-right: 20px;
        }
        .watch-info {
            flex: 1;
        }
        .watch-name {
            font-size: 16px;
            font-weight: 500;
            margin-bottom: 5px;
        }
        .watch-price {
            font-size: 14px;
            color: #6c757d;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #eaeaea;
            color: #6c757d;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def create_footer():
    """Create footer section."""
    st.markdown(
        """
    <div class="footer">
        <p>Luxury Watch Price Forecaster &copy; 2025</p>
        <p>Time series forecasting model using machine learning to predict luxury watch prices</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
