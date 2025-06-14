"""Watch-specific UI plugin."""

from typing import List, Dict, Any
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from .base import AssetPlugin


class WatchPlugin(AssetPlugin):
    """Watch-specific UI components."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("watch", config)
    
    def get_display_name(self) -> str:
        """Return human-readable display name."""
        return "Luxury Watches"
    
    def get_icon(self) -> str:
        """Return emoji for watches."""
        return "⌚"
    
    def get_custom_metrics(self) -> List[str]:
        """Return watch-specific metrics."""
        return ["brand_premium", "model_rarity", "condition_factor", "market_demand"]
    
    def render_overview(self) -> None:
        """Render watch-specific overview."""
        st.title("⌚ Luxury Watch Market Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Watches",
                value="1,234",
                delta="12"
            )
        
        with col2:
            st.metric(
                label="Avg Price (SGD)",
                value="$15,450",
                delta="2.5%"
            )
        
        with col3:
            st.metric(
                label="Market Trend",
                value="Bullish",
                delta="Positive"
            )
        
        with col4:
            st.metric(
                label="Active Brands",
                value="25",
                delta="3"
            )
        
        # Brand distribution
        st.subheader("Brand Market Share")
        
        # Mock brand data
        brand_data = pd.DataFrame({
            'Brand': ['Rolex', 'Omega', 'Tudor', 'Seiko', 'Cartier', 'Others'],
            'Market Share': [35, 15, 12, 10, 8, 20],
            'Avg Price': [25000, 8000, 4500, 1200, 15000, 6000]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                brand_data, 
                values='Market Share', 
                names='Brand',
                title="Market Share by Brand"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(
                brand_data,
                x='Brand',
                y='Avg Price',
                title="Average Price by Brand",
                color='Avg Price',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Recent price movements
        st.subheader("Recent Price Movements")
        self._render_price_trends()
    
    def render_analysis(self) -> None:
        """Render detailed watch analysis."""
        st.title("⌚ Watch Market Analysis")
        
        # Asset selector
        selected_asset = self.render_asset_selector()
        
        if not selected_asset:
            return
        
        # Filters
        filters = self.render_common_filters()
        
        # Price history
        st.subheader("Price History")
        
        days = int(filters['time_range'].split()[0])
        df = self.fetch_asset_history(selected_asset, days)
        
        if not df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['price'],
                mode='lines',
                name='Price',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title=f"Price History - {selected_asset}",
                xaxis_title="Date",
                yaxis_title=f"Price ({filters['currency']})",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical analysis
            self._render_technical_analysis(df)
            
            # Watch-specific metrics
            self._render_watch_metrics(selected_asset)
        else:
            st.warning("No price data available for selected watch.")
    
    def render_forecasting(self) -> None:
        """Render forecasting interface."""
        st.title("⌚ Watch Price Forecasting")
        
        # Asset selector
        selected_asset = self.render_asset_selector()
        
        if not selected_asset:
            return
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Forecast Settings")
            
            forecast_days = st.slider(
                "Forecast Horizon (days)",
                min_value=7,
                max_value=365,
                value=30
            )
            
            model = st.selectbox(
                "Model",
                ["XGBoost", "ARIMA", "SARIMA", "Linear Regression", "Random Forest"]
            )
            
            confidence = st.slider(
                "Confidence Level",
                min_value=0.8,
                max_value=0.99,
                value=0.95,
                step=0.01
            )
            
            if st.button("Generate Forecast", type="primary"):
                st.session_state.forecast_generated = True
        
        with col2:
            if st.session_state.get('forecast_generated', False):
                self._render_forecast_results(selected_asset, forecast_days, model, confidence)
    
    def _render_price_trends(self):
        """Render recent price trend chart."""
        # Mock trending data
        trending_watches = [
            {"name": "Rolex Submariner", "change": 2.5, "price": 15000},
            {"name": "Omega Speedmaster", "change": -1.2, "price": 8500},
            {"name": "Tudor Black Bay", "change": 3.8, "price": 4500},
            {"name": "Seiko Prospex", "change": 0.5, "price": 1200}
        ]
        
        for watch in trending_watches:
            delta_color = "normal" if watch["change"] >= 0 else "inverse"
            st.metric(
                label=watch["name"],
                value=f"${watch['price']:,}",
                delta=f"{watch['change']:+.1f}%",
                delta_color=delta_color
            )
    
    def _render_technical_analysis(self, df: pd.DataFrame):
        """Render technical analysis indicators."""
        st.subheader("Technical Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Moving averages
            df['MA_7'] = df['price'].rolling(window=7).mean()
            df['MA_30'] = df['price'].rolling(window=30).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['date'], y=df['price'], name='Price', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['date'], y=df['MA_7'], name='7-day MA', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df['date'], y=df['MA_30'], name='30-day MA', line=dict(color='red')))
            
            fig.update_layout(title="Moving Averages", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Volatility
            df['returns'] = df['price'].pct_change()
            df['volatility'] = df['returns'].rolling(window=7).std() * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['date'], y=df['volatility'], name='7-day Volatility', line=dict(color='purple')))
            fig.update_layout(title="Price Volatility (%)", xaxis_title="Date", yaxis_title="Volatility")
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_watch_metrics(self, asset_id: str):
        """Render watch-specific metrics."""
        st.subheader("Watch-Specific Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Brand Premium", "25%", "2%")
            st.metric("Model Rarity", "High", "Stable")
        
        with col2:
            st.metric("Condition Factor", "98%", "1%")
            st.metric("Market Demand", "Strong", "Growing")
        
        with col3:
            st.metric("Investment Grade", "A+", "Upgraded")
            st.metric("Liquidity Score", "85/100", "5")
    
    def _render_forecast_results(self, asset_id: str, days: int, model: str, confidence: float):
        """Render forecast results."""
        st.subheader(f"Forecast Results - {model}")
        
        # Mock forecast data
        import numpy as np
        
        historical_df = self.fetch_asset_history(asset_id, 30)
        
        if not historical_df.empty:
            last_price = historical_df['price'].iloc[-1]
            
            # Generate forecast
            future_dates = pd.date_range(
                start=historical_df['date'].iloc[-1] + timedelta(days=1),
                periods=days,
                freq='D'
            )
            
            # Mock forecast with trend and noise
            trend = np.linspace(0, days * 2, days)
            noise = np.random.randn(days) * 50
            forecast_prices = last_price + trend + noise
            
            # Confidence intervals
            lower_bound = forecast_prices * (1 - (1 - confidence) / 2)
            upper_bound = forecast_prices * (1 + (1 - confidence) / 2)
            
            # Create forecast plot
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical_df['date'],
                y=historical_df['price'],
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=forecast_prices,
                mode='lines',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=upper_bound,
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Upper Bound'
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name=f'{confidence*100:.0f}% Confidence',
                fillcolor='rgba(255,0,0,0.2)'
            ))
            
            fig.update_layout(
                title=f"Price Forecast - {days} Days",
                xaxis_title="Date",
                yaxis_title="Price (SGD)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Expected Price (30 days)",
                    f"${forecast_prices[-1]:,.0f}",
                    f"{((forecast_prices[-1] - last_price) / last_price * 100):+.1f}%"
                )
            
            with col2:
                st.metric(
                    "Confidence Range",
                    f"${lower_bound[-1]:,.0f} - ${upper_bound[-1]:,.0f}",
                    f"±{((upper_bound[-1] - lower_bound[-1]) / 2 / last_price * 100):.1f}%"
                )
            
            with col3:
                st.metric(
                    "Model Accuracy",
                    "92.5%",
                    "High"
                )