import streamlit as st
import requests
import json
from typing import Dict, Any
import time
import pandas as pd
import plotly.express as px
from datetime import datetime

# Custom CSS for enhanced styling
def load_css():
    st.markdown("""
        <style>
        .big-font {
            font-size: 24px !important;
            font-weight: bold;
            color: #1E88E5;
        }
        .card {
            padding: 20px;
            border-radius: 10px;
            background: #ffffff;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .status-success {
            color: #4CAF50;
            font-weight: bold;
        }
        .status-warning {
            color: #FFC107;
            font-weight: bold;
        }
        .status-danger {
            color: #F44336;
            font-weight: bold;
        }
        .sidebar-content {
            padding: 20px;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            transition: transform 0.3s ease;
            color: black
        }
        .metric-card:hover {
            transform: translateY(-5px);
        }
        </style>
    """, unsafe_allow_html=True)

def send_transaction(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Send transaction data to the fraud detection API."""
    url = "https://kairo-backend.onrender.com/api/transactions/"
    try:
        response = requests.post(url, json=transaction)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")

def create_mock_history():
    """Create mock transaction history for visualization."""
    dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='D')
    data = {
        'Date': dates,
        'Transactions': np.random.randint(50, 200, size=len(dates)),
        'Risk Score': np.random.uniform(0, 1, size=len(dates))
    }
    return pd.DataFrame(data)

def display_transaction_history(df):
    """Display transaction history visualization."""
    fig = px.line(df, x='Date', y=['Transactions', 'Risk Score'],
                  title='Transaction History and Risk Trends')
    st.plotly_chart(fig, use_container_width=True)

def main():
    # Page configuration
    st.set_page_config(
        page_title="KAIRO - Fraud Detection UI",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_css()

    # Main layout
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown('<p class="big-font">KAIRO</p>', unsafe_allow_html=True)
        st.markdown("### Advanced Fraud Detection System")

        # Mock metrics
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.markdown("""
                <div class="metric-card">
                    <h4>Today's Transactions</h4>
                    <p class="big-font">157</p>
                </div>
            """, unsafe_allow_html=True)
        with metrics_col2:
            st.markdown("""
                <div class="metric-card">
                    <h4 style="color:red">Risk Level</h4>
                    <p class="big-font">Low</p>
                </div>
            """, unsafe_allow_html=True)

    # Sidebar inputs with enhanced styling
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("### Transaction Details")

        user_id = st.text_input(
            "User ID",
            value="12345",
            help="Enter the unique identifier for the user"
        )

        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.01,
            value=100.0,
            help="Enter the transaction amount in dollars"
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            options=["Purchase", "Withdrawal", "Transfer"],
            help="Select the type of transaction"
        )

        col1, col2 = st.columns(2)
        with col1:
            location = st.text_input(
                "Location",
                value="New York",
                help="Enter the transaction location"
            )
        with col2:
            merchant = st.text_input(
                "Merchant",
                value="Amazon",
                help="Enter the merchant name"
            )

        with st.expander("Advanced Details"):
            device_info = st.text_area(
                "Device Info (JSON)",
                value="{}",
                help="Enter device information in valid JSON format"
            )

            ip_address = st.text_input(
                "IP Address",
                value="192.168.1.1",
                help="Enter the IP address of the transaction"
            )

            description = st.text_area(
                "Transaction Description",
                value="Online purchase of electronics.",
                help="Enter a description of the transaction"
            )

        submit_button = st.button("Analyze Transaction", use_container_width=True)

    # Main content area
    if submit_button:
        try:
            device_info_dict = json.loads(device_info)

            transaction = {
                "user_id": user_id,
                "amount": amount,
                "transaction_type": transaction_type,
                "location": location,
                "merchant": merchant,
                "device_info": device_info_dict,
                "ip_address": ip_address,
                "description": description,
                "timestamp": datetime.now().isoformat()
            }

            # Animated analysis process
            with st.spinner(""):
                progress_text = "Analyzing transaction..."
                my_bar = st.progress(0)

                # Simulate analysis steps
                for i in range(100):
                    time.sleep(0.01)
                    my_bar.progress(i + 1, text=progress_text)

                response = send_transaction(transaction)

                # Mock response for demonstration
                risk_score = response.get("risk_score", 0.7)

                # Display result with animation
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Analysis Results")

                # Risk level indicator
                risk_level = "Low" if risk_score < 0.3 else "Medium" if risk_score < 0.7 else "High"
                risk_color = ("status-success" if risk_score < 0.3
                             else "status-warning" if risk_score < 0.7
                             else "status-danger")

                st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4>Risk Score</h4>
                            <p class="{risk_color}">{risk_score:.2f}</p>
                        </div>
                        <div>
                            <h4>Risk Level</h4>
                            <p class="{risk_color}">{risk_level}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Display detailed results
                st.json(response)
                st.markdown('</div>', unsafe_allow_html=True)

                # Display transaction history
                st.subheader("Transaction History")
                df = create_mock_history()
                display_transaction_history(df)

        except json.JSONDecodeError:
            st.error("Error: Invalid JSON format in Device Info field")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
