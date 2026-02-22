"""Evaluation dashboard with metrics visualization"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys

# Add src to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)


def load_results(results_dir: Path):
    """Load evaluation results from directory"""
    results = []
    
    if not results_dir.exists():
        return pd.DataFrame()
    
    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                data["file"] = json_file.name
                results.append(data)
        except Exception as e:
            st.error(f"Error loading {json_file.name}: {e}")
    
    return pd.DataFrame(results) if results else pd.DataFrame()


def main():
    st.title("📊 RAG Evaluation Dashboard")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        results_dir = st.text_input(
            "Results Directory",
            value=str(project_root / "results")
        )
        
        metric_type = st.selectbox(
            "Metric Type",
            ["Safety Metrics", "RAG Metrics", "Comparison"]
        )
    
    results_path = Path(results_dir)
    
    if metric_type == "Safety Metrics":
        st.header("🛡️ Safety Metrics")
        
        safety_dir = results_path / "safety"
        df = load_results(safety_dir)
        
        if df.empty:
            st.warning("No safety metrics found. Run adversarial tests to generate data.")
        else:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            latest = df.iloc[-1] if not df.empty else {}
            
            with col1:
                asr = latest.get("attack_success_rate", 0)
                st.metric(
                    "Attack Success Rate",
                    f"{asr:.1%}",
                    delta=f"{'✓ Good' if asr < 0.05 else '⚠️ High'}",
                    delta_color="inverse"
                )
            
            with col2:
                fpr = latest.get("false_positive_rate", 0)
                st.metric(
                    "False Positive Rate",
                    f"{fpr:.1%}",
                    delta=f"{'✓ Good' if fpr < 0.10 else '⚠️ High'}",
                    delta_color="inverse"
                )
            
            with col3:
                hr = latest.get("hallucination_rate", 0)
                st.metric(
                    "Hallucination Rate",
                    f"{hr:.1%}",
                    delta=f"{'✓ Good' if hr < 0.10 else '⚠️ High'}",
                    delta_color="inverse"
                )
            
            with col4:
                pii = latest.get("pii_leakage_rate", 0)
                st.metric(
                    "PII Leakage Rate",
                    f"{pii:.1%}",
                    delta=f"{'✓ Good' if pii == 0 else '🚨 ALERT'}",
                    delta_color="inverse"
                )
            
            # Trends
            st.subheader("Metric Trends")
            
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["timestamp"],
                    y=df["attack_success_rate"] * 100,
                    name="Attack Success Rate",
                    line=dict(color="red")
                ))
                fig.add_trace(go.Scatter(
                    x=df["timestamp"],
                    y=df["false_positive_rate"] * 100,
                    name="False Positive Rate",
                    line=dict(color="orange")
                ))
                fig.add_trace(go.Scatter(
                    x=df["timestamp"],
                    y=df["hallucination_rate"] * 100,
                    name="Hallucination Rate",
                    line=dict(color="purple")
                ))
                
                fig.update_layout(
                    title="Safety Metrics Over Time",
                    xaxis_title="Date",
                    yaxis_title="Rate (%)",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    elif metric_type == "RAG Metrics":
        st.header("📈 RAG Quality Metrics")
        
        golden_dir = results_path / "golden"
        df = load_results(golden_dir)
        
        if df.empty:
            st.warning("No RAG metrics found. Run golden dataset evaluation to generate data.")
        else:
            latest = df.iloc[-1] if not df.empty else {}
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Faithfulness",
                    f"{latest.get('avg_faithfulness', 0):.2%}"
                )
            
            with col2:
                st.metric(
                    "Answer Relevance",
                    f"{latest.get('avg_answer_relevance', 0):.2%}"
                )
            
            with col3:
                st.metric(
                    "Context Precision",
                    f"{latest.get('avg_context_precision', 0):.2%}"
                )
            
            with col4:
                st.metric(
                    "Overall Score",
                    f"{latest.get('overall_score', 0):.2%}"
                )
            
            # Latency
            st.subheader("⚡ Performance")
            avg_latency = latest.get("avg_latency_ms", 0)
            st.metric("Average Latency", f"{avg_latency:.0f} ms")
            
            # Show raw data
            st.subheader("Raw Data")
            st.dataframe(df)
    
    elif metric_type == "Comparison":
        st.header("⚖️ Model Comparison: RAG vs Fine-tuned")
        
        comparison_files = list(results_path.glob("comparison_*.json"))
        
        if not comparison_files:
            st.warning("No comparison results found.")
        else:
            # Load latest comparison
            latest_file = sorted(comparison_files)[-1]
            
            with open(latest_file, 'r') as f:
                comparison = json.load(f)
            
            rag = comparison.get("rag_metrics", {})
            ft = comparison.get("finetuned_metrics", {})
            winners = comparison.get("winner_by_metric", {})
            
            if ft:
                # Comparison table
                st.subheader("Head-to-Head Comparison")
                
                comparison_data = {
                    "Metric": ["Accuracy", "Faithfulness", "Latency (p95)", "Cost per 1K", "Freshness", "Attack Success Rate"],
                    "RAG": [
                        rag.get("accuracy", "N/A"),
                        rag.get("faithfulness", "N/A"),
                        rag.get("latency_p95_ms", "N/A"),
                        rag.get("cost_per_1k", "N/A"),
                        rag.get("freshness_score", "N/A"),
                        rag.get("attack_success_rate", "N/A"),
                    ],
                    "Fine-tuned": [
                        ft.get("accuracy", "N/A"),
                        ft.get("faithfulness", "N/A"),
                        ft.get("latency_p95_ms", "N/A"),
                        ft.get("cost_per_1k", "N/A"),
                        ft.get("freshness_score", "N/A"),
                        ft.get("attack_success_rate", "N/A"),
                    ],
                    "Winner": [
                        winners.get("accuracy", ""),
                        winners.get("accuracy", ""),
                        winners.get("latency", ""),
                        winners.get("cost", ""),
                        winners.get("freshness", ""),
                        winners.get("safety", ""),
                    ]
                }
                
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
                
                # Summary
                st.subheader("Summary")
                rag_wins = sum(1 for w in winners.values() if w == "RAG")
                ft_wins = sum(1 for w in winners.values() if w == "Fine-tuned")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("RAG Wins", rag_wins)
                with col2:
                    st.metric("Fine-tuned Wins", ft_wins)
            else:
                st.info("Only RAG metrics available. Fine-tuned model not evaluated yet.")
                
                st.subheader("RAG Metrics")
                st.json(rag)


if __name__ == "__main__":
    main()
