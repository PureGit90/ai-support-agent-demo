import os
import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import anthropic
from datetime import datetime, timedelta
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Support QA Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }
    .metric-delta-pos {
        font-size: 0.9rem;
        color: #16a34a;
        font-weight: 600;
    }
    .metric-delta-neg {
        font-size: 0.9rem;
        color: #dc2626;
        font-weight: 600;
    }
    .failure-badge {
        display: inline-block;
        padding: 0.2rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .severity-high   { background:#fee2e2; color:#dc2626; }
    .severity-medium { background:#fef9c3; color:#b45309; }
    .severity-low    { background:#dcfce7; color:#16a34a; }
    .analysis-box {
        background: #f0f9ff;
        border-left: 4px solid #0284c7;
        border-radius: 4px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .prompt-fix-box {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        border-radius: 4px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        font-family: monospace;
        white-space: pre-wrap;
        font-size: 0.88rem;
    }
    .workflow-fix-box {
        background: #fdf4ff;
        border-left: 4px solid #9333ea;
        border-radius: 4px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .impact-box {
        background: #fff7ed;
        border-left: 4px solid #ea580c;
        border-radius: 4px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stTabs"] button {
        font-size: 0.95rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ── Example conversations ─────────────────────────────────────────────────────
EXAMPLE_CONVERSATIONS = {
    "Example 1 — Return Request (Wrong Policy)": """Customer: Hi, I want to return a jacket I bought 45 days ago. It has a defect in the zipper.
AI Agent: I can help with returns! Our return policy allows returns within 30 days of purchase. Unfortunately, since your order is past that window, I'm unable to process a return for you.
Customer: But the zipper is broken — that's a product defect, not a change of mind.
AI Agent: I understand your frustration, but our policy is 30 days regardless of the reason. I'm unable to make exceptions.
Customer: This is ridiculous. Can I speak to a human?
AI Agent: Of course! Let me transfer you to our team. Have a great day!
[Ticket escalated. Customer waited 2 hours. Human agent approved defect return immediately under manufacturer warranty policy.]""",

    "Example 2 — WISMO (Hallucinated Tracking)": """Customer: Where is my order #84291? It's been 9 days and I haven't received anything.
AI Agent: Hi! Let me check on that for you. Your order #84291 is currently in transit and was last scanned in Chicago, IL on May 18th. Estimated delivery is May 21st.
Customer: That's tomorrow. But the tracking link in my email says "label created" and it's never moved.
AI Agent: I apologize for the confusion. The system shows it is moving through our fulfillment network. Please allow 1-2 more business days.
Customer: I'm looking at the carrier site RIGHT NOW and it says label created 9 days ago. The package was never picked up.
AI Agent: I understand your concern. Please allow additional time for the carrier to update tracking information.
Customer: You literally made up that Chicago information. I want a refund.
[Customer churned. Order was stuck in warehouse due to inventory error — never shipped.]""",

    "Example 3 — Exchange (Wrong Action Offered)": """Customer: I ordered a size M blue hoodie but received a size S. I'd like to exchange it for the correct size.
AI Agent: I'm sorry to hear that! I can process a refund for the incorrect item. Would you like me to initiate that now?
Customer: No, I don't want a refund — I want the right hoodie. Can you just send me a medium?
AI Agent: Our policy is to issue a refund for incorrect items. Once you receive the refund in 5-7 business days, you can place a new order.
Customer: That's awful. The medium might sell out. Why can't you just send the right item?
AI Agent: I understand that's frustrating. I can only process refunds through this channel. Is there anything else I can help you with?
Customer: No. I'll dispute the charge with my bank.
[Customer filed chargeback. Human agent later confirmed exchanges are supported — AI agent lacked the workflow and authority to process them.]""",
}

# ── Pre-seeded playbook data ──────────────────────────────────────────────────
PLAYBOOK_DATA = [
    {
        "Date": "2026-05-01",
        "Failure Type": "Hallucination",
        "Root Cause Summary": "AI fabricated tracking location when order lookup returned no live carrier data",
        "Fix Applied": "Added explicit instruction: 'If tracking status is label_created for >72h, do not state a location. Trigger order_status_check tool and offer proactive reship.'",
        "Impact": "~20% reduction in WISMO escalations",
    },
    {
        "Date": "2026-04-24",
        "Failure Type": "Wrong Policy",
        "Root Cause Summary": "AI applied standard 30-day window to defective items; warranty exceptions not in prompt context",
        "Fix Applied": "Injected defect/warranty policy into system prompt; added conditional: 'If customer reports physical defect, apply manufacturer warranty policy (90 days) regardless of purchase date.'",
        "Impact": "~35% reduction in defect-related escalations",
    },
    {
        "Date": "2026-04-17",
        "Failure Type": "Wrong Action",
        "Root Cause Summary": "Exchange workflow missing from AI toolset; defaulted to refund-only path",
        "Fix Applied": "Added exchange_order tool with size/color swap capability; updated routing: 'Offer exchange first for wrong-item cases before offering refund.'",
        "Impact": "~40% reduction in wrong-item chargebacks",
    },
    {
        "Date": "2026-04-10",
        "Failure Type": "Bad Escalation",
        "Root Cause Summary": "AI escalated after 1 failed resolution attempt; no retry logic for order lookup timeouts",
        "Fix Applied": "Added retry logic: attempt order lookup 2x with 3s delay before escalating; added fallback message with estimated wait and self-service link.",
        "Impact": "~15% reduction in unnecessary escalations",
    },
    {
        "Date": "2026-04-03",
        "Failure Type": "Incomplete Resolution",
        "Root Cause Summary": "Discount code applied but order total mismatch not caught; customer received incorrect savings",
        "Fix Applied": "Added post-action validation step: verify applied discount matches advertised value before confirming to customer; surface discrepancy to human review queue.",
        "Impact": "~25% reduction in discount-related re-contacts",
    },
]

# ── Metrics mock data ─────────────────────────────────────────────────────────
def build_metrics():
    weeks = [f"Wk {i}" for i in range(1, 9)]
    resolution_rates = [62, 64, 65, 67, 69, 71, 73, 74]
    escalation_rates = [38, 36, 35, 33, 31, 29, 27, 26]
    failure_types = {
        "Wrong Policy": 24,
        "Hallucination": 19,
        "Wrong Action": 18,
        "Bad Escalation": 16,
        "Missing Data": 13,
        "Incomplete Resolution": 10,
    }
    return weeks, resolution_rates, escalation_rates, failure_types

# ── Claude analysis ───────────────────────────────────────────────────────────
ANALYSIS_SYSTEM_PROMPT = """You are a senior AI support quality analyst specializing in ecommerce customer service automation.

Your job is to analyze failed AI customer support conversations and identify exactly what went wrong, why, and how to fix it.

You must respond with a valid JSON object with these exact keys:
- failure_type: one of ["Wrong Policy", "Missing Data", "Bad Escalation", "Hallucination", "Incomplete Resolution", "Wrong Action"]
- root_cause: 2-3 sentences explaining the specific failure
- severity: one of ["Low", "Medium", "High"]
- prompt_fix: a specific rewrite of the instruction or system prompt section that would prevent this failure (write it as an actual instruction, not a description of one)
- workflow_fix: a structural or tooling change needed (e.g. adding a lookup tool, a routing rule, or a retry mechanism). If none is needed, write "None required."
- estimated_impact: a short quantified estimate of how often this fix would prevent similar failures (e.g. "Would prevent ~20% of WISMO escalations")

Be specific and technical. The output will be used directly by an engineering team to improve prompt configuration and workflow design.

Respond with ONLY the JSON object — no markdown fences, no preamble."""

def run_analysis(conversation: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {
            "failure_type": "Wrong Policy",
            "root_cause": "ANTHROPIC_API_KEY not set. This is a placeholder result. Add your API key to enable live analysis.",
            "severity": "Medium",
            "prompt_fix": "Set the ANTHROPIC_API_KEY environment variable to enable real-time Claude analysis.",
            "workflow_fix": "None required.",
            "estimated_impact": "N/A — API key missing",
            "_demo_mode": True,
        }

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=ANALYSIS_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Analyze this failed AI support conversation:\n\n{conversation}",
            }
        ],
    )
    raw = message.content[0].text.strip()
    # Strip markdown fences if model adds them anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


# ── Severity badge helper ─────────────────────────────────────────────────────
def severity_badge(s: str) -> str:
    css_class = {"High": "severity-high", "Medium": "severity-medium", "Low": "severity-low"}.get(s, "severity-low")
    return f'<span class="failure-badge {css_class}">{s}</span>'


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">AI Support QA Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Identify failure patterns in AI customer support conversations, surface root causes, and generate targeted prompt & workflow fixes.</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Upload / Paste", "Analysis", "Playbook", "Metrics"])

# ── Tab 1: Upload / Paste ─────────────────────────────────────────────────────
with tab1:
    st.subheader("Load a Conversation")
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### Use a pre-loaded example")
        selected_example = st.selectbox(
            "Choose an example failed conversation:",
            ["(none — I'll paste my own)"] + list(EXAMPLE_CONVERSATIONS.keys()),
            key="example_select",
        )
        # Sync dropdown change into the text area's session state key
        if selected_example != "(none — I'll paste my own)":
            if st.session_state.get("_last_example_select") != selected_example:
                st.session_state["conversation_text"] = EXAMPLE_CONVERSATIONS[selected_example]
                st.session_state["_last_example_select"] = selected_example
        else:
            if st.session_state.get("_last_example_select") is not None:
                st.session_state["conversation_text"] = ""
                st.session_state["_last_example_select"] = None

    with col_right:
        st.markdown("#### Or paste a conversation")
        conversation_input = st.text_area(
            "Paste the full conversation transcript here:",
            height=320,
            key="conversation_text",
            placeholder="Customer: ...\nAI Agent: ...\nCustomer: ...",
        )

    st.markdown("---")

    if conversation_input.strip():
        st.success(f"Conversation loaded — {len(conversation_input.split())} words. Switch to the **Analysis** tab and click Analyze.")
    else:
        st.info("Select a pre-loaded example or paste a conversation above, then move to the **Analysis** tab.")


# ── Tab 2: Analysis ───────────────────────────────────────────────────────────
with tab2:
    st.subheader("Analyze Failed Conversation")

    conversation = st.session_state.get("conversation_text", "").strip()

    if not conversation:
        st.warning("No conversation loaded. Go to the **Upload / Paste** tab first.")
    else:
        with st.expander("Show loaded conversation", expanded=False):
            st.text(conversation)

        if st.button("Analyze with Claude", type="primary", use_container_width=True):
            with st.spinner("Running analysis..."):
                try:
                    result = run_analysis(conversation)
                    st.session_state["analysis_result"] = result
                except json.JSONDecodeError as e:
                    st.error(f"Could not parse model response as JSON: {e}")
                    st.session_state["analysis_result"] = None
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.session_state["analysis_result"] = None

    result = st.session_state.get("analysis_result")

    if result:
        if result.get("_demo_mode"):
            st.warning("Demo mode — ANTHROPIC_API_KEY not set. Showing placeholder result.")

        st.markdown("---")

        # Header row
        col_a, col_b, col_c = st.columns([2, 2, 1])
        with col_a:
            st.markdown(f"**Failure Type:** `{result.get('failure_type', 'N/A')}`")
        with col_b:
            st.markdown(f"**Severity:** {severity_badge(result.get('severity', 'N/A'))}", unsafe_allow_html=True)
        with col_c:
            pass

        st.markdown("#### Root Cause")
        st.markdown(f'<div class="analysis-box">{result.get("root_cause", "")}</div>', unsafe_allow_html=True)

        st.markdown("#### Prompt Fix")
        st.markdown(
            f'<div class="prompt-fix-box">{result.get("prompt_fix", "")}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Workflow Fix")
        st.markdown(f'<div class="workflow-fix-box">{result.get("workflow_fix", "")}</div>', unsafe_allow_html=True)

        st.markdown("#### Estimated Impact")
        st.markdown(f'<div class="impact-box">📈 {result.get("estimated_impact", "")}</div>', unsafe_allow_html=True)

        # Save to playbook button
        st.markdown("---")
        if st.button("Add to Playbook", use_container_width=False):
            new_entry = {
                "Date": datetime.today().strftime("%Y-%m-%d"),
                "Failure Type": result.get("failure_type", "Unknown"),
                "Root Cause Summary": result.get("root_cause", "")[:120] + "...",
                "Fix Applied": result.get("prompt_fix", "")[:120] + "...",
                "Impact": result.get("estimated_impact", "TBD"),
            }
            if "custom_playbook" not in st.session_state:
                st.session_state["custom_playbook"] = []
            st.session_state["custom_playbook"].append(new_entry)
            st.success("Added to Playbook. Switch to the Playbook tab to view.")


# ── Tab 3: Playbook ───────────────────────────────────────────────────────────
with tab3:
    st.subheader("Fix Playbook")
    st.markdown("A running record of identified failure patterns and the fixes applied. Each entry represents a shipped improvement to prompt configuration or workflow logic.")

    custom = st.session_state.get("custom_playbook", [])
    all_entries = PLAYBOOK_DATA + custom

    df = pd.DataFrame(all_entries)
    # Color-code Failure Type column
    def color_type(val):
        colors = {
            "Wrong Policy":         "background-color: #fee2e2; color: #dc2626;",
            "Hallucination":        "background-color: #fef9c3; color: #b45309;",
            "Wrong Action":         "background-color: #ede9fe; color: #7c3aed;",
            "Bad Escalation":       "background-color: #fce7f3; color: #be185d;",
            "Missing Data":         "background-color: #e0f2fe; color: #0369a1;",
            "Incomplete Resolution":"background-color: #dcfce7; color: #15803d;",
        }
        return colors.get(val, "")

    styled = df.style.map(color_type, subset=["Failure Type"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">23</div><div class="metric-label">Total Fixes Shipped</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">+12%</div><div class="metric-label">Est. Resolution Rate Lift</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">6</div><div class="metric-label">Failure Types Covered</div></div>',
            unsafe_allow_html=True,
        )


# ── Tab 4: Metrics ────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Performance Dashboard")
    st.markdown("8-week trend since QA optimization program launched.")

    weeks, resolution_rates, escalation_rates, failure_types = build_metrics()

    # Top stat cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">74%</div><div class="metric-label">Current Resolution Rate</div><div class="metric-delta-pos">+12pp since Week 1</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">26%</div><div class="metric-label">Current Escalation Rate</div><div class="metric-delta-neg">-12pp since Week 1</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">23</div><div class="metric-label">Fixes Applied</div><div class="metric-delta-pos">+5 this month</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">80%</div><div class="metric-label">Target Resolution Rate</div><div class="metric-label">ETA: 3-4 weeks</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Line charts
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=weeks,
            y=resolution_rates,
            mode="lines+markers",
            name="Resolution Rate",
            line=dict(color="#0284c7", width=3),
            marker=dict(size=8, color="#0284c7"),
            fill="tozeroy",
            fillcolor="rgba(2,132,199,0.08)",
        ))
        fig_res.add_hline(
            y=80,
            line_dash="dot",
            line_color="#dc2626",
            annotation_text="Target 80%",
            annotation_position="bottom right",
        )
        fig_res.update_layout(
            title="Resolution Rate (%) — 8 Weeks",
            yaxis=dict(range=[55, 85], ticksuffix="%"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=50, b=20),
            height=300,
        )
        st.plotly_chart(fig_res, use_container_width=True)

    with col_right:
        fig_esc = go.Figure()
        fig_esc.add_trace(go.Scatter(
            x=weeks,
            y=escalation_rates,
            mode="lines+markers",
            name="Escalation Rate",
            line=dict(color="#dc2626", width=3),
            marker=dict(size=8, color="#dc2626"),
            fill="tozeroy",
            fillcolor="rgba(220,38,38,0.06)",
        ))
        fig_esc.update_layout(
            title="Escalation Rate (%) — 8 Weeks",
            yaxis=dict(range=[15, 45], ticksuffix="%"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=50, b=20),
            height=300,
        )
        st.plotly_chart(fig_esc, use_container_width=True)

    st.markdown("---")

    # Failure type breakdown
    col_pie, col_notes = st.columns([1, 1], gap="large")

    with col_pie:
        labels = list(failure_types.keys())
        values = list(failure_types.values())
        colors = ["#dc2626", "#b45309", "#7c3aed", "#be185d", "#0369a1", "#15803d"]

        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker=dict(colors=colors),
            textinfo="label+percent",
            textfont_size=12,
        )])
        fig_pie.update_layout(
            title="Failure Type Breakdown",
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10),
            height=340,
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_notes:
        st.markdown("#### Insights")
        st.markdown("""
**Wrong Policy** is the leading failure type (24%). Most cases involve defective item returns being rejected under the standard 30-day window — fixable with a single prompt amendment.

**Hallucination** (19%) is concentrated in WISMO flows where the order lookup tool is unavailable or times out. Adding a fallback guard clause eliminates this entirely.

**Wrong Action** (18%) primarily affects exchanges — the AI defaults to refunds because the exchange workflow was not available in its toolset at launch.

**Trend:** Week-over-week improvement of ~1.5pp resolution rate is on track to hit the **80% target within 3-4 weeks** if the next batch of 6 fixes ships as scheduled.
""")

        st.markdown("#### Weekly Fix Velocity")
        fix_weeks  = ["Wk 1-2", "Wk 3-4", "Wk 5-6", "Wk 7-8"]
        fix_counts = [4, 7, 7, 5]
        fig_bar = go.Figure(go.Bar(
            x=fix_weeks,
            y=fix_counts,
            marker_color="#0284c7",
        ))
        fig_bar.update_layout(
            yaxis_title="Fixes Shipped",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10),
            height=200,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
