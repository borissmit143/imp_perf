"""Streamlit app for digital-twin importance/performance analysis."""

from __future__ import annotations

import asyncio
import re
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


APP_DIR = Path(__file__).resolve().parent
DEFAULT_USERS_FILE = APP_DIR / "users.xlsx"
MODEL_NAME = "gemini-3.1-flash-lite"


class TwinRatings(BaseModel):
    importance: list[int] = Field(
        description="Importance ratings in the exact attribute order, integers from 1 to 10."
    )
    performance: list[int] = Field(
        description="Performance ratings in the exact attribute order, integers from 1 to 10."
    )


def clean_value(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


@st.cache_data(show_spinner=False)
def load_default_demographics() -> pd.DataFrame:
    return pd.read_excel(DEFAULT_USERS_FILE)


def load_demographics(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return load_default_demographics()
    if Path(uploaded_file.name).suffix.lower() == ".csv":
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def matching_column(dataframe: pd.DataFrame, wanted: str) -> str | None:
    names = {str(column).strip().casefold(): str(column) for column in dataframe.columns}
    return names.get(wanted.casefold())


def age_group_for(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    age = pd.to_numeric(value, errors="coerce")
    if pd.isna(age):
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if not numbers:
            return text or None
        age = float(numbers[0])
    if age < 18:
        return "Under 18"
    if age <= 24:
        return "18-24"
    if age <= 34:
        return "25-34"
    if age <= 44:
        return "35-44"
    if age <= 54:
        return "45-54"
    if age <= 64:
        return "55-64"
    return "65+"


def filter_demographics(dataframe: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Choose digital twins")
    age_column = matching_column(dataframe, "Age")
    gender_column = matching_column(dataframe, "Gender")
    location_column = matching_column(dataframe, "Location")
    columns = st.columns(3)
    mask = pd.Series(True, index=dataframe.index)

    for container, label, source_column, transformed in (
        (columns[0], "Age group", age_column, True),
        (columns[1], "Location", location_column, False),
        (columns[2], "Gender", gender_column, False),
    ):
        if not source_column:
            container.caption(f"{label} filter unavailable.")
            continue
        values = dataframe[source_column].map(age_group_for if transformed else clean_value)
        choices = sorted(set(values.dropna()), key=str)
        selected = container.multiselect(label, choices, placeholder=f"All {label.lower()}s")
        if selected:
            mask &= values.isin(selected)

    matching = dataframe.loc[mask].copy()
    if matching.empty:
        st.warning("No digital twins match the selected filters.")
        return matching
    count = int(
        st.number_input(
            "Number of digital twins to simulate",
            min_value=1,
            max_value=len(matching),
            value=len(matching),
            step=1,
        )
    )
    if count < len(matching):
        matching = matching.sample(count, random_state=42).sort_index()
    return matching


def configured_api_key() -> str:
    """Load the Google API key exclusively from Streamlit secrets."""
    try:
        return str(st.secrets["GOOGLE_API_KEY"])
    except (KeyError, FileNotFoundError):
        return ""


def make_prompt(service_name: str, description: str, attributes: list[str], profile: dict) -> str:
    profile_lines = "\n".join(f"- {key}: {value}" for key, value in profile.items())
    attribute_lines = "\n".join(f"{number}. {attribute}" for number, attribute in enumerate(attributes, 1))
    return f"""You are a digital twin representing a customer with this demographic profile:
{profile_lines}

Service: {service_name}
Service description: {description}

Attributes, in required response order:
{attribute_lines}

First rate how IMPORTANT each attribute is to you when evaluating this service.
Then assume you have personally consumed the service exactly as described and rate how well
the service PERFORMED on each attribute. Use integers from 1 to 10, where 1 means very
unimportant/very bad and 10 means extremely important/excellent. Be consistent with the
customer profile without stereotyping or inventing additional facts. Return exactly one
importance rating and one performance rating for every attribute, preserving the order."""


async def query_twin(index, row, service_name, description, attributes, llm, semaphore):
    profile = {
        str(column): value
        for column, raw in row.items()
        if (value := clean_value(raw)) is not None
    }
    prompt = make_prompt(service_name, description, attributes, profile)
    async with semaphore:
        for attempt in range(5):
            try:
                answer = await llm.ainvoke(prompt)
                importance = [int(value) for value in answer.importance]
                performance = [int(value) for value in answer.performance]
                if len(importance) != len(attributes) or len(performance) != len(attributes):
                    raise ValueError("The model returned the wrong number of ratings.")
                if any(not 1 <= value <= 10 for value in importance + performance):
                    raise ValueError("A model rating was outside the 1-10 range.")
                return index, profile, importance, performance, "Success"
            except Exception as exc:
                if attempt == 4:
                    return index, profile, [], [], f"Error: {exc}"
                await asyncio.sleep(2**attempt)


async def run_survey(demographics, service_name, description, attributes, api_key, model_name):
    llm = ChatGoogleGenerativeAI(
        model=model_name, temperature=0.5, google_api_key=api_key
    ).with_structured_output(TwinRatings)
    semaphore = asyncio.Semaphore(30)
    tasks = [
        query_twin(i, row, service_name, description, attributes, llm, semaphore)
        for i, (_, row) in enumerate(demographics.iterrows(), 1)
    ]
    records = []
    progress = st.progress(0, text="Preparing digital twins...")
    for completed, task in enumerate(asyncio.as_completed(tasks), 1):
        twin_id, profile, importance, performance, status = await task
        record = {"Twin ID": twin_id, **profile, "Status": status}
        for position, attribute in enumerate(attributes):
            record[f"Importance - {attribute}"] = importance[position] if importance else None
            record[f"Performance - {attribute}"] = performance[position] if performance else None
        records.append(record)
        progress.progress(completed / len(tasks), text=f"Digital twin {completed} of {len(tasks)}")
    progress.empty()
    return pd.DataFrame(records).sort_values("Twin ID").reset_index(drop=True)


def attribute_summary(results: pd.DataFrame, attributes: list[str]) -> pd.DataFrame:
    rows = []
    for attribute in attributes:
        rows.append(
            {
                "Attribute": attribute,
                "Average Importance": results[f"Importance - {attribute}"].mean(),
                "Average Performance": results[f"Performance - {attribute}"].mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_ipa(summary: pd.DataFrame, service_name: str):
    average_importance = summary["Average Importance"].mean()
    average_performance = summary["Average Performance"].mean()
    figure, axis = plt.subplots(figsize=(16, 9))
    colors = plt.get_cmap("tab20").colors

    for number, row in summary.reset_index(drop=True).iterrows():
        color = colors[number % len(colors)]
        axis.scatter(
            row["Average Importance"],
            row["Average Performance"],
            s=115,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
            label=f"{number + 1}. {row['Attribute']}",
        )
        axis.annotate(
            str(number + 1),
            (row["Average Importance"], row["Average Performance"]),
            ha="center", va="center", color="white", fontsize=8, fontweight="bold",
        )
    axis.axvline(
        average_importance, color="#dc2626", linestyle="--",
        label=f"Mean importance: {average_importance:.2f}",
    )
    axis.axhline(
        average_performance, color="#16a34a", linestyle="--",
        label=f"Mean performance: {average_performance:.2f}",
    )
    axis.set(xlim=(0.5, 10.5), ylim=(0.5, 10.5), xlabel="Importance", ylabel="Performance")
    axis.set_title(f"Importance-Performance Analysis: {service_name}")
    axis.grid(alpha=0.2)
    axis.legend(
        title="Attributes",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title_fontsize=10,
        frameon=True,
    )
    figure.tight_layout()
    return figure


def excel_workbook(results: pd.DataFrame, summary: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="All responses", index=False)
        summary.to_excel(writer, sheet_name="Attribute summary", index=False)
    return output.getvalue()


st.set_page_config(page_title="Importance-Performance Analysis", layout="wide")
st.title("Digital Twin Importance-Performance Analysis")
st.caption("Simulate customer importance and experienced-service performance ratings on a 1-10 scale.")

api_key = configured_api_key()

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader("Optional demographics file", type=["xlsx", "xls", "csv"])

st.subheader("Service")
service_name = st.text_input("Service name", placeholder="e.g., Uber ride hailing")
description = st.text_area(
    "Service description",
    placeholder="Describe what the service offers and the experience customers receive.",
    height=110,
)
attributes_text = st.text_area(
    "Important attributes (10 to 15; one statement per line)",
    placeholder="Price and fare transparency\nShort pickup wait time\nDriver safety and rating\n...",
    height=260,
)
attributes = [line.strip(" -\t") for line in attributes_text.splitlines() if line.strip(" -\t")]
st.caption(f"{len(attributes)} attributes entered; 10-15 required.")

try:
    demographics = load_demographics(uploaded_file).dropna(how="all").reset_index(drop=True)
    selected_demographics = filter_demographics(demographics)
    st.dataframe(selected_demographics, use_container_width=True, height=230)
except Exception as exc:
    st.error(f"Could not load demographics: {exc}")
    st.stop()

if st.button("Run importance-performance survey", type="primary", use_container_width=True):
    missing = [name for name, value in (("service name", service_name), ("service description", description), ("Google API key", api_key)) if not value.strip()]
    if missing:
        st.error("Please provide: " + ", ".join(missing) + ".")
    elif not 10 <= len(attributes) <= 15:
        st.error("Please enter between 10 and 15 attribute statements, one per line.")
    elif len({attribute.casefold() for attribute in attributes}) != len(attributes):
        st.error("Each attribute statement must be unique.")
    elif selected_demographics.empty:
        st.error("Select at least one digital twin.")
    else:
        with st.spinner("Digital twins are rating the service..."):
            st.session_state["ipa_results"] = asyncio.run(
                run_survey(selected_demographics, service_name, description, attributes, api_key, MODEL_NAME)
            )
            st.session_state["ipa_attributes"] = attributes
            st.session_state["ipa_service"] = service_name

if "ipa_results" in st.session_state:
    results = st.session_state["ipa_results"]
    attributes = st.session_state["ipa_attributes"]
    service_name = st.session_state["ipa_service"]
    successful = results.loc[results["Status"] == "Success"].copy()
    st.subheader("Results")
    if successful.empty:
        st.dataframe(results, use_container_width=True)
        st.error("No valid responses were generated. Review the Status column and model settings.")
    else:
        summary = attribute_summary(successful, attributes)
        figure = plot_ipa(summary, service_name)
        plot_buffer = BytesIO()
        figure.savefig(plot_buffer, format="png", dpi=200, bbox_inches="tight")
        plot_tab, summary_tab, responses_tab = st.tabs(["Importance-performance plot", "Attribute key", "All responses"])
        with plot_tab:
            st.pyplot(figure, use_container_width=True)
            st.caption("Point numbers correspond to the Attribute key. Dashed lines show the overall average of each axis.")
        with summary_tab:
            display_summary = summary.copy()
            display_summary.insert(0, "Point", range(1, len(display_summary) + 1))
            st.dataframe(display_summary, use_container_width=True, hide_index=True)
        with responses_tab:
            st.dataframe(results, use_container_width=True, height=500)
        plt.close(figure)
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", service_name).strip("_") or "service"
        left, right = st.columns(2)
        left.download_button(
            "Download all responses (Excel)", excel_workbook(results, summary),
            f"{safe_name}_importance_performance.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        right.download_button(
            "Download plot (PNG)", plot_buffer.getvalue(),
            f"{safe_name}_importance_performance.png", "image/png",
        )
