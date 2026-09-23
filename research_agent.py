 import streamlit as st

# =========================================================
# GROQ / CREWAI COMPATIBILITY FIX
# Removes cache_breakpoint before messages reach Groq.
# =========================================================

import litellm

_original_completion = litellm.completion


def _completion_without_cache_breakpoint(*args, **kwargs):
    kwargs["caching"] = False

    messages = kwargs.get("messages", [])

    for message in messages:
        if isinstance(message, dict):
            message.pop("cache_breakpoint", None)

            content = message.get("content")

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block.pop("cache_breakpoint", None)

    return _original_completion(*args, **kwargs)


litellm.completion = _completion_without_cache_breakpoint


# =========================================================
# IMPORTS
# =========================================================

from typing import Type

from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool

from ddgs import DDGS


# =========================================================
# DUCKDUCKGO TOOL INPUT SCHEMA
# =========================================================

class DuckDuckGoSearchInput(BaseModel):

    query: str = Field(
        ...,
        description="The exact web search query to search for."
    )


# =========================================================
# DUCKDUCKGO SEARCH TOOL
# =========================================================

class DuckDuckGoSearchTool(BaseTool):

    name: str = "duck_duck_go_web_search"

    description: str = (
        "Search the web using DuckDuckGo. "
        "Use this tool whenever you need current or factual "
        "information from the internet. "
        "The input must contain exactly one parameter named "
        "'query'. Do not use 'search'."
    )

    args_schema: Type[BaseModel] = DuckDuckGoSearchInput

    def _run(self, query: str) -> str:

        try:

            results = DDGS().text(
                query=query,
                max_results=3
            )

            if not results:
                return "No search results were found."

            formatted_results = []

            for index, result in enumerate(results, start=1):

                title = result.get(
                    "title",
                    "No title"
                )

                url = result.get(
                    "href",
                    ""
                )

                body = result.get(
                    "body",
                    ""
                )

                formatted_results.append(
                    f"""
SOURCE {index}

Title:
{title}

URL:
{url}

Description:
{body}
"""
                )

            return "\n".join(formatted_results)

        except Exception as e:

            return f"Search failed: {str(e)}"


# =========================================================
# CREATE GROQ LLM
# =========================================================

def create_llm():

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is not configured in Streamlit Secrets."
        )

    return LLM(

        model="groq/openai/gpt-oss-120b",

        api_key=api_key,

        temperature=0.2,

        max_tokens=400
    )


# =========================================================
# RUN RESEARCH
# =========================================================

def run_research(topic: str):

    llm = create_llm()

    search_tool = DuckDuckGoSearchTool()

    # =====================================================
    # AGENT
    # =====================================================

    researcher = Agent(

        role="Senior Research Analyst",

        goal=(
            "Research the user's topic using web search "
            "and produce an accurate, concise research report."
        ),

        backstory=(
            "You are an experienced research analyst. "
            "You research topics using multiple web sources, "
            "compare information, identify important evidence, "
            "and write clear factual reports. "
            "Never invent facts or sources."
        ),

        llm=llm,

        tools=[
            search_tool
        ],

        verbose=False,

        allow_delegation=False
    )

    # =====================================================
    # TASK
    # =====================================================

    research_task = Task(

        description=f"""
Research this topic:

{topic}

Follow these steps:

1. Use the DuckDuckGo web search tool.
2. The search tool requires a parameter named "query".
3. Search for the most relevant information.
4. Use multiple web sources when appropriate.
5. Do not invent facts.
6. Do not invent URLs.
7. Keep the final report concise.

Create a Markdown report with:

# Title

## Executive Summary

## Key Findings

## Detailed Analysis

## Evidence and Examples

## Limitations

## Conclusion

## Sources

For Sources, include the title and URL of
sources actually returned by the search tool.

Keep the final report under approximately
1500 words.
""",

        expected_output=(
            "A concise Markdown research report with "
            "an executive summary, key findings, detailed "
            "analysis, evidence, limitations, conclusion, "
            "and numbered source URLs."
        ),

        agent=researcher
    )

    # =====================================================
    # CREW
    # =====================================================

    crew = Crew(

        agents=[
            researcher
        ],

        tasks=[
            research_task
        ],

        verbose=False
    )

    # =====================================================
    # RUN
    # =====================================================

    result = crew.kickoff()

    return str(result)
