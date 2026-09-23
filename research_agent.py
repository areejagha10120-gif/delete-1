import streamlit as st

# ---------------------------------------------------------
# GROQ / CREWAI COMPATIBILITY FIX
# Removes cache_breakpoint before messages reach Groq.
# ---------------------------------------------------------

import litellm

_original_completion = litellm.completion


def _completion_without_cache_breakpoint(*args, **kwargs):
    # Disable LiteLLM caching
    kwargs["caching"] = False

    # Remove cache_breakpoint from normal messages
    messages = kwargs.get("messages", [])

    for message in messages:
        if isinstance(message, dict):
            message.pop("cache_breakpoint", None)

            # Also check nested content blocks
            content = message.get("content")

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block.pop("cache_breakpoint", None)

    return _original_completion(*args, **kwargs)


litellm.completion = _completion_without_cache_breakpoint


# ---------------------------------------------------------
# CREWAI IMPORTS
# ---------------------------------------------------------

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from ddgs import DDGS


# ---------------------------------------------------------
# DUCKDUCKGO SEARCH TOOL
# ---------------------------------------------------------

class DuckDuckGoSearchTool(BaseTool):

    name: str = "DuckDuckGo Web Search"

    description: str = (
        "Search the internet using DuckDuckGo to find "
        "relevant and current information about a research topic."
    )

    def _run(self, query: str) -> str:

        try:
            results = DDGS().text(
                query,
                max_results=5
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


# ---------------------------------------------------------
# CREATE GROQ LLM
# ---------------------------------------------------------

def create_llm():

    api_key = st.secrets.get(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is not configured in Streamlit Secrets."
        )

    return LLM(

        model="groq/openai/gpt-oss-120b",

        api_key=api_key,

        temperature=0.2,

        max_tokens=8000
    )


# ---------------------------------------------------------
# RUN RESEARCH
# ---------------------------------------------------------

def run_research(topic: str):

    llm = create_llm()

    search_tool = DuckDuckGoSearchTool()

    # -----------------------------------------------------
    # RESEARCH AGENT
    # -----------------------------------------------------

    researcher = Agent(

        role="Senior Research Analyst",

        goal=(
            "Research the user's topic using web search "
            "and produce an accurate, well-structured "
            "research report."
        ),

        backstory=(
            "You are an experienced research analyst. "
            "You investigate topics using multiple sources, "
            "compare information, identify important evidence, "
            "and write clear research reports. "
            "You never invent facts or sources."
        ),

        llm=llm,

        tools=[
            search_tool
        ],

        verbose=False,

        allow_delegation=False
    )

    # -----------------------------------------------------
    # RESEARCH TASK
    # -----------------------------------------------------

    research_task = Task(

        description=f"""
Research the following topic:

{topic}

Use the DuckDuckGo web search tool to find
relevant and reliable information.

Research requirements:

1. Search for multiple relevant sources.
2. Prefer recent and reliable information.
3. Compare information from different sources.
4. Do not invent facts.
5. Do not invent URLs or sources.
6. Clearly distinguish facts from opinions.
7. Include useful examples where appropriate.
8. Explain important limitations or uncertainties.

Create a detailed research report using this structure:

# Title

## Executive Summary

Give a concise overview of the research.

## Introduction

Explain the topic and why it matters.

## Key Findings

List the most important findings.

## Detailed Analysis

Explain the topic in depth using information
found through web research.

## Evidence and Examples

Provide relevant evidence, statistics,
examples, or real-world cases when available.

## Challenges and Limitations

Explain limitations, conflicting information,
or areas where evidence is uncertain.

## Conclusion

Summarize the main findings.

## Sources

Provide a numbered list of the sources used.

For every source include:

- Source title
- URL

Only include URLs that were actually returned
by the web search tool.
""",

        expected_output=(
            "A detailed Markdown research report containing "
            "an executive summary, introduction, key findings, "
            "detailed analysis, evidence and examples, "
            "challenges and limitations, conclusion, "
            "and numbered source URLs."
        ),

        agent=researcher
    )

    # -----------------------------------------------------
    # CREW
    # -----------------------------------------------------

    crew = Crew(

        agents=[
            researcher
        ],

        tasks=[
            research_task
        ],

        verbose=False
    )

    # -----------------------------------------------------
    # START RESEARCH
    # -----------------------------------------------------

    result = crew.kickoff()

    return str(result)
