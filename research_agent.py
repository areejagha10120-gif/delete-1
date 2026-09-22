import os

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from ddgs import DDGS


# --------------------------------------------------
# DuckDuckGo Search Tool
# --------------------------------------------------

class DuckDuckGoSearchTool(BaseTool):

    name: str = "DuckDuckGo Web Search"

    description: str = (
        "Search the internet using DuckDuckGo. "
        "Use this tool to find current and relevant information "
        "about the user's research topic."
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

                title = result.get("title", "No title")
                url = result.get("href", "")
                body = result.get("body", "")

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


# --------------------------------------------------
# Create LLM
# --------------------------------------------------

def create_llm():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Add your Groq API key to Streamlit Secrets."
        )

    return LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.2,
        max_tokens=8000
    )


# --------------------------------------------------
# Research function
# --------------------------------------------------

def run_research(topic: str) -> str:

    llm = create_llm()

    search_tool = DuckDuckGoSearchTool()

    # --------------------------------------------------
    # Single Research Agent
    # --------------------------------------------------

    researcher = Agent(
        role="Senior Research Analyst",

        goal=(
            "Research the user's topic using reliable web sources "
            "and produce an accurate, well-structured research report."
        ),

        backstory=(
            "You are an experienced research analyst. "
            "You investigate topics carefully, compare information "
            "from multiple sources, identify important facts, "
            "and write clear research reports. "
            "You never invent sources or facts."
        ),

        llm=llm,

        tools=[search_tool],

        verbose=False,

        allow_delegation=False
    )

    # --------------------------------------------------
    # Research Task
    # --------------------------------------------------

    research_task = Task(

        description=f"""
Research the following topic:

{topic}

Your job is to investigate this topic using the web search tool.

Research requirements:

1. Search for relevant information using multiple search queries
   when necessary.

2. Prefer reliable and authoritative sources.

3. Do not rely on only one website.

4. Cross-check important factual claims when possible.

5. Do not invent facts, statistics, studies, quotations,
   organizations, or sources.

6. Clearly distinguish established facts from interpretations
   or opinions.

7. Include source URLs for the important information used.

After completing the research, write a professional report.

The report should contain:

# Title

## Executive Summary

Give a concise overview of the main findings.

## Introduction

Explain the topic and why it matters.

## Key Findings

Present the most important findings from the research.

## Detailed Analysis

Explain the topic in more depth.

## Evidence and Examples

Include relevant examples, statistics, studies, or documented
evidence when available.

## Challenges and Limitations

Discuss important limitations, disagreements, uncertainties,
or research gaps.

## Conclusion

Summarize the main findings without introducing new claims.

## Sources

Provide a numbered list of the URLs used.

Important:

- Do not fabricate citations.
- Only include URLs actually returned by the search tool.
- Do not claim that a source says something if the search result
  does not support it.
- Keep the report factual and readable.
""",

        expected_output=(
            "A detailed Markdown research report with an executive "
            "summary, introduction, key findings, detailed analysis, "
            "evidence, limitations, conclusion, and numbered source URLs."
        ),

        agent=researcher
    )

    # --------------------------------------------------
    # Create Crew
    # --------------------------------------------------

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=False
    )

    # --------------------------------------------------
    # Run Crew
    # --------------------------------------------------

    result = crew.kickoff()

    return str(result)
