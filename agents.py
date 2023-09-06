from typing import Any, List, Sequence

from langchain.agents.agent import Agent, AgentOutputParser
from langchain.agents.agent_types import AgentType
from langchain.agents.react.output_parser import ReActOutputParser
from langchain.agents.react.wiki_prompt import EXAMPLES, SUFFIX
from langchain.agents.tools import Tool
from langchain.agents.utils import validate_tools_single_input
from langchain.pydantic_v1 import Field
from langchain.schema import BasePromptTemplate
from langchain.tools.base import BaseTool
from langchain.prompts.prompt import PromptTemplate


PREFIX = """Learn from the given examples, answer the following questions as best you can with same format and always provide an Action as end of your answer. You have access to the following action: \n"""

class ReActDocstoreAgent(Agent):
    """Agent for the ReAct chain."""

    output_parser: AgentOutputParser = Field(default_factory=ReActOutputParser)

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return ReActOutputParser()

    @property
    def _agent_type(self) -> str:
        """Return Identifier of an agent type."""
        return AgentType.REACT_DOCSTORE

    @classmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """Return default prompt."""
        tools.append(
            Tool(
                name="Finish",
                func=lambda x: x,
                description="use this when you think the question is answered",
            )
        )
        tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        print(tool_strings)
        WIKI_PROMPT = PromptTemplate.from_examples(
            EXAMPLES, SUFFIX, ["input", "agent_scratchpad"], prefix=PREFIX + tool_strings
        )
        return WIKI_PROMPT

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        validate_tools_single_input(cls.__name__, tools)
        super()._validate_tools(tools)
        if len(tools) != 2:
            raise ValueError(f"Exactly two tools must be specified, but got {tools}")
        tool_names = {tool.name for tool in tools}
        if tool_names != {"Lookup", "Search"}:
            raise ValueError(
                f"Tool names should be Lookup and Search, got {tool_names}"
            )

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def _stop(self) -> List[str]:
        return ["\nObservation:"]

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the LLM call with."""
        return "Thought:"