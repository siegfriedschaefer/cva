
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew, before_kickoff, after_kickoff
from crewai.agents.agent_builder.base_agent import BaseAgent


from crewai_tools import PDFSearchTool
from typing import Any, List, Dict, Optional

docs_scrape_tool = PDFSearchTool(pdf='./assets/900320_001.pdf')

@CrewBase
class SupportCrew:
    """Support crew.

    - Implements a support agent crew.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    tasks_config = 'config/tasks.yaml'
    agents_config = 'config/agents.yaml'

    def __init__(
        self,
        name: str = "Alex",
        input: Dict[str, str] = None,
    ) -> None:
        self.name = name
        self.input = {

            "customer": "Schaefer.AI",
            "person": "Siegfried Schaefer",
            "inquiry": "I need help with adjusting the hysterisis "
                        "of the temperature alarm in a ST121 controller "
                        "Can you provide guidance?"
        }

    @before_kickoff
    def prepare_inputs(self, inputs):
        # Modify inputs before the crew starts
        inputs['additional_data'] = "Some extra information"
        return inputs

    @after_kickoff
    def process_output(self, output):
        # Modify output after the crew finishes
        output.raw += "\nProcessed after kickoff."
        return output
    
    @agent
    def support_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['support_agent'], # type: ignore[index]
            verbose=True,
            tools=[docs_scrape_tool],
        )
    
    @agent
    def support_quality_assurance_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['support_quality_assurance_agent'], # type: ignore[index]
            verbose=True
#            tools=[DevTool()]
        )

    @task
    def inquiry_resolution_task(self) -> Task:
        return Task(
            config=self.tasks_config['inquiry_resolution_task'] # type: ignore[index]
        )

    @task
    def quality_assurance_review_task(self) -> Task:
        return Task(
            config=self.tasks_config['quality_assurance_review_task'] # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,  # Automatically collected by the @agent decorator
            tasks=self.tasks,    # Automatically collected by the @task decorator.
            process=Process.sequential,
            verbose=True,
        )    
    
    def set_input(self, params: Optional[Dict[str, str]] = None) -> None:
        return
    
    def kickoff(self) -> None:

        result = self.crew.kickoff(inputs=self.inputs)
        # print(result)


__all__ = ["SupportAgent"]