from agno.playground import Playground, serve_playground_app
from teams.discussion_team import discussion_team
from teams.news_agency_team import news_agency_team
from teams.multi_language_team import multi_language_team
from agents.rag_agent import internal_document_agent
from agents.calendar_agent import calendar_agent        
from agents.web_search_agent import web_search_agent
from workflows.employee_recruiter_workflow import employee_recruiter_workflow


app = Playground(
    agents=[internal_document_agent, calendar_agent, web_search_agent],
    teams=[discussion_team, multi_language_team],
    workflows=[employee_recruiter_workflow],
).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
