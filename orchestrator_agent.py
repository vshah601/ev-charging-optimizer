from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from typing import Dict, Any, Optional
import json

class OrchestratorAgent:

    def __init__(self, blackboard, forecast_agent, schedule_agent, advisor_agent):   
        self.bb = blackboard
        self.forecast_agent = forecast_agent
        self.schedule_agent = schedule_agent
        self.advisor_agent = advisor_agent
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.tools = self.create_tools()
        self.agent_executor = self.create_agent()

    def create_tools(self):

        @tool
        def train_models():
            """
            Create training and testing data to train multiple forecasting models such as linear regression,
            random forest, and XGB regressor
            """
            self.forecast_agent.train_models()
            return "Forecast models trained successfully"

        @tool
        def evaluate_models():
            """
            Evaluate trained models and select the best one based on MAPE value. Return best model and its performance metric
            """
            self.forecast_agent.evaluate_models()
            best_model = self.bb.read("best_model")
            model_performance = self.bb.read("model_performance")

            result = f"Best model: {best_model} (MAPE: {model_performance[best_model]:.2f}%)\n"          
            result += "All models: " + ", ".join([f"{k}: {v:.2f}%" for k, v in model_performance.items()])                                    
            return result

        @tool
        def forecast_next_24h():
            """
            Generate next day forecasted hourly load for the next 24 hours with the best model. Returns
            dataframe with next day datetimes and forecasted hourly loads
            """
            self.forecast_agent.forecast_next_24h()
                 
            return "Next day forecast successfully generated"   

        @tool
        def generate_schedule():
            """
            Analyzes forecasted next day hourly loads and finds the optimal 4 hour charging window based on lowest
            loads. Returns the time window and average load in that window
            """
            self.schedule_agent.generate_schedule()
            start_time = self.bb.read("best_window").iloc[0]["datetime"].hour
            end_time = self.bb.read("best_window").iloc[-1]["datetime"].hour
            avg_load = self.bb.read("best_window")["forecast_mw"].mean()

            result = f"Optimal charging window is from {start_time}:00 to {end_time}:00. "
            result += f"Average load in this period is {avg_load:.2f} MW\n"
            return result

        @tool
        def generate_report():
            """
            Generate full comprehensive advisory report with optimal window, average load, and model details.
            Returns full report
            """
            return self.advisor_agent.generate_report()

        @tool
        def check_system_status():
            """
            Check what data is currently available in the blackboard. Returns status of key elements
            """
            keys = ["best_model", "model_performance", "forecast", "forecast_times", "best_window"]
            status = "System Status:\n"
            for k in keys:
                status += f" - {k}: {'✓ Available' if self.bb.read(k) is not None else '✗ Not available'}\n"
            return status

        return [
            train_models,
            evaluate_models,
            forecast_next_24h,
            generate_schedule,
            generate_report,
            check_system_status
        ]

    def create_agent(self):

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent orchestrator for a load forecasting and cost optimization system.

            Your role is to:
            1. Understand user requests about load forecasting and charging optimization
            2. Execute tasks in the correct logical order
            3. Coordinate between specialist agents via their tools
            4. Provide clear, actionable responses
            
            IMPORTANT WORKFLOW ORDER:
            - Step 1: Train models (train_forecast_models)
            - Step 2: Evaluate models
            - Step 3: Generate next day forecast dataframe
            - Step 4: Find optimal charging window and average load in the window
            - Step 5: Generate comprehensive report
            
            Always follow this sequence. Check blackboard state if unsure what's available.
            
            When a user asks for a complete analysis, execute ALL steps in order. Do not offer further assistance.
            Be concise but informative in your responses."""),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            return_intermediate_steps=True
        )

    def run(self, query: str) -> Dict[str, Any]:

        return self.agent_executor.invoke({"input": query})

    def run_full_pipeline(self) -> str:

        query = "Run the complete analysis: train models, generate tomorrow's forecast, find the optimal charging window, and provide           recommendations."
        result = self.run(query)
        return result