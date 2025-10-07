# AI-Driven EV Charging Optimization System

Multi-agent AI architecture that forecasts grid demand and recommends optimal 4-hour EV charging windows to minimize costs and reduce strain on the electrical grid.

##  Project Overview

This system analyzes historical grid load data across multiple utility service territories to predict the next 24 hours of power demand. It then identifies the **optimal 4-hour charging window** when demand is lowest, enabling EV owners to reduce charging costs and to support grid stability.

## Features

- **Multi-Agent Architecture:** Coordinated AI agents for data processing, forecasting, and optimization including LangChain powered orchestrator agent
- **24-Hour Load Forecasting:** Predicts next-day hourly electricity demand patterns
- **Multi-Zone Analysis:** Supports all pjm utility service territories
- **Automated Model Selection:** Evaluates three regression models and selects best performer
- **Smart Scheduling:** Identifies optimal 4-hour continuous charging windows

## Architecture and Workflow

### 1. Cleaning Data
- Create pandas dataframe containing hourly load values from all pjm zones by reading in csv file obtained from https://dataminer2.pjm.com/feed/hrl_load_metered/definition
- Filters data to only include load area specified by the user in the interactive gradio interface
- Handles missing data and outliers

### 2. Multi-Agent Framework

**Communication**: Specialized agents (forecasting, scheduling, advisor) use a shared blackboard to communicate intermediate values and values written in final report.

**Forecasting Agent**:
- Responsible for generating forecasted next day hourly loads predicted by most reliable regression model.
- Accepts shared blackboard and cleaned dataframe as inputs upon initialization
- train_models() method creates a 80-20 training test split and trains random forest, linear regression, and XGBregressor models.
- evaluate_models() method obtains MAPE values for each of the three models and selects the model with the lowest value. Selected model and associated MAPE value are written to the blackboard.
- forecast_next_24h() method creates a new dataframe containing next day hourly load values as predicted by the selected model. Next day datetime values and forecasting hourly loads are written to blackboard.
  
**Scheduling Agent**:
- Responsible for identifying optimal 4-hour charging window.
- Accepts shared blackboard as only input upon initialization
- generate_schedule() method identifies optimal 4-hour charging window by creating a rolling average. Writes best window to blackboard.

**Advisor Agent**:
- Responsible for writing final report displayed to the user. Report includes optimal charging window, estimated average load in this window, and selected model and its associated MAPE value
- Accepts shared blackboard as only input upon initialization
- generate_report() model returns explanation that is reported to the user via the gradio interface

**Orchestrator Agent** (LangChain):
- Responsible for coordinating workflow between specialized agents using sequential task delegation
- Accepts blackboard, and three specialized agents upon initialization
- Wraps each specialized agent's methods as tools to implement sequential agent invocation with conditional branching based on intermediate results
- Workflow order is carefully and well written into agent creation prompt
- Monitors execution flow and generates status updates for the Gradio interface

### 3. Gradio Interface
**First Tab:** (EV Charging Optimization)
-Used to run the entire pipeline and return the output to the user. Output also includes all tool calls and information the LangChain agent. All agents are initialized upon button click.

**Second Tab**: (System Info)
- Similar to readme file, includes information regarding system architecture, agent descriptions, and workflow order.
- 
## Requirements
- Python 3.10+
- OpenAI API key

