import uuid
from typing import List
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv("../.env")

@tool
def get_all_valid_actions() -> List[str]:
    """Returns the list of valid actions that the deep learning model has the capability to generate."""
    valid_actions = ["walk", "run", "jump", "wave", "sit", "stand", "punch"]
    print(f"\n[Tool Executing] get_all_valid_actions -> {valid_actions}")
    return valid_actions

@tool
def generate_motion(action: str) -> str:
    """
    Generates motion for a given action using the DL model.
    Pass a valid action string. Returns a unique motion ID.
    """
    print(f"\n[Tool Executing] generate_motion(action='{action}')")
    # Simulating a deep learning model call returning a unique ID
    motion_id = f"motion_{uuid.uuid4().hex[:6]}"
    print(f"   -> Generated Motion ID: {motion_id}")
    return motion_id

@tool
def warp_motion_with_character(character_img_file: str, motion_id: str) -> str:
    """
    Warps the generated motion to a specific character image.
    Pass the character's image filename and the motion ID. Returns a unique warped motion ID.
    """
    print(f"\n[Tool Executing] warp_motion_with_character(file='{character_img_file}', motion='{motion_id}')")
    warped_id = f"warped_{uuid.uuid4().hex[:6]}"
    print(f"   -> Generated Warped Motion ID: {warped_id}")
    return warped_id

@tool
def stitch_motions_in_sequence(warped_motion_ids: List[str]) -> str:
    """
    Stitches all generated warped motions into a final sequence video.
    Pass a chronologically ordered list of warped motion IDs. Returns the final video ID.
    """
    print(f"\n[Tool Executing] stitch_motions_in_sequence(ids={warped_motion_ids})")
    final_video_id = f"final_vid_{uuid.uuid4().hex[:8]}"
    print(f"   -> Generated Final Sequence ID: {final_video_id}")
    return final_video_id

tools = [
    get_all_valid_actions, 
    generate_motion, 
    warp_motion_with_character, 
    stitch_motions_in_sequence
]


SYSTEM_PROMPT = """You are an autonomous AI Animation Director.
Your objective is to take a storyline and autonomously orchestrate its conversion into an animated sequence using your tools.

You have two strict pre-processing duties:
1. ACTION PLANNER: Break the story down into a sequence of actions. You MUST call `get_all_valid_actions()` first to ensure you only plan actions supported by the model. 
2. CHARACTER CHOOSER: Identify the characters in the story. Assign a descriptive image filename to each character (e.g., 'knight_char.png', 'dragon_char.png').

EXECUTION WORKFLOW (Execute autonomously):
1. Determine valid actions (`get_all_valid_actions`).
2. Map out the sequence of events and characters.
3. For each chronological event, call `generate_motion(action)`.
4. Take the returned motion ID and immediately warp it to the correct character's file using `warp_motion_with_character`.
5. Keep track of all warped motion IDs.
6. Finally, once all motions are warped, call `stitch_motions_in_sequence` with the list of warped IDs in chronological order.
"""

def agent_node(state: MessagesState):
    """The main LLM node that makes autonomous decisions."""
    
    # Initialize LLM and bind structured tools
    # Using GPT-4o here for complex multi-tool orchestration, but you can swap this.
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    # Inject System Prompt if it doesn't exist in the state yet
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
    # Invoke the model
    response = llm_with_tools.invoke(messages)
    
    # DEBUG: Print exactly what the LLM is responding back during invocation
    print("\n" + "="*40)
    print("LLM RESPONDED:")
    if response.content:
        print(f"Text Content:\n{response.content}")
    if response.tool_calls:
        print("Tool Calls Requested:")
        for tc in response.tool_calls:
            print(f" - {tc['name']} with args: {tc['args']}")
    print("="*40 + "\n")
        
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")

# Agent -> Tools (if tools were called) OR Agent -> END (if finished)
# The `tools_condition` automatically checks if `response.tool_calls` exists.
builder.add_conditional_edges("agent", tools_condition)

# Tools -> Agent (loop back to the LLM so it can assess tool outputs and decide the next step)
builder.add_edge("tools", "agent")

graph = builder.compile()
png_data = graph.get_graph().draw_png()
with open("../iofiles/langgraph_workflow.png", "wb") as f:
    f.write(png_data)


if __name__ == "__main__":
    storyline = """
    A brave hero swims.
    """
    
    print(f"Starting Graph Execution for Story: {storyline.strip()}")
    for event in graph.stream({"messages": [HumanMessage(content=storyline)]}):
        pass