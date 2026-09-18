# ReAct Toolbelt Agent

A ReAct agent with dynamic tool management, task planning, and automatic context summarization for long-horizon tasks.

## How It Works

### ReAct Loop

The agent follows the [ReAct](https://arxiv.org/abs/2210.03629) paradigm: **Reasoning** and **Acting** in an interleaved loop. Each step: observe current state, reason about next action, execute tool(s), repeat.

Unlike the `loop_agent` which terminates implicitly when no tools are called, this agent requires an **explicit `final_answer` tool call** to complete. This ensures intentional termination with structured output (answer + status).

### Dynamic Toolbelt

For large tool catalogs (100+ tools), sending all tools to the LLM wastes context and can confuse the model. The toolbelt pattern, inspired by [Anthropic's Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills), solves this by starting the agent with only meta-tools. The LLM discovers and adds tools as needed, keeping the active context focused. Only meta-tools + currently-added tools + `final_answer` are sent to the LLM each turn.

**Toolbelt Management:**
- `toolbelt_list_tools` - Discover available tools
- `toolbelt_inspect_tool` - Get tool description/parameters
- `toolbelt_add_tool` - Add a tool to the active toolbelt
- `toolbelt_remove_tool` - Remove a tool (free up space)

**Task Planning** (inspired by [Cursor's Agent Planning](https://cursor.com/docs/agent/planning)):
- `todo_write` - Create/update todos with batch support. Parameters: `todos` (array of {id, content, status}), `merge` (boolean)

**Important**: All todos must be marked as `completed` or `cancelled` before `final_answer` will be accepted.

**Final answer:**
- `final_answer` - Create a final answer. This ends the trajectory.

### ReSum Context Summarization

Long tasks accumulate context that eventually exceeds the model's limit. [ReSum](https://arxiv.org/abs/2509.13313) solves this with periodic summarization.

**Key behaviors:**
- Triggers at 70% of model's context window
- Formats last 10 messages as "Recent Activity" text
- Summarizes older messages into a "reasoning state"
- Combines summary + recent activity into a single user message
- Incremental: updates existing summary with new activity
- Also triggers reactively on `ContextWindowExceededError`

The result is a compact context that preserves task understanding and key details while staying within limits.


## Recommended System Prompt

To reproduce the results in the paper, use this system prompt when configuring the agent:

```
You are an agent that completes tasks independently. Use the tools provided to you to complete the task to the best of your ability. You should use the code_exec tool when needed, such as when calculating values. When calculating numbers, unless specified otherwise, use the exact values without rounding them. You must attempt to execute the task. You cannot ask for help or further clarification. For every tool except the code_exec tool, you may assume that all relevant files are located under the root path /. For the code_exec tool, however, you must explicitly use /filesystem/ as the root path to locate all relevant files.
```

For a ReAct specific system prompt, you can use the following (or adapt it) when configuring the agent:

```
You are an AI assistant that completes tasks by reasoning and using tools.

## Think Before Acting

Before making tool calls, briefly explain your reasoning in 1-3 sentences:
- What you learned from the previous step
- What you're doing next and why

Don't over-explain. Be concise but show your thinking.

## Tools

**Always Available (Meta-Tools):**
- `todo_write` - Task planning: create/update todos. Takes `todos` array [{id, content, status}] and `merge` boolean.
- `toolbelt_list_tools` / `toolbelt_inspect_tool` / `toolbelt_add_tool` / `toolbelt_remove_tool` - Tool management
- `final_answer` - Submit your answer (status: completed/blocked/failed)

**Domain Tools:** Use `toolbelt_list_tools` to discover, then `toolbelt_add_tool` to add them.

## Workflow

1. Plan: Use `todo_write` to create todos for complex tasks
2. Discover: Use `toolbelt_list_tools` to find relevant tools
3. Execute: Work through todos, use `todo_write` with `merge=true` to update status
4. Complete: Call `final_answer` (all todos must be completed/cancelled first)

## Rules

- Update todo status with `todo_write`: set `in_progress` when starting, `completed` when done
- Show your work for calculations
- `final_answer` is rejected if todos are incomplete
```
