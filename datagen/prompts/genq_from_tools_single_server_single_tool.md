## Task
Generate a **Tool Use Question** based on the provided MCP Server and its tool descriptions.

## Objective
Analyze the provided MCP Server and its available tools, then create a realistic user question that would naturally require the use of one of these tools to solve.

## Guidelines

### Question Realism
- Create questions that represent real-world scenarios where users would need to interact with the MCP Server's tools
- The question should sound natural and authentic, as if asked by someone genuinely needing to accomplish a task
- Consider common use cases, problems, or workflows that would require the functionality provided by the MCP Server's tools

### Tool Selection
- Focus on **ONE specific tool** from the MCP Server that would be most appropriate to answer the question
- Choose tools based on the core functionality they provide and how they would solve real user problems
- Consider each tool's description and purpose when crafting the question

### Question Complexity
- Create questions that are clear and specific enough to warrant tool usage
- Avoid overly simple questions that could be answered without tools
- Include relevant context or constraints that make the tool usage necessary
- Do not contain the exact tool name in the question

### Output Format
Your response should include:
1. **Tool Analysis**: Briefly analyze the MCP Server's available tools and their main functionalities.
2. **Target Tool**: The specific tool name from the MCP Server that should be used to answer this question.
3. **Question**: A clear, realistic user question that requires tool usage.

## MCP Server Description
{MCP_SERVER_NAME}: {MCP_SERVER_DESCRIPTION}

Available Tools:
{TOOL_LIST}

## Output
Provide your response in the following XML format:

<response>
  <server_analysis>
    <!-- Briefly analyze the MCP Server's available tools and their main functionalities. -->
  </server_analysis>
  <target_tool>
    <!-- The specific tool name from the MCP Server that should be used to answer this question. -->
  </target_tool>
  <question>
    <!-- A clear, realistic user question that requires tool usage. -->
  </question>
</response>