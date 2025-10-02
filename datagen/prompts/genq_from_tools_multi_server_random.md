## Task
Generate a **Multi-Server Tool Use Question** based on the provided MCP Servers and their tool descriptions.

## Objective
Analyze the provided MCP Servers and their available tools, then create a realistic user question that would naturally require the use of **{NUM_TOOLS} tools from at least 2 different MCP servers** to solve completely.

## Guidelines

### Question Realism
- Create questions that represent real-world scenarios where users would need to interact with tools from multiple MCP Servers
- The question should sound natural and authentic, as if asked by someone genuinely needing to accomplish a complex task
- Consider workflows that span across different services/domains that would require multiple servers
- Think about how different MCP servers complement each other in real-world use cases

### Server and Tool Selection
- Use tools from **at least 2 different MCP servers** to answer the question
- Select **{NUM_TOOLS} tools total** that work together across multiple servers
- The question should require a sequence or combination of tool calls from different servers to solve completely
- Choose tools based on how they complement each other across different services/domains
- Consider each tool's description and purpose when crafting the cross-server workflow
- Ensure tools from different servers create a logical, interconnected workflow

### Question Complexity
- Create questions that are complex enough to warrant using {NUM_TOOLS} tools across multiple servers
- The question should have multiple components or require several steps that span different services
- Include relevant context or constraints that make the multi-server tool usage necessary
- Do not contain the exact tool names or server names in the question
- Ensure the question cannot be reasonably answered with tools from just a single server
- Create scenarios that naturally require different types of services working together

### Cross-Server Integration
- Think about how different servers' capabilities can be combined
- Consider data flow between different services (e.g., retrieving data from one service to use in another)
- Create realistic scenarios where multiple services need to work together
- Focus on complementary functionalities across different domains

### Output Format
Your response should include:
1. **Server Analysis**: Briefly analyze all MCP Servers and their available tools, focusing on how they can work together.
2. **Cross-Server Workflow**: Describe the workflow showing how tools from different servers will be used together.
3. **Target Tools**: The specific tool names from different MCP Servers that should be used together, in the order they would likely be called, with their server names.
4. **Question**: A clear, realistic user question that requires multi-server tool usage.

## Available MCP Servers

{SERVER_DESCRIPTIONS}

## Output
Ensure your question requires exactly {NUM_TOOLS} tools from at least 2 different servers to solve completely. Provide your response in the following XML format:

<response>
  <server_analysis>
    <!-- Briefly analyze all MCP Servers and their available tools, focusing on how they can work together across different domains/services. -->
  </server_analysis>
  <cross_server_workflow>
    <!-- Describe the workflow showing how tools from different servers will be used together to solve the question. -->
  </cross_server_workflow>
  <target_tools>
    <!-- The specific tool names from different MCP Servers that should be used together, listed in order with their server names. e.g., <tool server="Server1">search_posts</tool> <tool server="Server2">send_email</tool> -->
  </target_tools>
  <question>
    <!-- A clear, realistic user question that requires multi-server tool usage spanning different services/domains. -->
  </question>
</response> 