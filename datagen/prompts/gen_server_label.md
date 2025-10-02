## Task
Generate **Server Labels** to categorize the provided MCP Server based on its description and available tools.

## Objective
Analyze the provided MCP Server's description and available tools, then assign appropriate category labels that best describe its primary functionality and use cases.

## Guidelines

### Label Selection
- Analyze the MCP Server's core functionality and purpose
- Consider the types of tools it provides and the problems it solves
- Select labels that accurately represent the server's primary use cases
- Choose from predefined categories when applicable, but also consider custom labels for unique functionality

### Predefined Categories
Choose from these established categories when appropriate:
- **Web Search & Research**: Tools for searching the web, gathering information, academic research
- **Browser Automation**: Web scraping, automated browsing, page interaction
- **Memory Management**: Data storage, retrieval, knowledge bases, note-taking
- **Operating System**: File operations, system commands, process management
- **Data Analysis & Processing**: Analytics, data transformation, statistical analysis
- **Cryptocurrency & Blockchain**: Trading, wallet management, DeFi, blockchain interaction
- **Daily Productivity**: Task management, scheduling, personal organization
- **File Management**: File operations, document handling, storage management
- **Database Operations**: Data querying, database management, SQL operations
- **API Integration**: Third-party service integration, webhook handling
- **Communication Tools**: Messaging, email, notifications, social interaction
- **Development Tools**: Code analysis, debugging, version control, CI/CD
- **Security & Authentication**: Password management, encryption, access control
- **Cloud Services**: Cloud platform integration, serverless functions
- **AI/ML Tools**: Machine learning, model interaction, AI-powered features
- **Content Creation**: Writing, editing, media generation, publishing
- **Social Media**: Social platform integration, posting, analytics
- **Financial Services**: Banking, payments, financial data, accounting
- **E-commerce**: Shopping, product management, order processing
- **Gaming**: Game-related tools, entertainment, interactive features
- **Education**: Learning tools, course management, educational content
- **Health & Fitness**: Health monitoring, fitness tracking, medical tools
- **Travel & Maps**: Location services, travel planning, navigation
- **News & Media**: News aggregation, media consumption, journalism tools
- **Weather**: Weather data, forecasting, climate information
- **Time & Calendar**: Scheduling, time management, calendar integration

### Custom Labels
- If the server doesn't fit well into predefined categories, create a custom label
- Custom labels should be descriptive and specific to the server's unique functionality
- Use clear, concise terminology that would be useful for clustering and organization

### Output Requirements
- **Primary Label**: The main category that best describes the server (from predefined list or custom)
- **Secondary Labels**: Additional relevant categories (0-2 labels)
- **Custom Label**: A free-form descriptive label if the server has unique functionality not covered by predefined categories

## MCP Server Description
{MCP_SERVER_NAME}: {MCP_SERVER_DESCRIPTION}

Available Tools:
{TOOL_LIST}

## Output
Provide your response in the following XML format:

<response>
  <analysis>
    <!-- Briefly analyze the MCP Server's core functionality and the types of problems it solves based on its description and available tools. -->
  </analysis>
  <reasoning>
    <!-- Brief explanation of why these labels were chosen and how they represent the server's functionality -->
  </reasoning>
  <primary_label>
    <!-- The main category that best describes this server's primary functionality -->
  </primary_label>
  <secondary_labels>
    <!-- Additional relevant categories (0-2 labels), separated by commas if multiple -->
  </secondary_labels>
  <custom_label>
    <!-- A free-form descriptive label if the server has unique functionality not covered by predefined categories. Leave empty if not needed. -->
  </custom_label>
</response>