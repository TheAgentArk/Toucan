## Task
Generate **augmented variations** of a given question that maintain the same target tool(s) usage and context but significantly increase the complexity and constraints required to solve the problem.

## Objective
Take an existing question and its associated target tool(s), then create multiple sophisticated variations that:
- Use the same target tool(s) to achieve the core goal while navigating additional complexity layers
- Maintain the same general context and domain as the original question
- Increase multi-dimensional complexity through realistic constraints, competing requirements, stakeholder considerations, and interconnected dependencies
- Embed the tool usage within larger, more complex workflows that require strategic thinking and coordination
- Demonstrate how the same core tool usage applies under vastly different complexity levels

## Guidelines
- Introduce realistic constraints such as resource limits, compliance requirements, tight timelines, or stakeholder conflicts
- Embed the same tool usage inside a broader workflow that requires coordination across teams or systems
- Escalate demands (performance, scalability, risk management) without changing the original domain or context
- Ensure each variation targets a different primary complexity angle (organizational, technical, strategic) while preserving tool relevance
- Ensure the question does not contain any tool names or explicit references to the target tools.

## Input Format
**Original Question**: {ORIGINAL_QUESTION}
**Target Tools**: {TARGET_TOOLS}
**Tool Descriptions**: {TOOL_DESCRIPTIONS}

## Output Requirements
Generate **{VARIATIONS_COUNT} strategically augmented variations** of the original question. Each variation should:
1. Maintain the same core goal that requires the target tool(s) while adding multiple complexity layers
2. Keep the same general context and domain as the original question
3. Introduce different but interconnected constraints and competing requirements
4. Feel like natural, high-stakes, real-world scenarios that professionals encounter
5. Be meaningfully different from the original and other variations in terms of complexity
6. Include specific details that make the constraints and requirements concrete and actionable
7. **Transform step-wise questions**: If the original question contains explicit steps, convert it to a goal-oriented format while maintaining the same tool usage requirements
8. Avoid including any explicit mentions, hints, or references to the target tool names within the question text

## Output
Provide your response in the following XML format:

<response>
  <analysis>
    <!-- Analyze the original question and target tool(s) to understand the core goal, current complexity level, and identify multiple complexity dimensions that can be naturally introduced while maintaining tool relevance and solution feasibility -->
  </analysis>
  <variations>
    <!-- Generate {VARIATIONS_COUNT} variations, each with <variation_X>, <constraints>, and <question> tags -->
    <variation_1>
      <constraints>
        <!-- Specific organizational, stakeholder, or coordination constraints that add realistic complexity -->
      </constraints>
      <question>
        <!-- The complex, organizationally-focused question that maintains the same target tool(s) usage within a more sophisticated workflow -->
      </question>
    </variation_1>
    <!-- Continue with variation_2, variation_3, etc. as needed based on number of variations -->
  </variations>
</response>
