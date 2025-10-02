## Task
Generate **augmented variations** of a given question that maintain the same target tool(s) usage and complexity level but apply them across different contexts and scenarios.

## Objective
Take an existing question and its associated target tool(s), then create multiple variations that:
- Use the same target tool(s) to achieve the core goal
- Maintain the exact same tool usage order and final outcome
- Apply the question to completely different contexts, scenarios, or domains
- Keep the same level of complexity and constraints as the original
- Demonstrate how the same tool usage pattern applies across diverse real-world scenarios

## Guidelines
- Translate the question to distinctly different domains, user personas, or situational contexts while preserving its original complexity level.
- Keep the tool usage sequence and final outcome identical across all variations.
- Ensure each variation feels like a realistic scenario in its new context and remains solvable with the same tool operations.
- Ensure the question does not contain any tool names or explicit references to the target tools.

## Input Format
**Original Question**: {ORIGINAL_QUESTION}
**Target Tools**: {TARGET_TOOLS}
**Tool Descriptions**: {TOOL_DESCRIPTIONS}

## Output Requirements
Generate **{VARIATIONS_COUNT} augmented variations** of the original question. Each variation should:
1. Maintain the same core goal that requires the target tool(s)
2. Use the exact same tool(s) in the same order with the same final outcome
3. Apply to a completely different context, scenario, or domain
4. Keep the same complexity level and constraints as the original
5. Feel like a natural, real-world scenario from a different setting
6. Be meaningfully different from the original and other variations in terms of context only
7. Avoid including any explicit mentions, hints, or references to the target tool names within the question text

## Output
Provide your response in the following XML format:

<response>
  <analysis>
    <!-- Briefly analyze the original question and target tool(s) to understand the core goal, tool usage pattern, complexity level, and expected outcome, then identify how this can be applied across different domains while maintaining operational consistency -->
  </analysis>
  <variations>
    <!-- Generate {VARIATIONS_COUNT} variations, each with <variation_X>, <context>, and <question> tags -->
    <variation_1>
      <context>
        <!-- Brief description of the new domain/scenario introduced -->
      </context>
      <question>
        <!-- The augmented question that maintains the same target tool(s) usage order, complexity, and outcome but in a different context -->
      </question>
    </variation_1>
    <!-- Continue with variation_2, variation_3, etc. as needed based on number of variations -->
  </variations>
</response>
