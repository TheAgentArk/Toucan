## Task
Conduct a **Response Quality Assessment** of a tool-use conversation across two LLM-scored dimensions, with a third dimension computed automatically outside the LLM.

## Objective
Analyze the provided conversation and assess its response quality across two primary dimensions scored by the LLM, while reserving an additional tool-call accuracy dimension for automated scoring:
1. Completeness — Whether the assistant fully accomplished the user’s request end-to-end
2. Conciseness — Whether the assistant solved the task using the minimum necessary steps and verbosity

## Assessment Criteria

### 1. Completeness
**What to Evaluate**: Did the assistant fully satisfy the user’s goal given the conversation context? Consider whether the assistant:
- Executed all required steps end-to-end (including saving/exporting/downloading where applicable)
- Provided the final deliverable or a working alternative when blocked (e.g., tool failure with a usable fallback)
- Included essential confirmations, paths, or instructions to achieve the outcome
- Avoided missing key requirements or leaving the user with unresolved gaps

**Rating Guidelines**:
- very incomplete: Major requirements missing; no usable outcome
- incomplete: Some key requirements missing; outcome is not directly usable
- partially complete: Core steps attempted; outcome usable only with user effort or missing minor requirements
- mostly complete: Meets most requirements; small omissions or minor issues remain
- fully complete: All requirements met with a usable outcome delivered

### 2. Conciseness
**What to Evaluate**: Did the assistant achieve the goal with minimal redundancy and steps? Consider whether the assistant:
- Avoided repetitive or unnecessary explanations/tool calls
- Used the minimal set of steps/tools to complete the task
- Kept language concise while preserving clarity

**Rating Guidelines**:
- very redundant: Excessive repetition or unnecessary steps/tool calls
- redundant: Noticeable verbosity or extra steps beyond what’s needed
- average: Reasonably concise with minor extraneous content
- concise: Efficient and to the point with minimal overhead
- very concise: Maximally efficient while clear and complete

## Response Analysis

### Question Content
```
{QUESTION_CONTENT}
```

### Intended Tool for This Question
```
{INTENDED_TOOL}
```

### Conversation History
```
{CONVERSATION_HISTORY}
```

## Output Requirements
- Provide detailed reasoning BEFORE ratings for Completeness and Conciseness
- Do NOT score Tool Call Accuracy; include placeholders only

## Output
Provide your response in the following XML format:

<response>
  <completeness>
    <reasoning>
      <!-- Evaluate if the assistant delivered an end-to-end usable outcome, addressed all requirements, handled tool failures with alternatives, and provided necessary confirmations/paths. -->
    </reasoning>
    <rating><!-- Rating: very incomplete, incomplete, partially complete, mostly complete, fully complete --></rating>
  </completeness>

  <conciseness>
    <reasoning>
      <!-- Evaluate if the assistant minimized redundant steps/explanations, avoided unnecessary tool calls, and kept messaging efficient while clear. -->
    </reasoning>
    <rating><!-- Rating: very redundant, redundant, average, concise, very concise --></rating>
  </conciseness>
</response>