## Task
Conduct a **Question Quality Assessment** of a tool use question across six key dimensions to ensure it meets high standards for realistic tool usage scenarios.

## Objective
Analyze the provided tool use question and assess its quality across six primary dimensions:
1. **Tool Selection Difficulty** - How challenging it is to determine which tools to use giving all available tools
2. **Tool Selection Uniqueness** - How unique and necessary the selected tools are for this specific task giving all available tools
3. **Question Quality** - Overall clarity, specificity, and effectiveness 
4. **Scenario Realism** - How authentic and believable the scenario is
5. **Verifiable** - How easy it is to verify the correctness of the final model answer
6. **Stability** - How stable the answer will be when requested under different time and geolocation

## Assessment Criteria

### 1. Tool Selection Difficulty
**What to Evaluate**: How difficult it would be for a user to determine which specific tools are needed to solve this question.

**Rating Guidelines**:
- **very easy**: Question explicitly mentions tool names or makes tool selection obvious
- **easy**: Tool selection is straightforward with clear indicators
- **medium**: Requires some reasoning but tool needs are fairly apparent  
- **hard**: Requires careful analysis to determine appropriate tools
- **very hard**: Requires extensive expertise and deep reasoning to identify the correct tools

### 2. Tool Selection Uniqueness
**What to Evaluate**: How unique and necessary the selected tools are for accomplishing this specific task, and whether the task can only be completed with these tools in the specified sequence.

**Rating Guidelines**:
- **not unique**: Many alternative tool combinations could accomplish the same task equally well
- **somewhat unique**: Some alternative approaches exist, but selected tools offer advantages
- **moderately unique**: Selected tools are well-suited, with limited alternative approaches
- **quite unique**: Selected tools are particularly well-matched to the task requirements
- **highly unique**: Task can only be accomplished effectively with these specific tools in this sequence

### 3. Question Quality
**What to Evaluate**: Overall quality, clarity, and effectiveness of the question as a realistic user query.

**Rating Guidelines**:
- **very poor**: Unclear, ambiguous, or poorly constructed question
- **poor**: Some clarity issues, missing important context
- **average**: Clear and understandable, but could be more specific or engaging
- **good**: Well-constructed, clear, specific, and realistic
- **excellent**: Exceptionally clear, detailed, engaging, and professionally written

### 4. Scenario Realism
**What to Evaluate**: How authentic, believable, and true-to-life the described scenario is.

**Rating Guidelines**:
- **unrealistic**: Artificial, contrived, or implausible scenario
- **somewhat unrealistic**: Some realistic elements but feels forced or unlikely
- **moderately realistic**: Believable scenario with minor authenticity issues
- **realistic**: Authentic scenario that represents genuine use cases
- **highly realistic**: Completely natural, authentic scenario indistinguishable from real user needs

### 5. Verifiable
**What to Evaluate**: How easy it is to verify the correctness of the final model answer.

**Rating Guidelines**:
- **hard to verify**: Fully free-form answer that requires extensive human judgment
- **somewhat hard**: Mostly subjective answer with some verifiable elements
- **moderately verifiable**: Short sentence that can be verified by LLM comparison
- **mostly verifiable**: Answer with clear, objective components and some subjective elements
- **easy to verify**: Answer can be verified by simple rules, exact matches, or clear success criteria

### 6. Stability (1-5 Scale)
**What to Evaluate**: How stable and consistent the answer will be when the question is asked under different environmental conditions and system contexts. Consider factors like temporal dependency, geographical variations, operating system differences, network environments, and software version variations.

**Rating Guidelines**:
- **highly unstable**: Answer changes significantly across different conditions (real-time data, location-specific, system-dependent)
- **somewhat unstable**: Answer may vary moderately based on environmental or system factors
- **moderately stable**: Answer mostly consistent with minor variations due to context
- **mostly stable**: Answer remains largely consistent across different conditions
- **highly stable**: Answer is completely independent of environmental and system factors

## Question Analysis

### All Available Tools```
{ALL_SERVER_AND_TOOL_INFORMATION}
```

### Question Content
```
{QUESTION_CONTENT}
```

### Intended Tool for This Question
```
{INTENDED_TOOL}
```

## Output Requirements

Provide analysis with detailed reasoning BEFORE scores for each of the six metrics.

## Output
Provide your response in the following XML format:

<response>
  <tool_selection_difficulty>
    <reasoning>
      <!-- Detailed explanation including ambiguity level, domain knowledge required, and alternative solutions giving all available tools -->
    </reasoning>
    <rating><!-- Rating: very easy, easy, medium, hard, very hard --></rating>
  </tool_selection_difficulty>
  
  <tool_selection_uniqueness>
    <reasoning>
      <!-- Detailed explanation of tool necessity, sequential dependencies, and alternative tool viability giving all available tools -->
    </reasoning>
    <rating><!-- Rating: not unique, somewhat unique, moderately unique, quite unique, highly unique --></rating>
  </tool_selection_uniqueness>
  
  <question_quality>
    <reasoning>
      <!-- Detailed explanation covering linguistic quality, information architecture, and actionability -->
    </reasoning>
    <rating><!-- Rating: very poor, poor, average, good, excellent --></rating>
  </question_quality>
  
  <scenario_realism>
    <reasoning>
      <!-- Detailed explanation of industry authenticity, workflow accuracy, and stakeholder behavior -->
    </reasoning>
    <rating><!-- Rating: unrealistic, somewhat unrealistic, moderately realistic, realistic, highly realistic --></rating>
  </scenario_realism>
  
  <verifiable>
    <reasoning>
      <!-- Detailed explanation of answer format, objective criteria, and ground truth availability -->
    </reasoning>
    <rating><!-- Rating: hard to verify, somewhat hard, moderately verifiable, mostly verifiable, easy to verify --></rating>
  </verifiable>
  
  <stability>
    <reasoning>
      <!-- Detailed explanation of temporal/geographical/system dependencies and environmental factors -->
    </reasoning>
    <rating><!-- Rating: highly unstable, somewhat unstable, moderately stable, mostly stable, highly stable --></rating>
  </stability>
</response> 
