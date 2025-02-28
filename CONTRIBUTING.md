# Contributing to TurnSense

Thank you for your interest in improving TurnSense! This document provides guidelines for contributing conversation examples to help improve the model's performance.

## Why Your Contributions Matter

End-of-utterance detection is highly context-dependent and varies across different conversation styles, domains, and speaker patterns. By contributing diverse examples, especially those where the model performs poorly, you help make TurnSense more robust and universally applicable.

## How to Contribute Conversation Examples

### Example Format

Please format your examples as JSON objects following this structure:

```json
{
  "context": [
    {
      "role": "user",
      "content": "I want to plan a beach vacation"
    },
    {
      "role": "assistant",
      "content": "Sounds great! Which beach destination are you considering?"
    },
    {
      "role": "user",
      "content": "Maybe somewhere in Florida or California"
    }
  ],
  "label": 0,
  "metadata": {
    "domain": "travel planning",
    "speaker_style": "decisive",
    "task_oriented": true,
    "notes": "User is still considering options, clearly not done speaking"
  }
}
```

### Fields Explanation

- **context**: An array of conversation turns, with at least 3 turns (minimum 1 user, 1 assistant, 1 user)
  - **role**: Either "user" or "assistant"
  - **content**: The actual text of the utterance

- **label**: Binary classification
  - **0**: NON_EOU (user is likely to continue speaking)
  - **1**: EOU (user has finished their utterance)

- **metadata** (optional but highly valuable):
  - **domain**: The topic/domain of conversation (e.g., "technical support", "casual chat", "medical")
  - **speaker_style**: Description of speaking pattern (e.g., "hesitant", "confident", "technical")
  - **task_oriented**: Boolean indicating if the conversation is task-oriented
  - **notes**: Any additional observations about why this example is challenging or interesting

### Submission Guidelines

1. **Focus on challenging cases**: We're particularly interested in examples where:
   - The model makes incorrect predictions
   - The end-of-utterance is ambiguous or context-dependent
   - The conversation involves domain-specific language
   - The speaker has unique patterns (hesitations, fillers, etc.)

2. **Include sufficient context**: Always include at least 3 turns of conversation (user-assistant-user)

3. **Provide diverse examples**: Include a variety of:
   - Conversation domains and topics
   - Speaker styles and patterns
   - Complete vs. incomplete utterances
   - Questions vs. statements
   - Emotional vs. neutral speech

## How to Submit

### Option 1: GitHub Pull Request

1. Fork the repository
2. Add your examples to a new JSON file in the `contributions` directory
3. Submit a pull request with a brief description of your examples

### Option 2: Issues

1. Create a new issue titled "Conversation Examples Contribution"
2. Paste your JSON-formatted examples in the issue description
3. Add any additional context about the examples

### Option 3: Email

If you prefer, you can email your examples directly to [contact@example.com](mailto:contact@example.com) with the subject "TurnSense Contribution".

## Example Collection

Here are some examples of the format we're looking for:

```json
[
  {
    "context": [
      {
        "role": "user",
        "content": "I want to plan a beach vacation"
      },
      {
        "role": "assistant",
        "content": "Sounds great! Which beach destination are you considering?"
      },
      {
        "role": "user",
        "content": "Maybe somewhere in Florida or California"
      }
    ],
    "label": 0,
    "metadata": {
      "domain": "travel planning",
      "speaker_style": "decisive",
      "task_oriented": true,
      "notes": "User is still considering options, clearly not done speaking"
    }
  },
  {
    "context": [
      {
        "role": "user",
        "content": "I need food for a party, about 20 guests."
      },
      {
        "role": "assistant",
        "content": "What type of cuisine are you interested in?"
      },
      {
        "role": "user",
        "content": "Maybe Italian or Mexican, but I'm not sure if I should do a buffet or individual orders"
      }
    ],
    "label": 0,
    "metadata": {
      "domain": "event planning",
      "speaker_style": "uncertain",
      "task_oriented": true,
      "notes": "User is expressing uncertainty and considering options"
    }
  },
  {
    "context": [
      {
        "role": "user",
        "content": "My computer keeps crashing when I open this application"
      },
      {
        "role": "assistant",
        "content": "That sounds frustrating. What application is causing the crashes?"
      },
      {
        "role": "user",
        "content": "It's Photoshop."
      }
    ],
    "label": 1,
    "metadata": {
      "domain": "technical support",
      "speaker_style": "direct",
      "task_oriented": true,
      "notes": "Short, definitive answer indicating completion"
    }
  },
  {
    "context": [
      {
        "role": "user",
        "content": "I'm trying to understand how to invest my savings"
      },
      {
        "role": "assistant",
        "content": "That's a great step to take. What are your financial goals?"
      },
      {
        "role": "user",
        "content": "Well, I'm hoping to buy a house in about five years, but also want to have some money for retirement, and maybe some shorter term goals too, um..."
      }
    ],
    "label": 0,
    "metadata": {
      "domain": "financial advice",
      "speaker_style": "hesitant",
      "task_oriented": true,
      "notes": "User trails off with 'um' and is likely to continue speaking"
    }
  }
]
```

## Thank You!

Your contributions help make TurnSense better for everyone. We appreciate your time and effort in helping improve this model.
