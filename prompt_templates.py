# This file contains the prompt templates for the feedback analyzer

SENTIMENT_ANALYSIS_PROMPT = """
Perform sentiment analysis on the following customer feedback:

CUSTOMER FEEDBACK:
{feedback}

Please provide a detailed analysis including:
- Overall sentiment (positive, negative, or mixed)
- Key pain points or concerns
- Specific positive aspects mentioned
- Emotion detection
- Priority level for response
"""

RESPONSE_GENERATION_PROMPT = """
Generate a {tone} response to the customer feedback below.

CUSTOMER FEEDBACK:
{feedback}

SENTIMENT ANALYSIS:
{analysis}

ADDITIONAL INSIGHTS TO INCORPORATE:
{insights}

Your response MUST directly address and quote the points mentioned in the ADDITIONAL INSIGHTS section.
Create a {tone} response that acknowledges the customer's feedback and addresses their concerns.
Make sure to explicitly reference the additional insights provided.
"""
