CONCEPTS_EXTRACTION = """
Extract key concepts from the following text in a concise, context-specific manner. Include only highly relevant and specific concepts.

The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}, "required": ["foo"]}} the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
{{
  "properties": {{
    "concepts": {{
      "description": "List of key concepts extracted from the text.",
      "items": {{
        "type": "string"
      }},
      "title": "Concepts",
      "type": "array"
    }}
  }},
  "required": ["concepts"]
}}

Text: {text}
"""