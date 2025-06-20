from marshmallow import Schema, fields

class AskSchema(Schema):
    question = fields.Str(required=True, error_messages={
        "required": "field 'question' is required.",
        "null": "field 'question' cannot be null.",
        "invalid": "field 'question' must be a string."
    })