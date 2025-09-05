from marshmallow import Schema, fields


class AskSchema(Schema):
    question = fields.Str(
        required=True,
        error_messages={
            "required": "field 'question' is required.",
            "null": "field 'question' cannot be null.",
            "invalid": "field 'question' must be a string.",
        },
    )
    user_id = fields.Int(
        required=True,
        error_messages={
            "required": "field 'user_id' is required.",
            "null": "field 'user_id' cannot be null.",
            "invalid": "field 'user_id' must be a integer.",
        },
    )
    session_id = fields.Int(
        required=True,
        error_messages={
            "required": "field 'user_id' is required.",
            "null": "field 'user_id' cannot be null.",
            "invalid": "field 'user_id' must be a integer.",
        },
    )
