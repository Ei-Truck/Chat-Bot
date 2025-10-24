from marshmallow import Schema, fields


# Define o schema para validação dos dados enviados pelo usuário
class AskSchema(Schema):
    # Campo obrigatório 'question' que deve ser uma string
    question = fields.Str(
        required=True,  # Obrigatório
        error_messages={
            "required": "field 'question' is required.",  # Mensagem se não enviado
            "null": "field 'question' cannot be null.",  # Mensagem se for nulo
            "invalid": "field 'question' must be a string.",  # Mensagem se não for string
        },
    )

    # Campo obrigatório 'user_id' que deve ser um inteiro
    user_id = fields.Int(
        required=True,
        error_messages={
            "required": "field 'user_id' is required.",
            "null": "field 'user_id' cannot be null.",
            "invalid": "field 'user_id' must be a integer.",
        },
    )

    # Campo obrigatório 'session_id' que também deve ser um inteiro
    session_id = fields.Int(
        required=True,
        error_messages={
            "required": "field 'user_id' is required.",  # Atenção: aqui a mensagem repetida de 'user_id', provavelmente deveria ser 'session_id'
            "null": "field 'user_id' cannot be null.",  # Mesma observação acima
            "invalid": "field 'user_id' must be a integer.",  # Mesma observação
        },
    )
