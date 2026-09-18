// Generated from ../python/decision_lab/schemas by scripts/build.mjs.
export const requestSchema = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Jev-shaped local request; runtime enforces loaded-model limits",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "model",
    "state",
    "questions"
  ],
  "properties": {
    "model": {
      "type": "string",
      "enum": [
        "decision-lab-v2",
        "jev-latest",
        "decision-lab-v3"
      ]
    },
    "state": {
      "oneOf": [
        {
          "type": "string"
        },
        {
          "type": "object"
        },
        {
          "type": "array"
        }
      ]
    },
    "questions": {
      "type": "object",
      "minProperties": 1,
      "maxProperties": 32,
      "additionalProperties": {
        "oneOf": [
          {
            "type": "object",
            "additionalProperties": false,
            "required": [
              "type",
              "instructions",
              "criteria"
            ],
            "properties": {
              "type": {
                "const": "choice"
              },
              "instructions": {
                "$ref": "#/$defs/text"
              },
              "criteria": {
                "type": "object",
                "minProperties": 2,
                "maxProperties": 8,
                "additionalProperties": {
                  "anyOf": [
                    {
                      "$ref": "#/$defs/text"
                    },
                    {
                      "type": "null"
                    }
                  ]
                }
              }
            }
          },
          {
            "type": "object",
            "additionalProperties": false,
            "required": [
              "type",
              "instructions",
              "criteria"
            ],
            "properties": {
              "type": {
                "const": "score"
              },
              "instructions": {
                "$ref": "#/$defs/text"
              },
              "criteria": {
                "type": "array",
                "minItems": 2,
                "maxItems": 8,
                "items": {
                  "$ref": "#/$defs/text"
                }
              }
            }
          },
          {
            "type": "object",
            "additionalProperties": false,
            "required": [
              "type",
              "instructions"
            ],
            "properties": {
              "type": {
                "const": "noul"
              },
              "instructions": {
                "$ref": "#/$defs/text"
              },
              "criteria": {
                "type": "object",
                "additionalProperties": false,
                "required": [
                  "true",
                  "false"
                ],
                "properties": {
                  "true": {
                    "type": "string"
                  },
                  "false": {
                    "type": "string"
                  }
                }
              }
            }
          }
        ]
      }
    }
  },
  "$defs": {
    "text": {
      "oneOf": [
        {
          "type": "string",
          "minLength": 1
        },
        {
          "type": "object"
        },
        {
          "type": "array"
        }
      ]
    }
  }
};
export const responseSchema = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Jev-shaped local response",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "model",
    "answers",
    "usage"
  ],
  "properties": {
    "model": {
      "enum": [
        "decision-lab-v2",
        "decision-lab-v3"
      ]
    },
    "answers": {
      "type": "object",
      "minProperties": 1,
      "additionalProperties": {
        "oneOf": [
          {
            "type": "object",
            "additionalProperties": false,
            "required": [
              "type",
              "choice",
              "probabilities",
              "confidence"
            ],
            "properties": {
              "type": {
                "const": "choice"
              },
              "choice": {
                "type": "string"
              },
              "probabilities": {
                "$ref": "#/$defs/probabilities"
              },
              "confidence": {
                "$ref": "#/$defs/probability"
              }
            }
          },
          {
            "type": "object",
            "additionalProperties": false,
            "required": [
              "type",
              "score",
              "legend",
              "probabilities",
              "confidence"
            ],
            "properties": {
              "type": {
                "const": "score"
              },
              "score": {
                "type": "number",
                "minimum": 0,
                "maximum": 7
              },
              "legend": {
                "type": "object",
                "additionalProperties": {}
              },
              "probabilities": {
                "$ref": "#/$defs/probabilities"
              },
              "confidence": {
                "$ref": "#/$defs/probability"
              }
            }
          },
          {
            "type": "object",
            "additionalProperties": false,
            "required": [
              "type",
              "noul"
            ],
            "properties": {
              "type": {
                "const": "noul"
              },
              "noul": {
                "$ref": "#/$defs/probability"
              }
            }
          }
        ]
      }
    },
    "usage": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "input_tokens",
        "output_tokens"
      ],
      "properties": {
        "input_tokens": {
          "type": "integer",
          "minimum": 0
        },
        "output_tokens": {
          "type": "integer",
          "minimum": 0
        }
      }
    }
  },
  "$defs": {
    "probability": {
      "type": "number",
      "minimum": 0,
      "maximum": 1
    },
    "probabilities": {
      "type": "object",
      "minProperties": 2,
      "maxProperties": 8,
      "additionalProperties": {
        "$ref": "#/$defs/probability"
      }
    }
  }
};
