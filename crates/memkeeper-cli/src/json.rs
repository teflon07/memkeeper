//! Order-preserving JSON value model and field-extraction helpers shared
//! by the CLI request parsers and the serve protocol.

#[allow(clippy::wildcard_imports)]
use super::*;

pub(crate) fn optional_raw_json_alias_field(
    object: &JsonObject,
    first: &str,
    second: &str,
) -> Result<Option<String>, CliError> {
    let first_value = optional_raw_json_field(object, first)?;
    let second_value = optional_raw_json_field(object, second)?;
    match (first_value, second_value) {
        (Some(_), Some(_)) => Err(CliError::InvalidRequest(format!(
            "only one of {first} or {second} may be supplied"
        ))),
        (Some(value), None) | (None, Some(value)) => Ok(Some(value)),
        (None, None) => Ok(None),
    }
}

pub(crate) fn optional_raw_json_field(
    object: &JsonObject,
    key: &str,
) -> Result<Option<String>, CliError> {
    match object.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::Object(value)) => Ok(Some(value.to_json())),
        Some(JsonValue::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(CliError::InvalidRequest(format!(
            "field {key} must be a JSON object or string"
        ))),
    }
}

pub(crate) fn input_path_field(object: &JsonObject) -> Result<PathBuf, CliError> {
    let input = optional_string_field(object, "input")?;
    let r#in = optional_string_field(object, "in")?;
    match (input, r#in) {
        (Some(_), Some(_)) => Err(CliError::InvalidRequest(
            "only one of input or in may be supplied".to_string(),
        )),
        (Some(path), None) | (None, Some(path)) => Ok(PathBuf::from(path)),
        (None, None) => Err(CliError::InvalidRequest(
            "missing required input path field".to_string(),
        )),
    }
}

pub(crate) fn output_path_field(object: &JsonObject) -> Result<PathBuf, CliError> {
    let output = optional_string_field(object, "output")?;
    let out = optional_string_field(object, "out")?;
    match (output, out) {
        (Some(_), Some(_)) => Err(CliError::InvalidRequest(
            "only one of output or out may be supplied".to_string(),
        )),
        (Some(path), None) | (None, Some(path)) => Ok(PathBuf::from(path)),
        (None, None) => Err(CliError::InvalidRequest(
            "missing required output path field".to_string(),
        )),
    }
}

pub(crate) fn reject_unknown_fields(object: &JsonObject, allowed: &[&str]) -> Result<(), CliError> {
    for key in object.keys() {
        if !allowed.contains(&key) {
            return Err(CliError::InvalidRequest(format!("unknown field: {key}")));
        }
    }
    Ok(())
}

pub(crate) fn required_string_field(object: &JsonObject, key: &str) -> Result<String, CliError> {
    optional_string_field(object, key)?
        .ok_or_else(|| CliError::InvalidRequest(format!("missing required string field: {key}")))
}

pub(crate) fn optional_string_field(
    object: &JsonObject,
    key: &str,
) -> Result<Option<String>, CliError> {
    match object.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(CliError::InvalidRequest(format!(
            "field {key} must be a string"
        ))),
    }
}

pub(crate) fn optional_bool_field(
    object: &JsonObject,
    key: &str,
) -> Result<Option<bool>, CliError> {
    match object.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(CliError::InvalidRequest(format!(
            "field {key} must be a boolean"
        ))),
    }
}

pub(crate) fn optional_number_field(
    object: &JsonObject,
    key: &str,
) -> Result<Option<f64>, CliError> {
    match object.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::Number(value)) => {
            let parsed = value.parse::<f64>().map_err(|_| {
                CliError::InvalidRequest(format!("field {key} must be a finite number"))
            })?;
            if parsed.is_finite() {
                Ok(Some(parsed))
            } else {
                Err(CliError::InvalidRequest(format!(
                    "field {key} must be a finite number"
                )))
            }
        }
        Some(_) => Err(CliError::InvalidRequest(format!(
            "field {key} must be a number"
        ))),
    }
}

pub(crate) fn optional_usize_field(
    object: &JsonObject,
    key: &str,
) -> Result<Option<usize>, CliError> {
    match object.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::Number(value)) => value.parse::<usize>().map(Some).map_err(|_| {
            CliError::InvalidRequest(format!("field {key} must be a non-negative integer"))
        }),
        Some(_) => Err(CliError::InvalidRequest(format!(
            "field {key} must be an integer"
        ))),
    }
}

pub(crate) fn optional_number_array_field(
    object: &JsonObject,
    key: &str,
) -> Result<Option<Vec<f32>>, CliError> {
    match object.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::Array(values)) => values
            .iter()
            .map(|value| match value {
                JsonValue::Number(number) => {
                    let parsed = number.parse::<f32>().map_err(|_| {
                        CliError::InvalidRequest(format!(
                            "field {key} must contain only finite numbers"
                        ))
                    })?;
                    if parsed.is_finite() {
                        Ok(parsed)
                    } else {
                        Err(CliError::InvalidRequest(format!(
                            "field {key} must contain only finite numbers"
                        )))
                    }
                }
                _ => Err(CliError::InvalidRequest(format!(
                    "field {key} must contain only numbers"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(_) => Err(CliError::InvalidRequest(format!(
            "field {key} must be an array"
        ))),
    }
}

pub(crate) fn optional_string_array_field(
    object: &JsonObject,
    key: &str,
) -> Result<Option<Vec<String>>, CliError> {
    match object.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::Array(values)) => values
            .iter()
            .map(|value| match value {
                JsonValue::String(string) => Ok(string.clone()),
                _ => Err(CliError::InvalidRequest(format!(
                    "field {key} must contain only strings"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(_) => Err(CliError::InvalidRequest(format!(
            "field {key} must be an array"
        ))),
    }
}

pub(crate) fn required_string_array_field(
    object: &JsonObject,
    key: &str,
) -> Result<Vec<String>, CliError> {
    optional_string_array_field(object, key)?
        .ok_or_else(|| CliError::InvalidRequest(format!("missing required array field: {key}")))
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum JsonValue {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<JsonValue>),
    Object(JsonObject),
}

impl JsonValue {
    pub(crate) fn as_object(&self) -> Option<&JsonObject> {
        match self {
            Self::Object(object) => Some(object),
            Self::Null | Self::Bool(_) | Self::Number(_) | Self::String(_) | Self::Array(_) => None,
        }
    }

    pub(crate) fn to_json(&self) -> String {
        match self {
            Self::Null => "null".to_string(),
            Self::Bool(value) => value.to_string(),
            Self::Number(value) => value.clone(),
            Self::String(value) => json_string(value),
            Self::Array(values) => {
                let mut output = String::from("[");
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        output.push(',');
                    }
                    output.push_str(&value.to_json());
                }
                output.push(']');
                output
            }
            Self::Object(object) => object.to_json(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct JsonObject(Vec<(String, JsonValue)>);

impl JsonObject {
    pub(crate) fn get(&self, key: &str) -> Option<&JsonValue> {
        self.0
            .iter()
            .rev()
            .find_map(|(candidate, value)| (candidate == key).then_some(value))
    }

    pub(crate) fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(key, _)| key.as_str())
    }

    /// Overwrite (or append) a key. Used to force a server-side flag onto a
    /// client-supplied payload (e.g. the read-only dashboard suppressing
    /// recall-telemetry writes regardless of what the client sent).
    pub(crate) fn set(&mut self, key: &str, value: JsonValue) {
        self.0.retain(|(candidate, _)| candidate != key);
        self.0.push((key.to_string(), value));
    }

    pub(crate) fn to_json(&self) -> String {
        let mut output = String::from("{");
        for (index, (key, value)) in self.0.iter().enumerate() {
            if index > 0 {
                output.push(',');
            }
            output.push_str(&json_string(key));
            output.push(':');
            output.push_str(&value.to_json());
        }
        output.push('}');
        output
    }
}

impl<'de> serde::Deserialize<'de> for JsonValue {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(JsonValueVisitor)
    }
}

pub(crate) struct JsonValueVisitor;

impl<'de> serde::de::Visitor<'de> for JsonValueVisitor {
    type Value = JsonValue;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a bounded JSON value")
    }

    fn visit_unit<E>(self) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(JsonValue::Null)
    }

    fn visit_none<E>(self) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(JsonValue::Null)
    }

    fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(JsonValue::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(JsonValue::Number(value.to_string()))
    }

    fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(JsonValue::Number(value.to_string()))
    }

    fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        if value.is_finite() {
            Ok(JsonValue::Number(value.to_string()))
        } else {
            Err(E::custom("number must be finite"))
        }
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(JsonValue::String(value.to_string()))
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(JsonValue::String(value))
    }

    fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = seq.next_element::<JsonValue>()? {
            if values.len() >= MAX_JSON_ARRAY_ITEMS {
                return Err(serde::de::Error::custom("JSON array has too many items"));
            }
            values.push(value);
        }
        Ok(JsonValue::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut pairs = Vec::new();
        let mut seen = BTreeSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if pairs.len() >= MAX_JSON_OBJECT_FIELDS {
                return Err(serde::de::Error::custom("JSON object has too many fields"));
            }
            if !seen.insert(key.clone()) {
                return Err(serde::de::Error::custom(format!(
                    "duplicate JSON object key: {key}"
                )));
            }
            let value = map.next_value::<JsonValue>()?;
            pairs.push((key, value));
        }
        Ok(JsonValue::Object(JsonObject(pairs)))
    }
}

pub(crate) fn parse_json(input: &str) -> Result<JsonValue, CliError> {
    if input.len() > MAX_JSON_BYTES {
        return Err(CliError::InvalidRequest(format!(
            "JSON request must be at most {MAX_JSON_BYTES} bytes"
        )));
    }
    let value = serde_json::from_str::<JsonValue>(input)
        .map_err(|error| CliError::InvalidRequest(format!("invalid JSON: {error}")))?;
    validate_json_depth(&value, 0)?;
    Ok(value)
}

pub(crate) fn validate_json_depth(value: &JsonValue, depth: usize) -> Result<(), CliError> {
    if depth > MAX_JSON_DEPTH {
        return Err(CliError::InvalidRequest(format!(
            "JSON nesting depth must be at most {MAX_JSON_DEPTH}"
        )));
    }
    match value {
        JsonValue::Array(values) => {
            for value in values {
                validate_json_depth(value, depth + 1)?;
            }
        }
        JsonValue::Object(object) => {
            for (_, value) in &object.0 {
                validate_json_depth(value, depth + 1)?;
            }
        }
        JsonValue::Null | JsonValue::Bool(_) | JsonValue::Number(_) | JsonValue::String(_) => {}
    }
    Ok(())
}
